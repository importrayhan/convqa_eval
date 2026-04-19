"""
Minimal Greedy Coordinate Gradient (GCG) attack for ambiguity classification.

Adapted from Zou et al. 2023 "Universal and Transferable Adversarial Attacks
on Aligned Language Models" (https://llm-attacks.org).

Instead of making the model output harmful content, we optimize an
adversarial suffix that forces the LLM to mispredict the ambiguity label
(e.g., flip a "clear" prediction to "ambiguous" or vice versa).

Algorithm (per Zou et al. 2023, Algorithm 1):
  1. Initialize suffix tokens (e.g., "! ! ! ..." of length L)
  2. For each optimization step:
     a. Compute gradient of loss w.r.t. one-hot token embeddings at suffix positions
     b. Select top-k candidate replacements per position (largest negative gradient)
     c. Sample B random single-token substitutions from candidates
     d. Evaluate loss for each substitution
     e. Keep the substitution with lowest loss
  3. Return the optimized suffix
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def token_gradients(model, input_ids, suffix_slice,
                    target_slice, loss_slice):
    """
    Compute gradient of cross-entropy loss w.r.t. one-hot token encodings
    at the suffix positions.

    Returns: gradient tensor [suffix_len, vocab_size]
    """
    import torch
    import torch.nn.functional as F

    embed_weights = model.get_input_embeddings().weight  # [V, d]
    vocab_size = embed_weights.shape[0]

    # Build one-hot for suffix tokens
    suffix_ids = input_ids[suffix_slice]
    one_hot = torch.zeros(
        suffix_ids.shape[0], vocab_size,
        device=model.device, dtype=embed_weights.dtype)
    one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1)
    one_hot.requires_grad_(True)

    # Build embeddings: replace suffix positions with differentiable one-hot @ W
    input_embeds = model.get_input_embeddings()(input_ids.unsqueeze(0))  # [1, T, d]
    suffix_embeds = (one_hot @ embed_weights).unsqueeze(0)  # [1, L, d]
    full_embeds = input_embeds.clone()
    full_embeds[:, suffix_slice, :] = suffix_embeds

    # Forward pass
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits  # [1, T, V]

    # Loss: cross-entropy at target positions
    targets = input_ids[target_slice]
    loss_logits = logits[0, loss_slice, :]
    loss = F.cross_entropy(loss_logits, targets)

    loss.backward()
    return one_hot.grad.clone()  # [suffix_len, V]


def sample_control(control_toks, grad, batch_size, top_k=256,
                   not_allowed_tokens=None):
    """
    Sample candidate suffix replacements based on gradient information.

    For each position, find top-k tokens with most negative gradient,
    then randomly select one position and one token per candidate.
    """
    import torch

    # Top-k candidates per position (most negative gradient = most loss-reducing)
    top_k_ids = (-grad).topk(top_k, dim=1).indices  # [suffix_len, top_k]

    # Random selection: pick one position and one candidate per batch element
    original = control_toks.repeat(batch_size, 1)  # [B, suffix_len]
    positions = torch.randint(0, len(control_toks), (batch_size, 1),
                              device=control_toks.device)
    candidates = torch.gather(
        top_k_ids[positions.squeeze(1)],
        1,
        torch.randint(0, top_k, (batch_size, 1), device=control_toks.device))

    new_controls = original.scatter_(1, positions, candidates)

    # Filter not-allowed tokens
    if not_allowed_tokens is not None:
        for i in range(batch_size):
            for j in range(new_controls.shape[1]):
                if new_controls[i, j].item() in not_allowed_tokens:
                    new_controls[i, j] = control_toks[j]

    return new_controls


def run_gcg_attack(
    model,
    tokenizer,
    prompt_messages: list,
    target_output: str,
    num_steps: int = 50,
    suffix_length: int = 20,
    batch_size: int = 64,
    top_k: int = 256,
    suffix_init: str = "! ",
) -> Tuple[str, List[float]]:
    """
    Run GCG to find an adversarial suffix that makes the model output
    `target_output` when appended to the classification prompt.

    For ambiguity attacks:
      - If the correct label is "0" (clear), set target_output="1" to
        force misprediction to "ambiguous"
      - Vice versa

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        prompt_messages: chat messages (from build_classification_prompt)
        target_output: desired (adversarial) output string
        num_steps: optimization iterations
        suffix_length: number of adversarial tokens
        batch_size: candidates evaluated per step
        top_k: top-k gradient candidates per position
        suffix_init: initial suffix token(s) to repeat

    Returns: (best_suffix_string, loss_history)
    """
    import torch

    device = model.device

    # Build the full input text from chat template
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True)

    # Tokenize prompt
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids[0].to(device)

    # Initialize suffix tokens
    suffix_token_id = tokenizer(suffix_init, add_special_tokens=False).input_ids[0]
    suffix_ids = torch.full((suffix_length,), suffix_token_id,
                            dtype=torch.long, device=device)

    # Tokenize target
    target_ids = tokenizer(target_output, return_tensors="pt",
                           add_special_tokens=False).input_ids[0].to(device)

    # Get non-ASCII tokens to exclude
    ascii_range = set(range(256))
    not_allowed = set()
    for i in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([i])
        if not decoded or not all(ord(c) < 128 for c in decoded):
            not_allowed.add(i)

    loss_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    for step in range(num_steps):
        # Build full input: prompt + suffix + target
        input_ids = torch.cat([prompt_ids, suffix_ids, target_ids])

        suffix_slice = slice(len(prompt_ids), len(prompt_ids) + suffix_length)
        target_slice = slice(len(prompt_ids) + suffix_length,
                             len(prompt_ids) + suffix_length + len(target_ids))
        loss_slice = slice(len(prompt_ids) + suffix_length - 1,
                           len(prompt_ids) + suffix_length + len(target_ids) - 1)

        # Step 1: compute gradients
        model.zero_grad()
        grad = token_gradients(model, input_ids, suffix_slice,
                               target_slice, loss_slice)

        # Step 2: sample candidate replacements
        candidates = sample_control(
            suffix_ids, grad, batch_size, top_k, not_allowed)

        # Step 3: evaluate each candidate
        with torch.no_grad():
            losses = []
            for i in range(0, batch_size, min(batch_size, 32)):
                batch = candidates[i:i + 32]
                batch_inputs = []
                for b in range(batch.shape[0]):
                    inp = torch.cat([prompt_ids, batch[b], target_ids])
                    batch_inputs.append(inp)

                # Pad and stack
                max_len = max(x.shape[0] for x in batch_inputs)
                padded = torch.zeros(len(batch_inputs), max_len,
                                     dtype=torch.long, device=device)
                attn = torch.zeros_like(padded)
                for j, inp in enumerate(batch_inputs):
                    padded[j, :len(inp)] = inp
                    attn[j, :len(inp)] = 1

                outputs = model(input_ids=padded, attention_mask=attn)
                logits = outputs.logits

                for j in range(len(batch_inputs)):
                    ls = slice(
                        len(prompt_ids) + suffix_length - 1,
                        len(prompt_ids) + suffix_length + len(target_ids) - 1)
                    tgt = target_ids
                    batch_loss = torch.nn.functional.cross_entropy(
                        logits[j, ls, :], tgt)
                    losses.append(batch_loss.item())

        # Step 4: select best
        best_idx = np.argmin(losses)
        current_loss = losses[best_idx]
        suffix_ids = candidates[best_idx]

        if current_loss < best_loss:
            best_loss = current_loss
            best_suffix = suffix_ids.clone()

        loss_history.append(current_loss)

        if step % 10 == 0:
            suffix_str = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            log.info(f"  GCG step {step:3d}  loss={current_loss:.4f}  "
                     f"suffix=\"{suffix_str[:40]}...\"")

    best_suffix_str = tokenizer.decode(best_suffix, skip_special_tokens=True)
    return best_suffix_str, loss_history


def run_gcg_batch(
    model,
    tokenizer,
    prompt_messages_list: List[list],
    target_output: str,
    num_steps: int = 100,
    suffix_length: int = 20,
    batch_size: int = 64,
    top_k: int = 256,
    suffix_init: str = "! ",
) -> Tuple[str, List[float], List[int]]:
    """
    Batch GCG: optimize a UNIVERSAL adversarial suffix across multiple
    prompts (Algorithm 2 from Zou et al. 2023).

    The suffix is optimized to flip predictions on ALL prompts, not just one.
    Gradients from each prompt are aggregated before candidate selection.

    Args:
        prompt_messages_list: list of chat message lists (one per sample)
        target_output: desired adversarial output (e.g. "1")
        Others: same as run_gcg_attack

    Returns: (best_suffix_string, loss_history, successes_per_step)
        successes_per_step: how many prompts produce target_output at each step
    """
    import torch

    device = model.device

    # Pre-tokenize all prompts
    prompt_ids_list = []
    for msgs in prompt_messages_list:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False).input_ids[0].to(device)
        prompt_ids_list.append(ids)

    # Initialize suffix
    suffix_token_id = tokenizer(suffix_init, add_special_tokens=False).input_ids[0]
    suffix_ids = torch.full((suffix_length,), suffix_token_id,
                            dtype=torch.long, device=device)

    target_ids = tokenizer(target_output, return_tensors="pt",
                           add_special_tokens=False).input_ids[0].to(device)

    # Build not-allowed set
    not_allowed = set()
    for i in range(min(tokenizer.vocab_size, 50000)):
        try:
            decoded = tokenizer.decode([i])
            if not decoded or not all(ord(c) < 128 for c in decoded):
                not_allowed.add(i)
        except Exception:
            not_allowed.add(i)

    loss_history = []
    successes_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()
    n_prompts = len(prompt_ids_list)

    for step in range(num_steps):
        # Aggregate gradients across all prompts
        agg_grad = None

        for prompt_ids in prompt_ids_list:
            input_ids = torch.cat([prompt_ids, suffix_ids, target_ids])
            suf_start = len(prompt_ids)
            suffix_slice = slice(suf_start, suf_start + suffix_length)
            target_slice = slice(suf_start + suffix_length,
                                 suf_start + suffix_length + len(target_ids))
            loss_slice = slice(suf_start + suffix_length - 1,
                               suf_start + suffix_length + len(target_ids) - 1)

            model.zero_grad()
            grad = token_gradients(model, input_ids, suffix_slice,
                                   target_slice, loss_slice)
            if agg_grad is None:
                agg_grad = grad
            else:
                agg_grad = agg_grad + grad

        agg_grad = agg_grad / n_prompts

        # Sample candidates using aggregated gradient
        candidates = sample_control(
            suffix_ids, agg_grad, batch_size, top_k, not_allowed)

        # Evaluate each candidate across ALL prompts
        with torch.no_grad():
            total_losses = np.zeros(batch_size)
            for ci in range(0, batch_size, min(batch_size, 16)):
                batch = candidates[ci:ci + 16]
                actual_bs = batch.shape[0]
                for prompt_ids in prompt_ids_list:
                    batch_inputs = []
                    for b in range(actual_bs):
                        inp = torch.cat([prompt_ids, batch[b], target_ids])
                        batch_inputs.append(inp)
                    max_len = max(x.shape[0] for x in batch_inputs)
                    padded = torch.zeros(actual_bs, max_len,
                                         dtype=torch.long, device=device)
                    attn = torch.zeros_like(padded)
                    for j, inp in enumerate(batch_inputs):
                        padded[j, :len(inp)] = inp
                        attn[j, :len(inp)] = 1
                    outputs = model(input_ids=padded, attention_mask=attn)
                    suf_start = len(prompt_ids)
                    ls = slice(suf_start + suffix_length - 1,
                               suf_start + suffix_length + len(target_ids) - 1)
                    for j in range(actual_bs):
                        batch_loss = torch.nn.functional.cross_entropy(
                            outputs.logits[j, ls, :], target_ids)
                        total_losses[ci + j] += batch_loss.item()

        # Select best
        avg_losses = total_losses / n_prompts
        best_idx = int(np.argmin(avg_losses))
        current_loss = avg_losses[best_idx]
        suffix_ids = candidates[best_idx]

        if current_loss < best_loss:
            best_loss = current_loss
            best_suffix = suffix_ids.clone()

        loss_history.append(float(current_loss))

        # Count successes
        n_success = 0
        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
                inp = torch.cat([prompt_ids, suffix_ids, target_ids[:0]])
                gen = model.generate(inp.unsqueeze(0), max_new_tokens=4,
                                     do_sample=False)
                gen_text = tokenizer.decode(gen[0][len(inp):],
                                            skip_special_tokens=True).strip()
                if gen_text.startswith(target_output):
                    n_success += 1
        successes_history.append(n_success)

        if step % 10 == 0:
            suffix_str = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            log.info(f"  GCG-batch step {step:3d}  avg_loss={current_loss:.4f}  "
                     f"success={n_success}/{n_prompts}  "
                     f"suffix=\"{suffix_str[:40]}...\"")

    best_suffix_str = tokenizer.decode(best_suffix, skip_special_tokens=True)
    return best_suffix_str, loss_history, successes_history
