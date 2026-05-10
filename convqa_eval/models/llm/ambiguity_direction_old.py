"""
Ambiguity Direction Analysis — finding the single feature vector that
determines whether a query is classified as ambiguous.

Inspired by:
  - Arditi et al. 2024 "Refusal in Language Models Is Mediated by a
    Single Direction" (https://arxiv.org/abs/2406.11717)
  - p-e-w/heretic: directional ablation for censorship removal

Key idea: Just as refusal is mediated by a single direction in the
residual stream, ambiguity detection may also be a linear feature.
We extract hidden states from the LLM for "clear" vs "ambiguous" prompts,
compute the mean-difference direction, and test whether projecting it
out changes the model's predictions.

Analysis outputs a layerwise cosine similarity plot (4 scenarios):
  1. ambiguous queries — should have HIGH cosine with ambiguity direction
  2. clear queries — should have LOW cosine
  3. ambiguous + GCG adversarial suffix — should SUPPRESS the direction
  4. ambiguous + random suffix — should remain HIGH

Functions:
  extract_hidden_states  — collect per-layer hidden states for a batch
  compute_ambiguity_direction — mean-difference direction (clear vs ambig)
  probe_accuracy — linear probe accuracy at each layer
  ablate_direction — remove the direction and re-run inference
  compute_layerwise_cosine_similarity — cos(activation, direction) per layer
  plot_cosine_similarity_layers — the 4-scenario reference plot
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def extract_hidden_states(
    model, tokenizer,
    messages_list: List[List[Dict]],
    layers: List[int] = None,
    pool: str = "last",
) -> Dict[int, np.ndarray]:
    """
    Extract hidden states from the model for a batch of chat prompts.

    Args:
        model: HuggingFace causal LM
        tokenizer: tokenizer
        messages_list: list of chat message lists
        layers: which layers to extract (None = all)
        pool: "last" (last token), "mean" (mean pool), "first" (first token)

    Returns: {layer_idx: np.array [n_samples, hidden_dim]}
    """
    import torch

    if layers is None:
        n_layers = model.config.num_hidden_layers
        layers = list(range(n_layers + 1))  # 0 = embedding, 1..N = layers

    all_hidden = {l: [] for l in layers}

    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of [1, seq_len, d]

        for l in layers:
            if l < len(hidden_states):
                h = hidden_states[l][0]  # [seq_len, d]
                if pool == "last":
                    vec = h[-1]
                elif pool == "mean":
                    vec = h.mean(dim=0)
                elif pool == "first":
                    vec = h[0]
                else:
                    vec = h[-1]
                all_hidden[l].append(vec.cpu().float().numpy())

    return {l: np.stack(vecs) for l, vecs in all_hidden.items() if vecs}


def compute_ambiguity_direction(
    clear_hidden: np.ndarray,
    ambig_hidden: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute the single direction vector that separates clear from
    ambiguous hidden states, following Arditi et al. 2024.

    Method: mean-difference direction.
      d = mean(ambiguous) - mean(clear)
      d_hat = d / ||d||

    This is the simplest approach from the refusal-direction paper.
    More sophisticated alternatives (PCA of difference, logistic
    regression weight vector) are possible but this captures the
    primary axis of variation.

    Args:
        clear_hidden: [n_clear, hidden_dim]
        ambig_hidden: [n_ambig, hidden_dim]

    Returns: (direction_unit_vector [hidden_dim], separation_score)
        separation_score = cosine distance between class means projected
        onto the direction.  Higher = better separated.
    """
    mean_clear = clear_hidden.mean(axis=0)
    mean_ambig = ambig_hidden.mean(axis=0)
    direction = mean_ambig - mean_clear
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return direction, 0.0
    d_hat = direction / norm

    # Separation: how far apart are the class means along this direction?
    proj_clear = clear_hidden @ d_hat
    proj_ambig = ambig_hidden @ d_hat
    separation = (proj_ambig.mean() - proj_clear.mean()) / (
        proj_clear.std() + proj_ambig.std() + 1e-8)

    return d_hat, float(separation)


def probe_accuracy(
    hidden_states: Dict[int, np.ndarray],
    labels: np.ndarray,
) -> Dict[int, float]:
    """
    Train a linear probe (logistic regression) at each layer and
    report accuracy.  This tests whether ambiguity is linearly
    decodable from the hidden states at each layer.

    Returns: {layer_idx: accuracy}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    results = {}
    for layer, X in sorted(hidden_states.items()):
        if len(np.unique(labels)) < 2:
            results[layer] = 0.5
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, labels, cv=min(5, len(labels)), scoring="accuracy")
        results[layer] = float(scores.mean())
    return results


def ablate_direction(
    model, direction: np.ndarray, layer: int, scale: float = 1.0,
):
    """
    Remove the ambiguity direction from a specific layer's output
    by modifying the model weights in-place.

    This is the directional ablation from Arditi et al. 2024:
      For residual stream vector r at the target layer:
        r' = r - scale * (r · d_hat) * d_hat

    Implemented as a forward hook on the target layer.

    Args:
        model: the HuggingFace model (modified in-place)
        direction: unit direction vector [hidden_dim]
        layer: which layer to ablate
        scale: ablation strength (1.0 = full removal, 0.5 = partial)

    Returns: hook handle (call .remove() to undo)
    """
    import torch

    d_hat = torch.tensor(direction, dtype=torch.float32,
                         device=model.device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Project out the direction: h' = h - scale * (h · d) * d
        proj = (hidden @ d_hat).unsqueeze(-1)  # [..., 1]
        hidden_ablated = hidden - scale * proj * d_hat.unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden_ablated,) + output[1:]
        return hidden_ablated

    # Access the target layer
    if hasattr(model, "model"):  # typical for LlamaForCausalLM etc.
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find model layers for hook installation")

    handle = layers[layer].register_forward_hook(hook_fn)
    return handle


def compute_layerwise_cosine_similarity(
    hidden_states: Dict[int, np.ndarray],
    direction_per_layer: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Compute cosine similarity between each sample's last-token activation
    and the ambiguity direction at each layer.

    Args:
        hidden_states: {layer: [n_samples, hidden_dim]}
        direction_per_layer: {layer: direction_vector [hidden_dim]}

    Returns: {layer: [n_samples] cosine similarities}
    """
    results = {}
    for layer in sorted(hidden_states.keys()):
        if layer not in direction_per_layer:
            continue
        X = hidden_states[layer]  # [n, d]
        d = direction_per_layer[layer]  # [d]
        # Cosine similarity: (X @ d) / (||X|| * ||d||)
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-8:
            results[layer] = np.zeros(X.shape[0])
            continue
        x_norms = np.linalg.norm(X, axis=1, keepdims=True)
        x_norms = np.maximum(x_norms, 1e-8)
        cos_sim = (X @ d) / (x_norms.squeeze() * d_norm)
        results[layer] = cos_sim
    return results


def plot_cosine_similarity_layers(
    scenarios: Dict[str, Dict[int, np.ndarray]],
    output_dir: str = "llm_results",
    title: str = "Cosine similarity with ambiguity direction",
) -> str:
    """
    Plot layerwise cosine similarity between last-token residual stream
    activations and the ambiguity direction for four scenarios:
      1. ambiguous queries (expect HIGH similarity)
      2. clear queries (expect LOW similarity)
      3. ambiguous + GCG adversarial suffix (expect REDUCED similarity)
      4. ambiguous + random suffix (expect still HIGH similarity)

    Matches the reference figure from Arditi et al. 2024 / Zou et al. 2023.

    Args:
        scenarios: {scenario_name: {layer: [n_samples] cosine_similarities}}
            Expected keys: "ambiguous", "clear", "ambiguous+adv_suffix",
                           "ambiguous+random_suffix"
        output_dir: where to save the plot
        title: plot title

    Returns: path to saved figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    style_map = {
        "ambiguous":              {"color": "#3F51B5", "label": "ambiguous"},
        "clear":                  {"color": "#4CAF50", "label": "clear"},
        "ambiguous+adv_suffix":   {"color": "#F44336", "label": "ambiguous + adv_suffix"},
        "ambiguous+random_suffix": {"color": "#FF9800", "label": "ambiguous + random_suffix"},
    }

    for scenario_name, layer_cos in scenarios.items():
        layers = sorted(layer_cos.keys())
        means = [float(np.mean(layer_cos[l])) for l in layers]
        stds = [float(np.std(layer_cos[l])) for l in layers]

        style = style_map.get(scenario_name,
                              {"color": "#999", "label": scenario_name})

        ax.plot(layers, means, linewidth=2, label=style["label"],
                color=style["color"])
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=style["color"])

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine similarity with\nambiguity direction", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(layers[0], layers[-1])

    fig.tight_layout()
    save_path = out / "cosine_similarity_layers.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved cosine similarity plot to {save_path}")
    return str(save_path)
