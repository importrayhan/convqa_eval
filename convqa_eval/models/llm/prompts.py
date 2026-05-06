"""
Prompt templates and label parsing for LLM-based ambiguity classification.

Constructs instruction-following prompts from SIP conversations for
zero-shot and few-shot inference.  The LLM sees conversation history
up to (but NOT including) the system response it is classifying.
"""

import re
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Label maps
# ══════════════════════════════════════════════════════════════════════════════
LABEL_NAMES_4 = {
    0: "clear",
    1: "slightly_ambiguous",
    2: "needs_clarification",
    3: "highly_ambiguous",
}

LABEL_NAMES_2 = {
    0: "clear",
    1: "ambiguous",
}

# Reverse maps for parsing
PARSE_MAP_4 = {}
for k, v in LABEL_NAMES_4.items():
    PARSE_MAP_4[str(k)] = k
    PARSE_MAP_4[v] = k
    PARSE_MAP_4[v.replace("_", " ")] = k
# Additional aliases
PARSE_MAP_4.update({"0": 0, "1": 1, "2": 2, "3": 3,
                     "clear": 0, "slightly ambiguous": 1,
                     "needs clarification": 2, "highly ambiguous": 3,
                     "not ambiguous": 0, "ambiguous": 2})

PARSE_MAP_2 = {"0": 0, "1": 1, "clear": 0, "ambiguous": 1,
               "not ambiguous": 0, "not_ambiguous": 0,
               "yes": 1, "no": 0}


SYSTEM_PROMPT_4CLASS = """You are a conversational AI assistant that detects when user queries are ambiguous.
Your task: given a multi-turn conversation between a user and a system, predict the ambiguity level of the NEXT system response. Given the user request, do you think you have a clearly specified search query? 

The ambiguity levels are:
  0 = clear: The user's query is unambiguous and can be answered directly.
  1 = slightly_ambiguous: Minor ambiguity, answer can be multiple but the system can still provide a reasonable answer.
  2 = needs_clarification: The query is ambiguous enough that the system should ask a clarifying question.
  3 = highly_ambiguous: The query is very unclear and the system cannot provide a useful answer without clarification.

You task is not to provide answer, You must respond with ONLY the number (0, 1, 2, or 3). Nothing else."""

SYSTEM_PROMPT_2CLASS = """You are a conversational AI assistant that detects when user queries are ambiguous.
Your task: given a multi-turn conversation between a user and a system, predict the ambiguity level of the NEXT system response. Given the user request, do you think you have a clearly specified search query? 

  0 = clear: The system can answer directly without clarification.
  1 = ambiguous: The query is unclear and the system should ask for clarification.

You must respond with ONLY the number (0 or 1). Nothing else."""


def get_system_prompt(num_classes: int) -> str:
    return SYSTEM_PROMPT_4CLASS if num_classes == 4 else SYSTEM_PROMPT_2CLASS


# ══════════════════════════════════════════════════════════════════════════════
# Prompt construction
# ══════════════════════════════════════════════════════════════════════════════
def build_classification_prompt(
    conversation: Dict,
    target_turn_idx: int,
    num_classes: int = 2,
    include_observations: bool = True,
    adversarial_suffix: str = "",
) -> List[Dict[str, str]]:
    """
    Build a chat-template message list for classifying the ambiguity
    of the system response at target_turn_idx.

    The prompt includes all conversation turns UP TO the target system
    turn (human query, observations) but EXCLUDES the system's actual
    response text — the LLM must predict the ambiguity without seeing it.

    Args:
        conversation: raw SIP conversation dict
        target_turn_idx: index of the gpt turn to classify (in the
                         conversations list, NOT pair index)
        num_classes: 2 or 4
        include_observations: whether to include observation text
        adversarial_suffix: GCG adversarial tokens appended to the
                            classification query (for robustness testing)

    Returns: list of {"role": ..., "content": ...} messages for
             tokenizer.apply_chat_template()
    """
    convs = conversation.get("conversations", conversation.get("turns", []))
    messages = [{"role": "system", "content": get_system_prompt(num_classes)}]

    # Build conversation history as a single user message
    history_lines = ["Conversation history:"]

    for i, turn in enumerate(convs):
        if i >= target_turn_idx:
            break
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", "")

        if role == "human":
            history_lines.append(f"[User]: {value}")
        elif role == "observation" and include_observations:
            history_lines.append(f"[Retrieved Context]: {value}") #{value[:500]}
        elif role == "gpt":
            history_lines.append(f"[System]: {value}")
        elif role == "function_call":
            continue  # skip

    # Add the current user query (the turn just before the target gpt turn)
    # Find the human turn immediately preceding target_turn_idx
    for i in range(target_turn_idx - 1, -1, -1):
        turn = convs[i]
        role = turn.get("from", turn.get("role", ""))
        if role == "human":
            # Already included in history
            break
        elif role == "observation" and include_observations:
            # Already included
            break

    history_text = "\n".join(history_lines)

    classification_query = (
        f"{history_text}\n\n"
        f"Based on the conversation above, predict the ambiguity level "
        f"of the NEXT system response.\n"
        f"Respond with ONLY the number "
        f"({'0, 1, 2, or 3' if num_classes == 4 else '0 or 1'})."
    )

    if adversarial_suffix:
        classification_query += f" {adversarial_suffix}"

    messages.append({"role": "user", "content": classification_query})
    return messages


def build_per_turn_prompts(
    conversation: Dict,
    num_classes: int = 2,
    per_turn: bool = True,
    include_observations: bool = True,
    adversarial_suffix: str = "",
) -> List[Tuple[List[Dict], int, int]]:
    """
    Build prompts for all target system turns in a conversation.

    Returns: list of (messages, gpt_turn_index, gold_label) tuples.
`
    per_turn=True:  one prompt per gpt turn
    per_turn=False: one prompt for the last gpt turn only
    """
    from convqa_eval.data.loader import remap_label

    convs = conversation.get("conversations", conversation.get("turns", []))

    # Find all gpt turns with labels
    gpt_turns = []
    for i, turn in enumerate(convs):
        role = turn.get("from", turn.get("role", ""))
        if role == "gpt":
            raw_label = int(turn.get("ambiguous_type", 0))
            label = remap_label(raw_label, num_classes)
            gpt_turns.append((i, label, raw_label))

    if not gpt_turns:
        return []

    if not per_turn:
        gpt_turns = [gpt_turns[-1]]

    results = []
    for gpt_idx, label, raw_label in gpt_turns:
        messages = build_classification_prompt(
            conversation, gpt_idx, num_classes,
            include_observations, adversarial_suffix)
        results.append((messages, gpt_idx, label))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Label parsing
# ══════════════════════════════════════════════════════════════════════════════
def parse_llm_label(text: str, num_classes: int = 2) -> int:
    """
    Parse LLM output text into an integer label.

    Handles various output formats:
      "0", "1", "2", "3"
      "clear", "ambiguous", "needs_clarification", etc.
      "The answer is 2", "I predict: 3", etc.

    Returns: integer label (0-based), or -1 if unparseable.
    """
    text = text.strip().lower()
    parse_map = PARSE_MAP_4 if num_classes == 4 else PARSE_MAP_2

    # Direct match
    if text in parse_map:
        return parse_map[text]

    # First number in the text
    numbers = re.findall(r"\b([0-3])\b", text)
    if numbers:
        n = int(numbers[0])
        if num_classes == 2:
            return min(n, 1)
        return min(n, 3)

    # Keyword search
    for keyword, label in sorted(parse_map.items(), key=lambda x: -len(x[0])):
        if keyword in text:
            return label

    return -1  # unparseable
