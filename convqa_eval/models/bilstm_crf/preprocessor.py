"""
SIP Preprocessor 
1. num_classes=2: groups 'clear'+'slightly_ambiguous' → 0,
   'needs_clarification'+'highly_ambiguous' → 1
2. Prints per-class statistics vs total turns for any num_classes
"""

import json
import torch
from collections import Counter
from typing import List, Dict, Tuple
from transformers import BertTokenizer


# Raw 4-class names as stored in SIP data
_CLASS4_NAMES = ['clear', 'slightly_ambiguous', 'needs_clarification', 'highly_ambiguous']
_CLASS3_NAMES = ['clear', 'needs_clarification', 'highly_ambiguous']
_CLASS2_NAMES = ['clear', 'ambiguous']

# For 2-class grouping: original labels 0,1 → 0 ; 2,3 → 1
_BINARY_MAP = {0: 0, 1: 0, 2: 1, 3: 1}


def remap_label(raw_label: int, num_classes: int) -> int:
    """
    Remap a raw 4-level label to the target number of classes.

    raw_label ∈ {0,1,2,3}  (clear / slightly_ambiguous /
                             needs_clarification / highly_ambiguous)
    num_classes == 2 → {0,1}   groups: (0,1)→0,  (2,3)→1
    num_classes == 3 → {0,1,2} groups: 0→0, (1,2)→1, 3→2
    num_classes == 4 → identity
    """
    raw_label = max(0, min(raw_label, 3))
    if num_classes == 4:
        return raw_label
    if num_classes == 2:
        return _BINARY_MAP[raw_label]
    if num_classes == 3:
        # 0→0, 1→1, 2→1, 3→2
        return {0: 0, 1: 1, 2: 1, 3: 2}[raw_label]
    raise ValueError(f"num_classes must be 2, 3 or 4, got {num_classes}")


def print_label_statistics(
    labels: List[int],
    num_classes: int,
    title: str = "Label Statistics"
) -> None:
    """
    Print per-class count and percentage against total turns.

    Example output (2-class, 100 turns):
    ┌─────────────────────────────────────────────┐
    │ Label Statistics (2 classes, 100 turns)     │
    ├──────────────────────┬────────┬─────────────┤
    │ Class                │  Count │     %       │
    ├──────────────────────┼────────┼─────────────┤
    │ 0  clear/slight      │     72 │   72.0 %    │
    │ 1  ambiguous         │     28 │   28.0 %    │
    ├──────────────────────┼────────┼─────────────┤
    │ Total turns          │    100 │  100.0 %    │
    └──────────────────────┴────────┴─────────────┘
    """
    total = len(labels)
    counts = Counter(labels)

    class_labels = _get_display_names(num_classes)

    bar   = "─" * 48
    hline = "├" + bar[:24] + "┬" + bar[:8] + "┬" + bar[:13] + "┤"
    top   = "┌" + bar[:24] + "┬" + bar[:8] + "┬" + bar[:13] + "┐"
    bot   = "└" + bar[:24] + "┴" + bar[:8] + "┴" + bar[:13] + "┘"

    print(f"\n┌{'─'*46}┐")
    print(f"│  {title} ({num_classes} classes, {total} turns){' '*(max(0,46-len(title)-len(str(num_classes))-len(str(total))-15))}│")
    print(f"├{'─'*24}┬{'─'*8}┬{'─'*13}┤")
    print(f"│ {'Class':<22} │{'Count':>7} │{'%':>12} │")
    print(f"├{'─'*24}┼{'─'*8}┼{'─'*13}┤")

    for idx, display in class_labels.items():
        cnt = counts.get(idx, 0)
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"│ {display:<22} │{cnt:>7} │{pct:>11.1f}% │")

    print(f"├{'─'*24}┼{'─'*8}┼{'─'*13}┤")
    print(f"│ {'Total turns':<22} │{total:>7} │{'100.0':>11}% │")
    print(f"└{'─'*24}┴{'─'*8}┴{'─'*13}┘")


def _get_display_names(num_classes: int) -> Dict[int, str]:
    if num_classes == 2:
        return {
            0: "0  clear/slight_ambig",
            1: "1  needs_clari/highly",
        }
    if num_classes == 3:
        return {
            0: "0  clear",
            1: "1  needs_clarification",
            2: "2  highly_ambiguous",
        }
    # 4-class
    return {
        0: "0  clear",
        1: "1  slightly_ambiguous",
        2: "2  needs_clarification",
        3: "3  highly_ambiguous",
    }


class SIPPreprocessor:
    """
    SIP Preprocessor with:
    - 2-class grouping: (clear, slightly_ambiguous) → 0,
                        (needs_clarification, highly_ambiguous) → 1
    - Class statistics printed for any num_classes
    - Observations merged into user utterances
    - Per-GPT-utterance labels
    """

    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        max_len: int = 128,
        num_classes: int = 2,
    ):
        assert num_classes in [2, 3, 4], "num_classes must be 2, 3, or 4"
        self.max_len = max_len
        self.num_classes = num_classes

        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        self.class_names = {
            2: ['clear/slightly_ambiguous', 'needs_clarification/highly_ambiguous'],
            3: ['clear', 'needs_clarification', 'highly_ambiguous'],
            4: ['clear', 'slightly_ambiguous', 'needs_clarification', 'highly_ambiguous'],
        }

        print(f"[SIPPreprocessor] bert={bert_model}  max_len={max_len}  "
              f"num_classes={num_classes}")
        print(f"  Classes: {self.class_names[num_classes]}")

    # ------------------------------------------------------------------
    def parse_conversations(
        self,
        data: Dict
    ) -> Tuple[List[str], List[str], List[int], List[Dict]]:
        """
        Walk the conversation list and build aligned (user, system, label) triples.

        - function_call turns are skipped
        - observation turns are appended to the preceding human utterance
        - label is read from gpt['ambiguous_type'] then remapped to num_classes
        """
        conversations = data.get('conversations', [])

        user_utterances: List[str] = []
        system_utterances: List[str] = []
        system_labels: List[int] = []
        turn_metadata: List[Dict] = []

        i = 0
        while i < len(conversations):
            conv = conversations[i]

            if conv['from'] in ('function_call',):
                i += 1
                continue

            if conv['from'] == 'human':
                user_parts = [conv['value']]
                i += 1

                # Collect observations + find matching gpt
                while i < len(conversations):
                    cur = conversations[i]
                    if cur['from'] == 'function_call':
                        i += 1
                    elif cur['from'] == 'observation':
                        obs_type = cur.get('observation_type', 'context').upper()
                        user_parts.append(f"[{obs_type}] {cur['value']}")
                        i += 1
                    elif cur['from'] == 'gpt':
                        break
                    elif cur['from'] == 'human':
                        break  # missing gpt – skip
                    else:
                        i += 1

                user_text = " ".join(user_parts)

                if i < len(conversations) and conversations[i]['from'] == 'gpt':
                    gpt = conversations[i]
                    raw_label = int(gpt.get('ambiguous_type', 0))
                    label = remap_label(raw_label, self.num_classes)

                    user_utterances.append(user_text)
                    system_utterances.append(gpt.get('value', ''))
                    system_labels.append(label)

                    meta = dict(gpt.get('metadata', {}))
                    meta['turn_id'] = gpt.get('turn_id', len(system_labels))
                    meta['raw_label'] = raw_label
                    turn_metadata.append(meta)
                    i += 1
            else:
                i += 1

        return user_utterances, system_utterances, system_labels, turn_metadata

    # ------------------------------------------------------------------
    def process_conversation(
        self,
        data: Dict,
        use_ground_truth: bool = True,
        print_stats: bool = False,
    ) -> Dict:
        """Process a single conversation dict into tensors."""
        user_utts, sys_utts, sys_labels, meta_list = \
            self.parse_conversations(data)

        if print_stats and use_ground_truth:
            print_label_statistics(
                sys_labels,
                self.num_classes,
                title="Conversation Label Distribution"
            )

        # Tokenise
        def tok(texts: List[str]) -> torch.Tensor:
            enc = self.tokenizer(
                texts,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            return enc['input_ids']

        n = len(user_utts)
        user_ids  = tok(user_utts)  if n else torch.empty(0, self.max_len, dtype=torch.long)
        sys_ids   = tok(sys_utts)   if n else torch.empty(0, self.max_len, dtype=torch.long)
        labels_t  = torch.tensor(sys_labels, dtype=torch.long) if use_ground_truth \
                    else torch.zeros(n, dtype=torch.long)

        return {
            'user_utterance':  user_ids,
            'system_utterance': sys_ids,
            'user_I_label':    torch.zeros(n, dtype=torch.long),
            'system_I_label':  labels_t,
            'metadata': {
                'num_pairs':    n,
                'num_classes':  self.num_classes,
                'class_names':  self.class_names[self.num_classes],
                'turn_metadata': meta_list,
                'label_distribution': {
                    self.class_names[self.num_classes][i]: sys_labels.count(i)
                    for i in range(self.num_classes)
                } if use_ground_truth else {},
                'has_observations': any(
                    '[TABLE]' in u or '[CONTEXT]' in u or '[SCENE]' in u
                    for u in user_utts
                ),
            },
        }

    # ------------------------------------------------------------------
    def process_dataset(
        self,
        dataset: List[Dict],
        use_ground_truth: bool = True,
        print_stats: bool = True,
    ) -> Tuple[List[Dict], List[int]]:
        """
        Process a full list of SIP conversations.

        Returns (processed_list, all_labels).
        Prints aggregate class statistics at the end.
        """
        processed = []
        all_labels: List[int] = []

        for item in dataset:
            p = self.process_conversation(item, use_ground_truth=use_ground_truth)
            processed.append(p)
            all_labels.extend(p['metadata']['label_distribution'].get(
                name, 0
            ) * [idx]
            for idx, name in enumerate(self.class_names[self.num_classes]))

        # Flatten — easier to collect directly
        all_labels = []
        for item in dataset:
            _, _, labels, _ = self.parse_conversations(item)
            all_labels.extend(labels)

        if print_stats:
            print_label_statistics(
                all_labels,
                self.num_classes,
                title="Dataset Label Distribution"
            )

        return processed, all_labels

    def get_class_name(self, idx: int) -> str:
        return self.class_names[self.num_classes][idx]


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "conversations": [
            {"from": "human",         "value": "What is X?",          "turn_id": 1},
            {"from": "function_call", "value": "retrieve_table()",     "turn_id": 1},
            {"from": "observation",   "value": "Col1|Col2\n100|200",   "turn_id": 1, "observation_type": "table"},
            {"from": "gpt",           "value": "Which year?",          "turn_id": 1, "ambiguous_type": 2},
            {"from": "human",         "value": "2019",                 "turn_id": 2},
            {"from": "gpt",           "value": "The value is 100",     "turn_id": 2, "ambiguous_type": 1},
            {"from": "human",         "value": "Thanks",               "turn_id": 3},
            {"from": "gpt",           "value": "You're welcome",       "turn_id": 3, "ambiguous_type": 0},
        ]
    }

    for nc in [2, 3, 4]:
        print(f"\n{'='*55}")
        print(f"  num_classes = {nc}")
        print('='*55)
        pre = SIPPreprocessor(
            bert_model='bert-base-uncased',
            max_len=64,
            num_classes=nc,
        )
        result = pre.process_conversation(sample, print_stats=True)
        print(f"  labels tensor: {result['system_I_label'].tolist()}")
