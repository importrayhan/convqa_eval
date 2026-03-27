"""
SIP-format data loader with train-fraction sampling and non-overlapping
conversation-level validation splits.
"""

import json, random, logging, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Registry ─────────────────────────────────────────────────────────────────
BENCHMARK_REGISTRY: Dict[str, Dict] = {
    "pacific": {"description": "Finance proactive ConvQA", "per_turn_default": True},
    "simmic":  {"description": "SIMMC 2.1 shopping disambiguation", "per_turn_default": True},
    "claqua":  {"description": "CLaQuA 2-class ambiguous system initiative prediction",
                "per_turn_default": False},
}

LABEL_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3,
    "clear": 0, "slightly_ambiguous": 1,
    "needs_clarification": 2, "highly_ambiguous": 3,
    "ambiguous": 1, "not_ambiguous": 0,
}

BINARY_MAP = {0: 0, 1: 1, 2: 1, 3: 1}

def remap_label(raw: int, num_classes: int) -> int:
    raw = max(0, min(int(raw), 3))
    if num_classes == 4: return raw
    if num_classes == 2: return BINARY_MAP[raw]
    if num_classes == 3: return {0:0,1:1,2:1,3:2}[raw]
    raise ValueError(num_classes)

# ── SIP Parser ───────────────────────────────────────────────────────────────
def parse_sip_conversation(raw: Dict, num_classes: int = 2):
    """Return (user_texts, sys_texts, labels, metadata_list)."""
    convs = raw.get("conversations", raw.get("turns", []))
    user_utts, sys_utts, labels, metas = [], [], [], []
    i = 0
    while i < len(convs):
        c = convs[i]
        role = c.get("from", c.get("role", ""))
        if role == "function_call":
            i += 1; continue
        if role == "human":
            parts = [c.get("value", "")]
            i += 1
            while i < len(convs):
                cur = convs[i]
                r = cur.get("from", cur.get("role", ""))
                if r == "function_call":
                    i += 1
                elif r == "observation":
                    ot = cur.get("observation_type", "context").upper()
                    parts.append(f"<|context_start|> {cur.get('value','')} <|context_end|>")
                    i += 1
                elif r == "gpt":
                    break
                elif r == "human":
                    break
                else:
                    i += 1
            user_text = " ".join(parts)
            if i < len(convs) and convs[i].get("from", convs[i].get("role","")) == "gpt":
                gpt = convs[i]
                raw_label = int(gpt.get("ambiguous_type", 0))
                label = remap_label(raw_label, num_classes)
                user_utts.append(user_text)
                sys_utts.append(gpt.get("value", ""))
                labels.append(label)
                meta = dict(gpt.get("metadata", {}))
                meta["turn_id"] = gpt.get("turn_id", len(labels))
                meta["raw_label"] = raw_label
                metas.append(meta)
                i += 1
        else:
            i += 1
    return user_utts, sys_utts, labels, metas


# ── Dataset ──────────────────────────────────────────────────────────────────
class SIPDataset(Dataset):
    """Each item is one *conversation* (variable-length turns)."""
    def __init__(self, processed: List[Dict]):
        self.data = processed
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def sip_collate_single(batch):
    """batch_size=1 collator (preserves per-conv structure)."""
    assert len(batch) == 1
    return batch[0]


# ── Tokenise one conversation ────────────────────────────────────────────────
def tokenize_conversation(
    raw: Dict,
    tokenizer,
    max_length: int = 512,
    num_classes: int = 2,
    use_ground_truth: bool = True,
) -> Dict:
    """Tokenize a single SIP conversation into tensors."""
    user_utts, sys_utts, labels, metas = parse_sip_conversation(raw, num_classes)
    n = len(user_utts)
    if n == 0:
        return {
            "user_utterance": torch.empty(0, max_length, dtype=torch.long),
            "system_utterance": torch.empty(0, max_length, dtype=torch.long),
            "user_I_label": torch.zeros(0, dtype=torch.long),
            "system_I_label": torch.zeros(0, dtype=torch.long),
            "metadata": {"num_pairs": 0},
        }

    def tok(texts):
        enc = tokenizer(texts, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        return enc["input_ids"]

    return {
        "user_utterance": tok(user_utts),
        "system_utterance": tok(sys_utts),
        "user_I_label": torch.zeros(n, dtype=torch.long),
        "system_I_label": torch.tensor(labels, dtype=torch.long) if use_ground_truth
                          else torch.zeros(n, dtype=torch.long),
        "metadata": {
            "num_pairs": n,
            "num_classes": num_classes,
            "turn_metadata": metas,
            "user_texts": user_utts,
            "system_texts": sys_utts,
        },
    }


# ── Splitting utilities ─────────────────────────────────────────────────────
def train_val_split(
    data: List[Dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Non-overlapping *conversation-level* split (no data leakage)."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = max(1, int(len(data) * val_fraction))
    val_idx, train_idx = set(indices[:n_val]), set(indices[n_val:])
    return [data[i] for i in sorted(train_idx)], [data[i] for i in sorted(val_idx)]


def sample_train_fraction(
    data: List[Dict],
    fraction: float = 1.0,
    seed: int = 42,
) -> List[Dict]:
    """Sample a fraction of conversations (stratified when possible)."""
    if fraction >= 1.0:
        return data
    rng = random.Random(seed)
    n = max(1, int(len(data) * fraction))
    return rng.sample(data, n)


# ── High-level loaders (CoIR-style) ─────────────────────────────────────────
def load_benchmark(
    name: str,
    split: str = "train",
    data_dir: str = "benchmarks",
) -> List[Dict]:
    """Load raw SIP JSON for a registered benchmark."""
    path = Path(data_dir) / name / "data" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else [raw]


def get_tasks(tasks: List[str] = None):
    """Return task descriptors (mirrors CoIR `get_tasks`)."""
    if tasks is None:
        tasks = list(BENCHMARK_REGISTRY.keys())
    return {t: BENCHMARK_REGISTRY[t] for t in tasks if t in BENCHMARK_REGISTRY}
