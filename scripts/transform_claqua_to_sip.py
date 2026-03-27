#!/usr/bin/env python3
"""
transform_claqua_to_sip.py — ShareGPT Format (Corrected)

Transform CLAQUA conversational QA dataset into ShareGPT-style SIP format.

Correct ShareGPT position rules:
  Position 0: human (always first)
  Position 1: function_call (tool invocation)
  Position 2: observation (tool result)
  Position 3: gpt (system response)
  Position 4: human (next turn)
  ...

Key corrections:
1. Conversation ALWAYS starts with human (no observation at position 0)
2. Pattern: [human, function_call, observation, gpt, human, function_call, observation, gpt, ...]
3. Every conversation ends with gpt turn (contains ambiguous_type)
4. Topic type truncated to first 5 space-separated words
5. Topic category inferred from entity types (music, film, tv, etc.)

Author: Claude (Anthropic)
Date: 2026-03-08
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random
import re


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC CATEGORIZATION
# ══════════════════════════════════════════════════════════════════════════════

TOPIC_KEYWORDS = {
    "music": ["music", "song", "album", "artist", "musician", "composer", "lyricist", "songwriter"],
    "tv": ["tv", "television", "program", "series", "episode"],
    "books": ["book", "author", "writer", "publication", "novel"],
    "film": ["film", "movie", "actor", "director", "cinema"],
    "traffic": ["traffic", "transport", "vehicle", "road"],
    "people": ["people", "person", "human", "individual"],
    "sports": ["sport", "athlete", "game", "team", "player"],
    "geography": ["geography", "geographic", "terrain", "climate"],
    "location": ["location", "place", "city", "country", "region"],
    "organization": ["organization", "company", "institution", "corporation", "business"]
}


def infer_topic_category(entity1_type: str, entity2_type: str) -> str:
    """
    Infer topic category from entity types.
    
    Categories: Music, TV, Books, Film, Traffic, People, Sports, Geography,
                Location, Organization
    
    Strategy: Count keyword occurrences in concatenated entity types,
              return category with most matches.
    """
    combined = (entity1_type + " " + entity2_type).lower()
    
    scores = Counter()
    for category, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            scores[category] += combined.count(keyword)
    
    if not scores:
        return "general"
    
    top_category = scores.most_common(1)[0][0]
    return top_category.capitalize()


def truncate_entity_type(entity_type: str, max_words: int = 5) -> str:
    """
    Truncate entity type to first N space-separated words.
    
    Example:
      Input: "business.board_member book.author biology.organism award.winner ..."
      Output: "business.board_member book.author biology.organism award.winner award.ranked_item"
    """
    words = entity_type.split()
    return " ".join(words[:max_words])


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_entity(entity_str: str) -> Tuple[str, str, str]:
    """Parse entity: 'name <S> type <S> description' → (name, type, desc)"""
    parts = entity_str.split("<S>", 2)
    return (parts[0].strip(),
            parts[1].strip() if len(parts) > 1 else "",
            parts[2].strip() if len(parts) > 2 else "")


def build_entity_observation(entity1: str, entity2: str) -> str:
    """
    Build observation text for entity disambiguation.
    
    Topic types are truncated to first 5 space-separated words.
    """
    e1_name, e1_type, e1_desc = parse_entity(entity1)
    e2_name, e2_type, e2_desc = parse_entity(entity2)
    
    # Truncate types
    e1_type = truncate_entity_type(e1_type, 5)
    e2_type = truncate_entity_type(e2_type, 5)
    
    lines = [
        f"Topic: {e1_name} (Type: {e1_type})",
        f"Description: {e1_desc}" if e1_desc else "",
        "",
        f"Topic: {e2_name} (Type: {e2_type})",
        f"Description: {e2_desc}" if e2_desc else ""
    ]
    return "\n".join(l for l in lines if l)


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def parse_context_turns(context: str) -> List[str]:
    """Split 'Q1 <EOS> A1 <EOS> Q2' → ['Q1', 'A1', 'Q2']"""
    return [t.strip() for t in context.split("<EOS>") if t.strip()]


def build_sharegpt_conversation(
    context: str, entity1: str, entity2: str, label: str,
    conv_index: int, split: str
) -> Dict:
    """
    Convert CLAQUA item to ShareGPT SIP format.
    
    Correct position rules:
      [0] human
      [1] function_call
      [2] observation
      [3] gpt
      [4] human (if multi-turn)
      [5] function_call
      [6] observation
      [7] gpt
      ...
    
    Pattern: [human, function_call, observation, gpt] repeats per turn.
    Always ends with gpt turn containing ambiguous_type.
    """
    turns = parse_context_turns(context)
    if not turns:
        raise ValueError(f"Empty context at index {conv_index}")
    
    entity_obs = build_entity_observation(entity1, entity2)
    
    # Get full entity types for topic categorization
    e1_name, e1_type_full, e1_desc = parse_entity(entity1)
    e2_name, e2_type_full, e2_desc = parse_entity(entity2)
    topic_category = infer_topic_category(e1_type_full, e2_type_full)
    
    # Map CLAQUA binary label to 4-level ambiguous_type
    gold_label = int(label)
    final_ambiguous_type = 0 if gold_label == 0 else 3
    
    conversations = []
    
    # Process turns: turns[0,2,4,...] = human, turns[1,3,5,...] = gpt
    # function_call and observation ONLY in first turn
    turn_id = 1
    for i, turn_text in enumerate(turns):
        if i % 2 == 0:
            # User turn: [human]
            conversations.append({
                "from": "human",
                "value": turn_text,
                "turn_id": turn_id
            })
            
            # Add function_call and observation ONLY for first turn (i==0)
            if i == 0:
                conversations.append({
                    "from": "function_call",
                    "value": "retrieve_entities()",
                    "turn_id": turn_id
                })
                conversations.append({
                    "from": "observation",
                    "value": entity_obs,
                    "turn_id": turn_id,
                    "observation_type": "entity_disambiguation"
                })
        else:
            # System turn: [gpt]
            # All prior system turns get ambiguous_type=0 (clear)
            # Final system turn gets the gold label
            is_final = (i == len(turns) - 1)
            ambiguous_type = final_ambiguous_type if is_final else 0
            
            conversations.append({
                "from": "gpt",
                "value": turn_text,
                "turn_id": turn_id,
                "ambiguous_type": ambiguous_type
            })
            turn_id += 1
    
    # CRITICAL: If conversation ends with human turn, add empty gpt with gold label
    # This ensures every conversation ends with a gpt turn carrying ambiguous_type
    if len(turns) % 2 == 1:
        # Odd number of turns → last turn is human (at even index)
        # Need to add [gpt] with the gold label
        conversations.append({
            "from": "gpt",
            "value": "",
            "turn_id": turn_id,
            "ambiguous_type": final_ambiguous_type
        })
    
    # Calculate num_turns and turn_labels
    # num_turns = number of user utterances
    num_user_utterances = len([t for i, t in enumerate(turns) if i % 2 == 0])
    num_turns = num_user_utterances
    
    # Build turn_labels array (one label per turn)
    turn_labels = []
    for i in range(num_turns):
        # Check if this turn has a gpt response in the original context
        gpt_index = 2 * i + 1  # turns[1], turns[3], turns[5], ...
        if gpt_index < len(turns):
            # Turn i has a gpt response in context
            is_final_gpt = (gpt_index == len(turns) - 1)
            turn_labels.append(final_ambiguous_type if is_final_gpt else 0)
        else:
            # Turn i has NO gpt response in context (incomplete turn)
            # This is the final turn where system needs to respond
            turn_labels.append(final_ambiguous_type)
    
    label_dist = {
        "clear": turn_labels.count(0),
        "slightly_ambiguous": turn_labels.count(1),
        "needs_clarification": turn_labels.count(2),
        "highly_ambiguous": turn_labels.count(3)
    }
    
    metadata = {
        "num_turns": num_turns,
        "turn_labels": turn_labels,
        "label_distribution": label_dist,
        "topic_category": topic_category,
        "topic_types": {
            "entity1": truncate_entity_type(e1_type_full, 5),
            "entity2": truncate_entity_type(e2_type_full, 5)
        },
        "source": {
            "format": "CLAQUA",
            "split": split,
            "index": conv_index,
            "original_label": label
        }
    }
    
    return {
        "conversations": conversations,
        "metadata": metadata,
        "prompt": entity_obs
    }


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_claqua_item(item: Dict, line_no: int) -> List[str]:
    """Validate CLAQUA input item."""
    errors = []
    for field in ["context", "entity1", "entity2", "label"]:
        if field not in item:
            errors.append(f"Line {line_no}: missing '{field}'")
    if not errors:
        if item["label"] not in ("0", "1"):
            errors.append(f"Line {line_no}: label must be '0' or '1'")
        if not parse_context_turns(item["context"]):
            errors.append(f"Line {line_no}: empty context")
        for efield in ["entity1", "entity2"]:
            if "<S>" not in item[efield]:
                errors.append(f"Line {line_no}: {efield} missing <S>")
    return errors


def validate_sharegpt_item(item: Dict, idx: int) -> List[str]:
    """Validate ShareGPT output item."""
    errors = []
    for field in ["conversations", "metadata", "prompt"]:
        if field not in item:
            errors.append(f"Item {idx}: missing '{field}'")
    if errors:
        return errors
    
    convs = item["conversations"]
    
    # First message must be human
    if not convs or convs[0]["from"] != "human":
        errors.append(f"Item {idx}: first conv must be 'human'")
    
    # Last message must be gpt
    if convs and convs[-1]["from"] != "gpt":
        errors.append(f"Item {idx}: last conv must be 'gpt' (with ambiguous_type)")
    
    # Check first turn pattern: [human, function_call, observation, gpt]
    if len(convs) >= 4:
        expected_first = ["human", "function_call", "observation", "gpt"]
        for i, expected_role in enumerate(expected_first):
            if i < len(convs) and convs[i]["from"] != expected_role:
                errors.append(
                    f"Item {idx}: conv[{i}] (first turn) should be '{expected_role}', got '{convs[i]['from']}'"
                )
    
    # Check subsequent turns: [human, gpt, human, gpt, ...]
    # Starting from position 4 (after first turn)
    if len(convs) > 4:
        for i in range(4, len(convs)):
            # Positions 4, 6, 8, ... should be human
            # Positions 5, 7, 9, ... should be gpt
            if (i - 4) % 2 == 0:
                expected = "human"
            else:
                expected = "gpt"
            
            if convs[i]["from"] != expected:
                errors.append(
                    f"Item {idx}: conv[{i}] should be '{expected}', got '{convs[i]['from']}'"
                )
    
    # Check turn_id presence
    for i, c in enumerate(convs):
        if "turn_id" not in c:
            errors.append(f"Item {idx}: conv[{i}] missing turn_id")
    
    # Check gpt turns have ambiguous_type
    for i, c in enumerate(convs):
        if c["from"] == "gpt" and "ambiguous_type" not in c:
            errors.append(f"Item {idx}: conv[{i}] (gpt) missing ambiguous_type")
    
    # Check observation has observation_type
    for i, c in enumerate(convs):
        if c["from"] == "observation" and "observation_type" not in c:
            errors.append(f"Item {idx}: conv[{i}] (observation) missing observation_type")
    
    # Check metadata
    meta = item["metadata"]
    for field in ["num_turns", "turn_labels", "label_distribution", "topic_category"]:
        if field not in meta:
            errors.append(f"Item {idx}: metadata missing '{field}'")
    
    if "turn_labels" in meta and meta["turn_labels"]:
        if not all(l in (0,1,2,3) for l in meta["turn_labels"]):
            errors.append(f"Item {idx}: invalid turn_labels")
        if len(meta["turn_labels"]) > 1:
            if any(l != 0 for l in meta["turn_labels"][:-1]):
                errors.append(f"Item {idx}: prior labels must be 0")
    
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistics(items: List[Dict]) -> Dict:
    """Compute dataset statistics including topic category distribution."""
    label_counts = Counter()
    turn_dist = Counter()
    topic_dist = Counter()
    total_user = total_sys = 0
    
    for item in items:
        convs = item["conversations"]
        meta = item["metadata"]
        
        label_counts[meta["turn_labels"][-1]] += 1
        topic_dist[meta["topic_category"]] += 1
        
        n_user = sum(1 for c in convs if c["from"] == "human")
        n_sys = sum(1 for c in convs if c["from"] == "gpt")
        total_user += n_user
        total_sys += n_sys
        
        turn_dist["single-turn" if n_user == 1 else "multi-turn"] += 1
    
    return {
        "total": len(items),
        "label_counts": dict(label_counts),
        "turn_dist": dict(turn_dist),
        "topic_dist": dict(topic_dist),
        "avg_user_turns": total_user / max(len(items), 1),
        "avg_system_turns": total_sys / max(len(items), 1),
    }


def print_statistics(stats: Dict, input_file: str, output_file: str, split: str):
    """Print formatted statistics."""
    print("\n" + "=" * 80)
    print("CLAQUA → ShareGPT SIP Transformation Complete")
    print("=" * 80)
    print(f"\nInput : {input_file}")
    print(f"Output: {output_file}")
    print(f"Split : {split}")
    print(f"\nTotal : {stats['total']:,} conversations")
    
    # Label distribution
    lc = stats['label_counts']
    total = stats['total']
    print(f"\nFinal turn label distribution:")
    label_names = {0: "clear", 1: "slightly_ambiguous",
                   2: "needs_clarification", 3: "highly_ambiguous"}
    for label in [0, 1, 2, 3]:
        count = lc.get(label, 0)
        if count > 0:
            print(f"  {label} ({label_names[label]:20s}): {count:6,} ({100*count/total:5.1f}%)")
    
    # Topic distribution
    topic_dist = stats['topic_dist']
    print(f"\nTopic category distribution:")
    for topic in sorted(topic_dist.keys(), key=lambda x: topic_dist[x], reverse=True):
        count = topic_dist[topic]
        print(f"  {topic:15s}: {count:6,} ({100*count/total:5.1f}%)")
    
    # Turn distribution
    td = stats['turn_dist']
    print(f"\nConversation structure:")
    print(f"  Single-turn: {td.get('single-turn', 0):6,} ({100*td.get('single-turn',0)/total:5.1f}%)")
    print(f"  Multi-turn : {td.get('multi-turn', 0):6,} ({100*td.get('multi-turn',0)/total:5.1f}%)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Transform CLAQUA to ShareGPT SIP format")
    parser.add_argument("--input", required=True, help="Input CLAQUA JSONL")
    parser.add_argument("--output", help="Output ShareGPT JSON (required unless --stats)")
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--stats", action="store_true", help="Stats only, no output")
    args = parser.parse_args()
    
    if not args.stats and not args.output:
        parser.error("--output required unless --stats")
    
    # Load
    print(f"\n[Load] {args.input}")
    claqua_items = []
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    claqua_items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  WARN line {line_no}: {e}", file=sys.stderr)
    print(f"  Loaded {len(claqua_items):,} items")
    
    # Validate input
    print(f"\n[Validate Input]")
    all_errors = []
    for i, item in enumerate(claqua_items, 1):
        all_errors.extend(validate_claqua_item(item, i))
    if all_errors:
        for err in all_errors[:10]:
            print(f"  ERROR: {err}")
        sys.exit(1)
    print(f"  ✓ Passed")
    
    # Transform
    print(f"\n[Transform]")
    sip_items = []
    for i, item in enumerate(claqua_items):
        try:
            sip_items.append(build_sharegpt_conversation(
                item["context"], item["entity1"], item["entity2"],
                item["label"], i, args.split))
        except Exception as e:
            print(f"  ERROR item {i}: {e}", file=sys.stderr)
            continue
    print(f"  ✓ Transformed {len(sip_items):,} conversations")
    
    # Validate output
    print(f"\n[Validate Output]")
    output_errors = []
    for i, item in enumerate(sip_items):
        output_errors.extend(validate_sharegpt_item(item, i))
    if output_errors:
        for err in output_errors[:10]:
            print(f"  ERROR: {err}")
        sys.exit(1)
    print(f"  ✓ Passed")
    
    # Spot check
    print(f"\n[Spot Check] 2 random examples:")
    for idx in random.sample(range(len(sip_items)), min(2, len(sip_items))):
        item = sip_items[idx]
        print(f"\n  Example {idx}: {item['metadata']['num_turns']} turn(s), "
              f"topic={item['metadata']['topic_category']}, labels={item['metadata']['turn_labels']}")
        for i, c in enumerate(item['conversations'][:6]):
            val = c['value'][:50] + "..." if len(c['value']) > 50 else c['value']
            extra = f", ambiguous_type={c['ambiguous_type']}" if 'ambiguous_type' in c else ""
            print(f"    [{i}] {c['from']:13s} (turn {c['turn_id']}{extra}): {val}")
        if len(item['conversations']) > 6:
            print(f"    ... and {len(item['conversations']) - 6} more")
    
    # Statistics
    stats = compute_statistics(sip_items)
    print_statistics(stats, args.input, args.output or "<none>", args.split)
    
    # Write
    if not args.stats:
        print(f"[Write] {args.output}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(sip_items, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Wrote {len(sip_items):,} conversations\n")


if __name__ == "__main__":
    main()
