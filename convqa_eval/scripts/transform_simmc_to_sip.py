"""
Works with python 3.10
---------------------------
Change this snippet according to your need:
                    # Add observation
                    conversations.append({
                        "from": "observation",
                        "value": ----> observation_text,
                        "turn_id": turn_idx + 1,
                        "observation_type": "scene",
                        "scene_id": scene_id,
                        "scene_data": -----> obs_data
---------------------------
Converts SIMMC 2.1 dialogue data to SIP format with:
- Scene observations with positional grouping (up, down, left, right, center)
- Disambiguation labels from REQUEST:DISAMBIGUATE or disambiguation_label=1
- Function call + observation pattern for scenes
- First turn scene observation

Input Format:
{
    "dialogue_data": [
        {
            "dialogue": [{"turn_idx": 0, "transcript": "...", "system_transcript": "..."}],
            "dialogue_idx": 10507,
            "domain": "fashion",
            "scene_ids": {"0": "scene_name", "5": "scene_name2"}
        }
    ],
    "split": "dev",
    "domain": "fashion/furniture"
}

Output Format:
{
    "conversations": [
        {"from": "human", "value": "...", "turn_id": 1},
        {"from": "function_call", "value": "observe_scene()", "turn_id": 1},
        {"from": "observation", "value": "...", "turn_id": 1, "observation_type": "scene"},
        {"from": "gpt", "value": "...", "turn_id": 1, "ambiguous_type": 0}
    ],
    "metadata": {...}
}

Usage:
python scripts/transform_simmc_to_sip.py \
    --input data/simmc/dev.json \
    --scene_dir data/simmc/scene_jsons \
    --fashion_metadata data/simmc/fashion_prefab_metadata_all.json \
    --furniture_metadata data/simmc/furniture_prefab_metadata_all.json \
    --output data/simmc/simmc_dev_sip.json
"""

import json
import os
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


class SIMMCToSIPTransformer:
    """Transform SIMMC 2.1 dialogues to SIP format with scene observations"""
    
    def __init__(
        self,
        scene_dir: str,
        fashion_metadata_path: str,
        furniture_metadata_path: str
    ):
        self.scene_dir = Path(scene_dir)
        
        # Load metadata
        print("[Loading] Metadata files...")
        with open(fashion_metadata_path, 'r') as f:
            self.fashion_metadata = json.load(f)
        
        with open(furniture_metadata_path, 'r') as f:
            self.furniture_metadata = json.load(f)
        
        print(f"  ✓ Loaded {len(self.fashion_metadata)} fashion items")
        print(f"  ✓ Loaded {len(self.furniture_metadata)} furniture items")
    
    def load_scene(self, scene_id: str) -> Optional[Dict]:
        """Load scene JSON file"""
        scene_path = self.scene_dir / f"{scene_id}_scene.json"
        
        if not scene_path.exists():
            print(f"  [Warning] Scene file not found: {scene_id}")
            return None
        
        with open(scene_path, 'r') as f:
            return json.load(f)
    
    def get_object_details(self, prefab_path: str, domain: str) -> Dict:
        """Get object details from metadata"""
        metadata = self.fashion_metadata if domain == 'fashion' else self.furniture_metadata
        
        # Try exact match
        if prefab_path in metadata:
            return metadata[prefab_path]
        
        # Try partial match
        for key in metadata:
            if prefab_path in key or key in prefab_path:
                return metadata[key]
        
        # Return minimal info
        return {
            "type": "unknown",
            "prefab_path": prefab_path
        }
    
    def process_scene_with_relationships(
        self,
        scene_data: Dict,
        domain: str
    ) -> Dict[str, List[Dict]]:
        """
        Process scene with relationships into positional groups.
        
        Returns:
            Dict with keys: up, down, left, right, center
        """
        scene = scene_data['scenes'][0]
        objects = scene.get('objects', [])
        relationships = scene.get('relationships', {})
        
        # Initialize groups
        groups = {
            'up': [],
            'down': [],
            'left': [],
            'right': [],
            'center': []
        }
        
        # Build object lookup
        obj_by_index = {}
        for obj in objects:
            idx = obj.get('index', obj.get('unique_id'))
            obj_by_index[idx] = obj
        
        # Track positioned objects
        positioned = set()
        
        # Process each direction
        for direction in ['up', 'down', 'left', 'right']:
            if direction not in relationships:
                continue
            
            for obj_idx_str, related_indices in relationships[direction].items():
                obj_idx = int(obj_idx_str)
                
                if obj_idx not in obj_by_index:
                    continue
                
                obj = obj_by_index[obj_idx]
                prefab_path = obj.get('prefab_path', '')
                
                if not prefab_path:
                    continue
                
                # Get object details
                details = self.get_object_details(prefab_path, domain)
                
                # Add to group
                groups[direction].append({
                    **details,
                    'unique_id': obj.get('unique_id', obj_idx),
                    'index': obj_idx,
                    'bbox': obj.get('bbox', [])
                })
                
                positioned.add(obj_idx)
        
        # Add unpositione objects to center
        for idx, obj in obj_by_index.items():
            if idx not in positioned:
                prefab_path = obj.get('prefab_path', '')
                if prefab_path:
                    details = self.get_object_details(prefab_path, domain)
                    groups['center'].append({
                        **details,
                        'unique_id': obj.get('unique_id', idx),
                        'index': idx,
                        'bbox': obj.get('bbox', [])
                    })
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def process_scene_with_items(
        self,
        scene_data: Dict,
        domain: str
    ) -> List[Dict]:
        """
        Process scene with simple Items list.
        
        Returns:
            List of item details
        """
        items = scene_data.get('Items', [])
        processed_items = []
        
        for item in items:
            prefab_path = item.get('prefabPath', '')
            
            if not prefab_path:
                continue
            
            details = self.get_object_details(prefab_path, domain)
            
            processed_items.append({
                **details,
                'name': item.get('name', ''),
                'bbox': item.get('bbox', []),
                'position': item.get('position', [])
            })
        
        return processed_items
    
    def format_observation(self, processed_scene: Dict | List, has_relationships: bool) -> str:
        """
        Format processed scene as observation text.
        
        Args:
            processed_scene: Either dict (with relationships) or list (items)
            has_relationships: True if positional groups, False if items list
        """
        if has_relationships:
            # Format with positional groups
            parts = []
            
            for direction in ['up', 'down', 'left', 'right', 'center']:
                if direction not in processed_scene or not processed_scene[direction]:
                    continue
                
                items = processed_scene[direction]
                item_descriptions = []
                
                for item in items:
                    desc_parts = []
                    
                    # Color
                    if 'color' in item and item['color']:
                        desc_parts.append(item['color'])
                    
                    # Pattern
                    if 'pattern' in item and item['pattern'] and item['pattern'] != 'plain':
                        desc_parts.append(item['pattern'])
                    
                    # Type
                    item_type = item.get('type', 'item')
                    desc_parts.append(item_type)
                    
                    # Brand (optional)
                    if 'brand' in item and item['brand']:
                        desc_parts.append(f"by {item['brand']}")
                    
                    item_descriptions.append(' '.join(desc_parts))
                
                parts.append(f"{direction.upper()}: {', '.join(item_descriptions)}")
            
            return "Scene - " + "; ".join(parts)
        
        else:
            # Format items list
            item_descriptions = []
            
            for item in processed_scene:
                desc_parts = []
                
                # Color
                if 'color' in item and item['color']:
                    desc_parts.append(item['color'])
                
                # Type
                item_type = item.get('type', 'item')
                desc_parts.append(item_type)
                
                # Brand
                if 'brand' in item and item['brand']:
                    desc_parts.append(f"by {item['brand']}")
                
                # Price
                if 'price' in item and item['price']:
                    price = item['price']
                    if isinstance(price, str):
                        desc_parts.append(f"({price})")
                    else:
                        desc_parts.append(f"(${price})")
                
                item_descriptions.append(' '.join(desc_parts))
            
            return "Available: " + ", ".join(item_descriptions)
    
    def determine_ambiguity_label(self, turn: Dict) -> int:
        """
        Determine ambiguity label from turn.
        
        Returns:
            0: clear
            1: disambiguation needed
        """
        # Check system annotation for REQUEST:DISAMBIGUATE
        system_ann = turn.get('system_transcript_annotated', {})
        act = system_ann.get('act', '')
        
        if act == 'REQUEST:DISAMBIGUATE':
            return 1
        
        # Check transcript annotation for disambiguation_label
        trans_ann = turn.get('transcript_annotated', {})
        disambig_label = trans_ann.get('disambiguation_label', 0)
        
        if disambig_label == 1:
            return 1
        
        return 0
    
    def transform_dialogue(self, dialogue_data: Dict) -> Dict:
        """Transform single dialogue to SIP format"""
        dialogue = dialogue_data.get('dialogue', [])
        scene_ids = dialogue_data.get('scene_ids', {})
        domain = dialogue_data.get('domain', 'fashion')
        dialogue_idx = dialogue_data.get('dialogue_idx', -1)
        
        conversations = []
        turn_labels = []
        
        for turn_idx, turn in enumerate(dialogue):
            # Human utterance
            conversations.append({
                "from": "human",
                "value": turn.get('transcript', ''),
                "turn_id": turn_idx + 1
            })
            
            # Check if scene changes at this turn
            turn_idx_str = str(turn_idx)
            if turn_idx_str in scene_ids:
                scene_id = scene_ids[turn_idx_str]
                
                # Load scene
                scene_data = self.load_scene(scene_id)
                
                if scene_data:
                    # Add function call
                    conversations.append({
                        "from": "function_call",
                        "value": "observe_scene()",
                        "turn_id": turn_idx + 1
                    })
                    
                    # Process scene
                    if 'scenes' in scene_data:
                        # Has relationships
                        processed = self.process_scene_with_relationships(scene_data, domain)
                        observation_text = self.format_observation(processed, has_relationships=True)
                        obs_data = processed
                    elif 'Items' in scene_data:
                        # Simple items
                        processed = self.process_scene_with_items(scene_data, domain)
                        observation_text = self.format_observation(processed, has_relationships=False)
                        obs_data = processed
                    else:
                        observation_text = "Scene data unavailable"
                        obs_data = {}
                    
                    # Add observation
                    conversations.append({
                        "from": "observation",
                        "value": observation_text,
                        "turn_id": turn_idx + 1,
                        "observation_type": "scene",
                        "scene_id": scene_id,
                        "scene_data": obs_data
                    })
            
            # System response (GPT)
            system_text = turn.get('system_transcript', '')
            ambiguity_label = self.determine_ambiguity_label(turn)
            
            system_ann = turn.get('system_transcript_annotated', {})
            
            conversations.append({
                "from": "gpt",
                "value": system_text,
                "turn_id": turn_idx + 1,
                "ambiguous_type": ambiguity_label,
                "metadata": {
                    "act": system_ann.get('act', ''),
                    "objects": system_ann.get('act_attributes', {}).get('objects', []),
                    "slot_values": system_ann.get('act_attributes', {}).get('slot_values', {})
                }
            })
            
            turn_labels.append(ambiguity_label)
        
        return {
            "conversations": conversations,
            "metadata": {
                "num_turns": len(turn_labels),
                "turn_labels": turn_labels,
                "label_distribution": {
                    "clear": turn_labels.count(0),
                    "disambiguation": turn_labels.count(1)
                },
                "source": {
                    "format": "SIMMC2.0",
                    "dialogue_idx": dialogue_idx,
                    "domain": domain,
                    "scene_ids": scene_ids,
                    "mentioned_objects": dialogue_data.get('mentioned_object_ids', [])
                }
            }
        }
    
    def transform_batch(self, simmc_data: Dict) -> List[Dict]:
        """Transform entire SIMMC dataset"""
        dialogue_data = simmc_data.get('dialogue_data', [])
        
        print(f"\n[Transform] Processing {len(dialogue_data)} dialogues...")
        
        transformed = []
        errors = []
        
        for dialogue in dialogue_data:
            try:
                sip_format = self.transform_dialogue(dialogue)
                transformed.append(sip_format)
            except Exception as e:
                dialogue_idx = dialogue.get('dialogue_idx', '?')
                errors.append(f"Dialogue {dialogue_idx}: {str(e)}")
        
        print(f"  ✓ Successfully transformed {len(transformed)} dialogues")
        
        if errors:
            print(f"  ⚠ Errors in {len(errors)} dialogues:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
        
        return transformed
    
    def save(self, data: List[Dict], output_path: str):
        """Save to JSON"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[Save] Saved {len(data)} conversations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Transform SIMMC 2.0 to SIP Format')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input SIMMC 2.0 JSON (e.g., dev.json)')
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='Directory containing scene JSON files')
    parser.add_argument('--fashion_metadata', type=str, required=True,
                       help='fashion_prefab_metadata_all.json')
    parser.add_argument('--furniture_metadata', type=str, required=True,
                       help='furniture_prefab_metadata_all.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output SIP format JSON')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SIMMC 2.0 to SIP Transformation")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Scene directory: {args.scene_dir}")
    print(f"Fashion metadata: {args.fashion_metadata}")
    print(f"Furniture metadata: {args.furniture_metadata}")
    print(f"Output: {args.output}")
    print("="*70 + "\n")
    
    # Load input
    print("[Loading] Input data...")
    with open(args.input, 'r') as f:
        simmc_data = json.load(f)
    
    num_dialogues = len(simmc_data.get('dialogue_data', []))
    print(f"  ✓ Loaded {num_dialogues} dialogues")
    print(f"  Domain: {simmc_data.get('domain', 'unknown')}")
    print(f"  Split: {simmc_data.get('split', 'unknown')}")
    
    # Initialize transformer
    transformer = SIMMCToSIPTransformer(
        scene_dir=args.scene_dir,
        fashion_metadata_path=args.fashion_metadata,
        furniture_metadata_path=args.furniture_metadata
    )
    
    # Transform
    sip_data = transformer.transform_batch(simmc_data)
    
    # Save
    transformer.save(sip_data, args.output)
    
    # Print sample
    if sip_data:
        print("\n" + "="*70)
        print("Sample Output (first dialogue)")
        print("="*70)
        sample = sip_data[0]
        print(f"Dialogue index: {sample['metadata']['source']['dialogue_idx']}")
        print(f"Num turns: {sample['metadata']['num_turns']}")
        print(f"Turn labels: {sample['metadata']['turn_labels']}")
        print(f"Domain: {sample['metadata']['source']['domain']}")
        print(f"Scene changes: {list(sample['metadata']['source']['scene_ids'].keys())}")
        
        print("\nConversation structure:")
        for i, conv in enumerate(sample['conversations'][:8]):
            obs_info = f" ({conv.get('observation_type', '')})" if 'observation_type' in conv else ""
            value_preview = conv.get('value', '')[:50]
            print(f"  {i+1}. [{conv['from']:12}]{obs_info} {value_preview}...")
            
            if 'ambiguous_type' in conv:
                label_name = 'disambiguation' if conv['ambiguous_type'] == 1 else 'clear'
                print(f"      Label: {conv['ambiguous_type']} ({label_name})")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
