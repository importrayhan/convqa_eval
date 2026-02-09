"""
Revised ConvQA to SIP Transformation with Paragraphs as Observations

CRITICAL CHANGES:
1. Paragraphs included as observations in second turn
2. Per-GPT labels maintained
3. Observations treated as context input

Output Format:
{
  "conversations": [
    {"from": "human", "value": "...", "turn_id": 1},
    {"from": "function_call", "value": "retrieve_table()", "turn_id": 1},
    {"from": "observation", "value": "table data...", "turn_id": 1},
    {"from": "gpt", "value": "...", "turn_id": 1, "ambiguous_type": 2},
    {"from": "human", "value": "...", "turn_id": 2},
    {"from": "function_call", "value": "retrieve_context()", "turn_id": 2},
    {"from": "observation", "value": "paragraph text...", "turn_id": 2},
    {"from": "gpt", "value": "...", "turn_id": 2, "ambiguous_type": 0}
  ]
}
"""

import json
import argparse
from typing import Dict, List, Tuple
from pathlib import Path


class ConvQAToSIPTransformer:
    """Transform ConvQA with per-GPT labels and paragraphs as observations"""
    
    def __init__(self):
        self.class_names = {
            0: "clear",
            1: "slightly_ambiguous",
            2: "needs_clarification",
            3: "highly_ambiguous"
        }
    
    def format_table(self, table_data: List[List[str]], max_rows: int = 10) -> str:
        """Format table as text"""
        if not table_data:
            return ""
        
        rows = table_data[:max_rows]
        lines = []
        for row in rows:
            cells = [str(cell).strip() for cell in row if str(cell).strip()]
            if cells:
                lines.append(" | ".join(cells))
        
        return "\n".join(lines)
    
    def extract_context(self, paragraphs: List[Dict]) -> str:
        """Extract context from paragraphs"""
        if not paragraphs:
            return ""
        
        sorted_paras = sorted(paragraphs, key=lambda p: p.get('order', 0))
        texts = [p.get('text', '') for p in sorted_paras]
        return "\n\n".join(texts)
    
    def determine_ambiguity_type(self, question: Dict) -> int:
        """
        Determine ambiguity type for THIS specific question.
        
        Priority:
        1. answer_type='question' → 3
        2. req_clari=True → 2
        3. follow_up=True → 1
        4. Otherwise → 0
        """
        answer_type = question.get('answer_type', '')
        req_clari = question.get('req_clari', False)
        follow_up = question.get('follow_up', False)
        
        if answer_type == 'question':
            return 3
        if req_clari:
            return 2
        if follow_up:
            return 1
        return 0
    
    def format_answer(self, question: Dict) -> str:
        """Format answer from question"""
        answer = question.get('answer', [])
        
        if isinstance(answer, list):
            if len(answer) == 1:
                return str(answer[0])
            return ", ".join(str(a) for a in answer)
        
        return str(answer)
    
    def create_gpt_utterance(
        self,
        question: Dict,
        turn_id: int
    ) -> Dict:
        """
        Create GPT utterance with PER-TURN label and metadata.
        
        CRITICAL: Each GPT gets its own ambiguous_type!
        """
        # Determine ambiguity type for THIS turn
        ambig_type = self.determine_ambiguity_type(question)
        
        # Create response
        req_clari = question.get('req_clari', False)
        answer_type = question.get('answer_type', '')
        
        if req_clari or answer_type == 'question':
            # Clarification request
            answer = question.get('answer', [])
            if isinstance(answer, list) and answer:
                clari_text = str(answer[0])
            else:
                clari_text = "Could you please clarify?"
            
            gpt_value = f"{clari_text} req_clarification({ambig_type})"
        else:
            # Normal answer
            gpt_value = self.format_answer(question)
            
            # Add scale if present
            scale = question.get('scale', '')
            if scale:
                gpt_value = f"{gpt_value} ({scale})"
        
        return {
            "from": "gpt",
            "value": gpt_value,
            "turn_id": turn_id,
            "ambiguous_type": ambig_type,  # PER-TURN LABEL!
            "metadata": {
                "answers": question.get('answer', []),
                "original_question": question.get('original_question', ''),
                "answer_type": question.get('answer_type', ''),
                "answer_from": question.get('answer_from', ''),
                "scale": question.get('scale', ''),
                "derivation": question.get('derivation', ''),
                "req_clari": question.get('req_clari', False),
                "follow_up": question.get('follow_up', False)
            }
        }
    
    def build_conversations(
        self,
        questions: List[Dict],
        table: Dict,
        paragraphs: List[Dict]
    ) -> Tuple[List[Dict], List[int]]:
        """
        Build conversations with per-turn labels.
        
        IMPORTANT CHANGES:
        - First turn: Table retrieval if available
        - Second turn: Paragraph retrieval if available
        - Observations provide context for model
        
        Returns:
            conversations: List of conversation turns
            turn_labels: List of labels (one per system turn)
        """
        conversations = []
        turn_labels = []
        
        for turn_idx, question in enumerate(questions, start=1):
            user_question = question.get('question', '')
            
            # Human turn
            conversations.append({
                "from": "human",
                "value": user_question,
                "turn_id": turn_idx
            })
            
            # First turn: Add table retrieval if available
            if turn_idx == 1 and table and table.get('table'):
                table_data = table.get('table', [])
                if table_data:
                    conversations.append({
                        "from": "function_call",
                        "value": "retrieve_table()",
                        "turn_id": turn_idx
                    })
                    
                    formatted_table = self.format_table(table_data)
                    conversations.append({
                        "from": "observation",
                        "value": formatted_table,
                        "turn_id": turn_idx,
                        "observation_type": "table"
                    })
            
            # Second turn: Add paragraph retrieval if available
            if turn_idx == 2 and paragraphs:
                paragraph_text = self.extract_context(paragraphs)
                if paragraph_text:
                    conversations.append({
                        "from": "function_call",
                        "value": "retrieve_context()",
                        "turn_id": turn_idx
                    })
                    
                    conversations.append({
                        "from": "observation",
                        "value": paragraph_text,
                        "turn_id": turn_idx,
                        "observation_type": "context"
                    })
            
            # GPT turn with per-turn label
            gpt_utt = self.create_gpt_utterance(question, turn_idx)
            conversations.append(gpt_utt)
            
            # Collect label
            turn_labels.append(gpt_utt['ambiguous_type'])
        
        return conversations, turn_labels
    
    def transform_single(self, convqa_data: Dict) -> Dict:
        """
        Transform single ConvQA item to SIP format with per-GPT labels.
        """
        table = convqa_data.get('table', {})
        paragraphs = convqa_data.get('paragraphs', [])
        questions = convqa_data.get('questions', [])
        
        # Build conversations with per-turn labels
        conversations, turn_labels = self.build_conversations(questions, table, paragraphs)
        
        # Extract context
        context = self.extract_context(paragraphs)
        
        # Create prompt
        prompt = ""
        if paragraphs and len(paragraphs) > 0:
            prompt = paragraphs[0].get('text', '')[:100]
        
        # Build SIP format
        sip_data = {
            "conversations": conversations,
            "metadata": {
                "num_turns": len(turn_labels),
                "turn_labels": turn_labels,
                "label_distribution": {
                    self.class_names[i]: turn_labels.count(i)
                    for i in range(4)
                },
                "source": {
                    "format": "ConvQA",
                    "table_uid": table.get('uid', ''),
                    "num_questions": len(questions),
                    "has_table": bool(table.get('table')),
                    "has_paragraphs": bool(paragraphs),
                    "num_observations": sum(1 for c in conversations if c['from'] == 'observation')
                },
                "context": context,
                "prompt": prompt or "Answer questions about table and text"
            }
        }
        
        return sip_data
    
    def transform_batch(self, convqa_list: List[Dict]) -> List[Dict]:
        """Transform batch"""
        return [self.transform_single(item) for item in convqa_list]
    
    def save_to_file(self, data: List[Dict], output_path: str):
        """Save to JSON"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} items to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Transform ConvQA to SIP (Per-GPT Labels + Paragraphs)')
    parser.add_argument('--input', required=True, help='Input ConvQA JSON')
    parser.add_argument('--output', required=True, help='Output SIP JSON')
    args = parser.parse_args()
    
    # Load
    print(f"Loading from {args.input}...")
    with open(args.input, 'r') as f:
        convqa_data = json.load(f)
    
    if isinstance(convqa_data, dict):
        convqa_data = [convqa_data]
    
    # Transform
    print(f"Transforming {len(convqa_data)} items...")
    transformer = ConvQAToSIPTransformer()
    sip_data = transformer.transform_batch(convqa_data)
    
    # Save
    transformer.save_to_file(sip_data, args.output)
    
    # Print sample
    if sip_data:
        print("\nSample output (first item):")
        sample = sip_data[0]
        print(f"  Num turns: {sample['metadata']['num_turns']}")
        print(f"  Turn labels: {sample['metadata']['turn_labels']}")
        print(f"  Num observations: {sample['metadata']['source']['num_observations']}")
        print(f"  Label distribution: {sample['metadata']['label_distribution']}")
        
        print("\n  Conversation structure:")
        for i, conv in enumerate(sample['conversations'][:10]):
            obs_type = f" ({conv.get('observation_type', '')})" if 'observation_type' in conv else ""
            print(f"    {i+1}. [{conv['from']:12}]{obs_type} {conv.get('value', '')[:50]}...")
            if 'ambiguous_type' in conv:
                print(f"        Label: {conv['ambiguous_type']}")


if __name__ == "__main__":
    main()
