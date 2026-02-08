"""
Transform ConvQA JSON to SIP Format

FIELD MAPPING:
==============

ConvQA → SIP:
-------------
table.table → Used in observation field (formatted)
paragraphs → context field (concatenated)
questions → conversations list (multi-turn dialogue)
questions[i].question → human value
questions[i].answer → gpt value or req_clarification
questions[i].req_clari → Determines if gpt requests clarification
questions[i].original_question → Used for conditions generation
questions[-1].req_clari or answer_type → ambiguous_type label

Conversation Structure:
-----------------------
Turn 1: human asks → gpt responds (or req_clarification)
Turn 2: human clarifies → gpt responds
...
Turn N: human asks → gpt responds (LABELED)

For first turn with table:
- human: question
- function_call: "retrieve_table()"
- observation: table content (formatted)
- gpt: answer or req_clarification

Label Extraction:
-----------------
- If req_clari=True → ambiguous_type=2 (needs clarification)
- If answer_type='question' → ambiguous_type=3 (highly ambiguous)
- If follow_up=True → ambiguous_type=1 (slightly ambiguous)
- Otherwise → ambiguous_type=0 (clear)
"""

import json
import argparse
from typing import Dict, List, Tuple, Any
from pathlib import Path


class ConvQAToSIPTransformer:
    """Transform ConvQA format to SIP format"""
    
    def __init__(self):
        # Class name mapping for 4-class configuration
        self.class_names = {
            0: "clear",
            1: "slightly_ambiguous",  # follow_up questions
            2: "needs_clarification",  # req_clari=True
            3: "highly_ambiguous"      # answer_type='question'
        }
    
    def format_table(self, table_data: List[List[str]], max_rows: int = 10) -> str:
        """
        Format table data as readable string for observation field.
        
        Args:
            table_data: 2D array of table content
            max_rows: Maximum rows to include
        
        Returns:
            Formatted table string
        """
        if not table_data:
            return ""
        
        # Take first max_rows
        rows = table_data[:max_rows]
        
        # Simple text table format
        lines = []
        for row in rows:
            # Clean and join cells
            cells = [str(cell).strip() for cell in row if str(cell).strip()]
            if cells:
                lines.append(" | ".join(cells))
        
        return "\n".join(lines)
    
    def extract_context(self, paragraphs: List[Dict]) -> str:
        """
        Extract context from paragraphs.
        
        Args:
            paragraphs: List of paragraph objects
        
        Returns:
            Concatenated context string
        """
        if not paragraphs:
            return ""
        
        # Sort by order and concatenate
        sorted_paras = sorted(paragraphs, key=lambda p: p.get('order', 0))
        texts = [p.get('text', '') for p in sorted_paras]
        return " ".join(texts)
    
    def determine_ambiguity_type(self, question: Dict, is_last: bool = False) -> int:
        """
        Determine ambiguity type from question metadata.
        
        Priority:
        1. answer_type='question' → 3 (highly ambiguous)
        2. req_clari=True → 2 (needs clarification)
        3. follow_up=True → 1 (slightly ambiguous)
        4. Otherwise → 0 (clear)
        
        Args:
            question: Question object
            is_last: Whether this is the last question (to be labeled)
        
        Returns:
            Ambiguity type (0-3)
        """
        # For last question, use its metadata to determine label
        if is_last:
            answer_type = question.get('answer_type', '')
            req_clari = question.get('req_clari', False)
            follow_up = question.get('follow_up', False)
            
            # Highest priority: answer is a question
            if answer_type == 'question':
                return 3
            
            # Requires clarification
            if req_clari:
                return 2
            
            # Follow-up question
            if follow_up:
                return 1
            
            # Clear
            return 0
        
        # For non-last questions, same logic
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
        """
        Format answer from question object.
        
        Args:
            question: Question object
        
        Returns:
            Formatted answer string
        """
        answer = question.get('answer', [])
        
        # If answer is a list
        if isinstance(answer, list):
            if len(answer) == 1:
                return str(answer[0])
            else:
                # Multiple values
                return ", ".join(str(a) for a in answer)
        
        # If answer is a number/string
        return str(answer)
    
    def create_gpt_response(self, question: Dict) -> str:
        """
        Create GPT response based on question metadata.
        
        If req_clari=True or answer_type='question':
            Return clarification request with label
        Otherwise:
            Return the answer
        
        Args:
            question: Question object
        
        Returns:
            GPT response string
        """
        req_clari = question.get('req_clari', False)
        answer_type = question.get('answer_type', '')
        
        # Determine ambiguity level for this response
        ambig_type = self.determine_ambiguity_type(question)
        
        # If requires clarification or answer is question
        if req_clari or answer_type == 'question':
            # Get the clarification question from answer
            answer = question.get('answer', [])
            if isinstance(answer, list) and answer:
                clarification_text = str(answer[0])
            else:
                clarification_text = "Could you please clarify?"
            
            # Include req_clarification label
            return f"{clarification_text} req_clarification({ambig_type})"
        
        # Otherwise, return the actual answer
        return self.format_answer(question)
    
    def build_conversations(
        self,
        questions: List[Dict],
        table: Dict,
        include_table: bool = True
    ) -> Tuple[List[Dict], int, List[Dict]]:
        """
        Build conversations list from questions.
        
        Args:
            questions: List of question objects
            table: Table data
            include_table: Whether to include table retrieval in first turn
        
        Returns:
            Tuple of (conversations, ambiguous_type, conditions)
        """
        conversations = []
        conditions = []
        
        # Process each question as a turn
        for i, question in enumerate(questions):
            user_question = question.get('question', '')
            
            # Add human turn
            conversations.append({
                "from": "human",
                "value": user_question
            })
            
            # For first question with table, add table retrieval
            if i == 0 and include_table and table:
                table_data = table.get('table', [])
                if table_data:
                    # Function call
                    conversations.append({
                        "from": "function_call",
                        "value": "retrieve_table()"
                    })
                    
                    # Observation with table content
                    formatted_table = self.format_table(table_data)
                    conversations.append({
                        "from": "observation",
                        "value": formatted_table
                    })
            
            # Add GPT response
            gpt_response = self.create_gpt_response(question)
            conversations.append({
                "from": "gpt",
                "value": gpt_response
            })
            
            # Extract conditions from original_question if different from question
            original = question.get('original_question', '')
            if original and original != user_question:
                conditions.append({
                    "condition": user_question,
                    "clarified": original
                })
        
        # Determine final ambiguity type from last question
        if questions:
            ambiguous_type = self.determine_ambiguity_type(questions[-1], is_last=True)
        else:
            ambiguous_type = 0
        
        return conversations, ambiguous_type, conditions
    
    def transform_single(self, convqa_data: Dict) -> Dict:
        """
        Transform a single ConvQA item to SIP format.
        
        Args:
            convqa_data: Single ConvQA data item
        
        Returns:
            SIP formatted data
        """
        table = convqa_data.get('table', {})
        paragraphs = convqa_data.get('paragraphs', [])
        questions = convqa_data.get('questions', [])
        
        # Build conversations
        conversations, ambiguous_type, conditions = self.build_conversations(
            questions, table
        )
        
        # Extract context from paragraphs
        context = self.extract_context(paragraphs)
        
        # Create prompt from first paragraph title
        prompt = ""
        if paragraphs and len(paragraphs) > 0:
            prompt = paragraphs[0].get('text', '')[:100]  # First 100 chars
        
        # Determine if ambiguous
        is_ambiguous = ambiguous_type > 0
        
        # Build SIP format
        sip_data = {
            "prompt": prompt or "Answer questions about table and text",
            "context": context,
            "can_retrieve": bool(table),
            "tools": "retrieve_table(), req_clarification(2-4)",
            "ambiguous": str(is_ambiguous).lower(),
            "ambiguous_type": ambiguous_type,
            "conversations": conversations,
            "system": "You are a helpful assistant that answers questions about financial tables and documents",
            "metadata": {
                "table_uid": table.get('uid', ''),
                "num_questions": len(questions),
                "conditions": conditions,
                "original_format": "ConvQA"
            }
        }
        
        return sip_data
    
    def transform_batch(self, convqa_list: List[Dict]) -> List[Dict]:
        """
        Transform a list of ConvQA items.
        
        Args:
            convqa_list: List of ConvQA data items
        
        Returns:
            List of SIP formatted data
        """
        return [self.transform_single(item) for item in convqa_list]
    
    def save_to_file(self, data: List[Dict], output_path: str):
        """Save transformed data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} items to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Transform ConvQA to SIP format')
    parser.add_argument('--input', required=True, help='Input ConvQA JSON file')
    parser.add_argument('--output', required=True, help='Output SIP JSON file')
    args = parser.parse_args()
    
    # Load input
    print(f"Loading from {args.input}...")
    with open(args.input, 'r') as f:
        convqa_data = json.load(f)
    
    # Ensure list
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
        print(json.dumps(sip_data[0], indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
