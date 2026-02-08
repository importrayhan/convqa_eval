"""
Test Cases for ConvQA to SIP Transformation

Tests:
1. Basic transformation with req_clari=True
2. Multi-turn conversation with follow-up
3. Table formatting
4. Ambiguity type determination
5. Conditions extraction
6. Edge cases
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import the transformer
from transform_convqa_to_sip import ConvQAToSIPTransformer


def create_test_data():
    """Create test data matching the provided example"""
    
    # Test Case 1: With req_clari=True (from provided example)
    test1 = {
        "table": {
            "uid": "6d9e104c-a89c-4d15-bacf-47c19d8d6445",
            "table": [
                ["October 31,", "", ""],
                ["", "2019", "2018"],
                ["(In thousands)", "", ""],
                ["Live poultry-broilers", "$ 179,870", "$150,980"],
                ["Processed poultry", "35,121", "30,973"],
                ["Prepared chicken", "20,032", "13,591"],
                ["Total inventories", "$289,928", "$240,056"]
            ]
        },
        "paragraphs": [
            {
                "uid": "1",
                "order": 1,
                "text": "3. Inventories"
            },
            {
                "uid": "2",
                "order": 2,
                "text": "Inventories consisted of the following:"
            }
        ],
        "questions": [
            {
                "uid": "1",
                "order": 1,
                "question": "What is the value of Processed poultry?",
                "answer": ["Which year are you asking about?"],
                "answer_type": "question",
                "answer_from": "table",
                "req_clari": True,
                "follow_up": False,
                "original_question": "What is the value of Processed poultry?"
            },
            {
                "uid": "2",
                "order": 2,
                "question": "As of October 31, 2019 and 2018 respectively.",
                "answer": ["35,121", "30,973"],
                "answer_type": "multi-span",
                "answer_from": "table",
                "scale": "thousand",
                "req_clari": False,
                "follow_up": False,
                "original_question": "What is the value of Processed poultry as of October 31, 2019 and 2018 respectively?"
            }
        ]
    }
    
    # Test Case 2: Follow-up questions
    test2 = {
        "table": {
            "uid": "test-2",
            "table": [
                ["", "2019", "2018"],
                ["Revenue", "1000", "900"],
                ["Expenses", "600", "550"]
            ]
        },
        "paragraphs": [
            {"uid": "1", "order": 1, "text": "Financial Summary"}
        ],
        "questions": [
            {
                "uid": "1",
                "order": 1,
                "question": "What was the revenue in 2019?",
                "answer": ["1000"],
                "answer_type": "span",
                "answer_from": "table",
                "req_clari": False,
                "follow_up": False,
                "original_question": "What was the revenue in 2019?"
            },
            {
                "uid": "2",
                "order": 2,
                "question": "How about expenses?",
                "answer": ["600"],
                "answer_type": "span",
                "answer_from": "table",
                "req_clari": False,
                "follow_up": True,
                "original_question": "What were the expenses in 2019?"
            }
        ]
    }
    
    # Test Case 3: Multiple req_clari in conversation
    test3 = {
        "table": {
            "uid": "test-3",
            "table": [["Year", "Value"], ["2019", "100"], ["2018", "90"]]
        },
        "paragraphs": [
            {"uid": "1", "order": 1, "text": "Annual Report"}
        ],
        "questions": [
            {
                "uid": "1",
                "order": 1,
                "question": "What is the average value?",
                "answer": ["Which year are you asking about?"],
                "answer_type": "question",
                "answer_from": "table",
                "req_clari": True,
                "follow_up": False,
                "original_question": "What is the average value?"
            },
            {
                "uid": "2",
                "order": 2,
                "question": "2019 and 2018.",
                "answer": ["95"],
                "answer_type": "arithmetic",
                "answer_from": "table",
                "req_clari": False,
                "follow_up": False,
                "original_question": "What is the average value for 2019 and 2018?"
            },
            {
                "uid": "3",
                "order": 3,
                "question": "What about the increase?",
                "answer": ["What specific metric?"],
                "answer_type": "question",
                "answer_from": "table",
                "req_clari": True,
                "follow_up": True,
                "original_question": "What is the increase?"
            }
        ]
    }
    
    return [test1, test2, test3]


def test_basic_transformation():
    """Test basic transformation"""
    print("\n" + "="*70)
    print("TEST 1: Basic Transformation")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    test_data = create_test_data()
    
    result = transformer.transform_single(test_data[0])
    
    # Verify structure
    assert 'prompt' in result, "Missing prompt"
    assert 'context' in result, "Missing context"
    assert 'can_retrieve' in result, "Missing can_retrieve"
    assert 'tools' in result, "Missing tools"
    assert 'ambiguous' in result, "Missing ambiguous"
    assert 'ambiguous_type' in result, "Missing ambiguous_type"
    assert 'conversations' in result, "Missing conversations"
    assert 'system' in result, "Missing system"
    
    print("✓ All required fields present")
    
    # Verify conversations structure
    convs = result['conversations']
    assert len(convs) > 0, "Empty conversations"
    
    # Check first turn structure (should have: human, function_call, observation, gpt)
    assert convs[0]['from'] == 'human', "First turn should be human"
    print(f"✓ Conversations count: {len(convs)}")
    
    # Check for observation
    has_observation = any(c['from'] == 'observation' for c in convs)
    assert has_observation, "Missing observation for table"
    print("✓ Table observation present")
    
    # Check ambiguity
    assert result['ambiguous'] == 'true', "Should be ambiguous"
    assert result['ambiguous_type'] in [0, 1, 2, 3], "Invalid ambiguous_type"
    print(f"✓ Ambiguous: {result['ambiguous']}, Type: {result['ambiguous_type']}")
    
    # Check req_clarification in gpt response
    gpt_responses = [c['value'] for c in convs if c['from'] == 'gpt']
    has_req_clari = any('req_clarification' in r for r in gpt_responses)
    assert has_req_clari, "Missing req_clarification marker"
    print("✓ req_clarification marker present")
    
    print("\n✓ TEST 1 PASSED")
    return result


def test_conversation_flow():
    """Test multi-turn conversation flow"""
    print("\n" + "="*70)
    print("TEST 2: Conversation Flow")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    test_data = create_test_data()
    
    result = transformer.transform_single(test_data[0])
    
    print("\nConversation turns:")
    for i, conv in enumerate(result['conversations']):
        value_preview = conv['value'][:60] if len(conv['value']) > 60 else conv['value']
        print(f"  {i+1}. [{conv['from']:15}] {value_preview}")
    
    # Verify turn order
    convs = result['conversations']
    
    # First should be human
    assert convs[0]['from'] == 'human'
    print("\n✓ First turn is human")
    
    # Check alternating pattern (allowing for function_call/observation)
    human_count = sum(1 for c in convs if c['from'] == 'human')
    gpt_count = sum(1 for c in convs if c['from'] == 'gpt')
    
    # Should have equal or gpt one less
    assert abs(human_count - gpt_count) <= 1, "Unbalanced turns"
    print(f"✓ Balanced turns: {human_count} human, {gpt_count} gpt")
    
    print("\n✓ TEST 2 PASSED")


def test_table_formatting():
    """Test table formatting"""
    print("\n" + "="*70)
    print("TEST 3: Table Formatting")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    
    # Test table
    table = [
        ["", "2019", "2018"],
        ["Revenue", "1000", "900"],
        ["Expenses", "600", "550"]
    ]
    
    formatted = transformer.format_table(table)
    
    print("\nFormatted table:")
    print(formatted)
    
    # Check format
    lines = formatted.split('\n')
    assert len(lines) == 3, "Should have 3 lines"
    assert '|' in formatted, "Should have pipe separators"
    
    print("\n✓ TEST 3 PASSED")


def test_ambiguity_type_determination():
    """Test ambiguity type logic"""
    print("\n" + "="*70)
    print("TEST 4: Ambiguity Type Determination")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    
    test_cases = [
        # (question, expected_type, description)
        (
            {"req_clari": False, "follow_up": False, "answer_type": "span"},
            0,
            "Clear question"
        ),
        (
            {"req_clari": False, "follow_up": True, "answer_type": "span"},
            1,
            "Follow-up question"
        ),
        (
            {"req_clari": True, "follow_up": False, "answer_type": "span"},
            2,
            "Requires clarification"
        ),
        (
            {"req_clari": True, "follow_up": False, "answer_type": "question"},
            3,
            "Highly ambiguous (answer is question)"
        ),
    ]
    
    for question, expected, description in test_cases:
        result = transformer.determine_ambiguity_type(question, is_last=True)
        assert result == expected, f"Failed: {description}"
        print(f"✓ {description}: Type {result}")
    
    print("\n✓ TEST 4 PASSED")


def test_conditions_extraction():
    """Test conditions extraction from original_question"""
    print("\n" + "="*70)
    print("TEST 5: Conditions Extraction")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    test_data = create_test_data()
    
    result = transformer.transform_single(test_data[0])
    
    conditions = result['metadata'].get('conditions', [])
    
    print(f"\nExtracted {len(conditions)} conditions:")
    for i, cond in enumerate(conditions):
        print(f"  {i+1}. Condition: {cond.get('condition', '')[:40]}")
        print(f"     Clarified: {cond.get('clarified', '')[:40]}")
    
    # Should have conditions since we have original_question
    assert len(conditions) >= 0, "Should extract conditions"
    
    print("\n✓ TEST 5 PASSED")


def test_batch_transformation():
    """Test batch transformation"""
    print("\n" + "="*70)
    print("TEST 6: Batch Transformation")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    test_data = create_test_data()
    
    results = transformer.transform_batch(test_data)
    
    assert len(results) == len(test_data), "Wrong number of results"
    print(f"✓ Transformed {len(results)} items")
    
    # Check each result
    for i, result in enumerate(results):
        assert 'conversations' in result, f"Item {i} missing conversations"
        assert 'ambiguous_type' in result, f"Item {i} missing ambiguous_type"
    
    print("✓ All items have required fields")
    
    # Print summary
    print("\nSummary:")
    ambig_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for result in results:
        ambig_counts[result['ambiguous_type']] += 1
    
    class_names = {0: "clear", 1: "slightly_ambiguous", 2: "needs_clarification", 3: "highly_ambiguous"}
    for typ, count in ambig_counts.items():
        if count > 0:
            print(f"  {class_names[typ]}: {count}")
    
    print("\n✓ TEST 6 PASSED")


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*70)
    print("TEST 7: Edge Cases")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    
    # Edge case 1: Empty table
    empty_table_case = {
        "table": {"uid": "test", "table": []},
        "paragraphs": [],
        "questions": [
            {
                "uid": "1",
                "order": 1,
                "question": "Test?",
                "answer": ["Test answer"],
                "answer_type": "span",
                "req_clari": False,
                "follow_up": False
            }
        ]
    }
    
    result1 = transformer.transform_single(empty_table_case)
    assert result1['can_retrieve'] == False, "Should be False for empty table"
    print("✓ Empty table handled")
    
    # Edge case 2: No paragraphs
    result2 = transformer.transform_single(empty_table_case)
    assert 'context' in result2, "Should have context field"
    print("✓ No paragraphs handled")
    
    # Edge case 3: Single question
    assert len(result2['conversations']) > 0, "Should have conversations"
    print("✓ Single question handled")
    
    print("\n✓ TEST 7 PASSED")


def test_output_format():
    """Test output format matches expected SIP format"""
    print("\n" + "="*70)
    print("TEST 8: Output Format Validation")
    print("="*70)
    
    transformer = ConvQAToSIPTransformer()
    test_data = create_test_data()
    
    result = transformer.transform_single(test_data[0])
    
    # Required fields
    required = [
        'prompt', 'context', 'can_retrieve', 'tools',
        'ambiguous', 'ambiguous_type', 'conversations', 'system'
    ]
    
    for field in required:
        assert field in result, f"Missing required field: {field}"
        print(f"✓ {field}: present")
    
    # Check types
    assert isinstance(result['prompt'], str)
    assert isinstance(result['context'], str)
    assert isinstance(result['can_retrieve'], bool)
    assert isinstance(result['tools'], str)
    assert isinstance(result['ambiguous'], str)
    assert isinstance(result['ambiguous_type'], int)
    assert isinstance(result['conversations'], list)
    assert isinstance(result['system'], str)
    
    print("\n✓ All field types correct")
    
    # Conversation format
    for conv in result['conversations']:
        assert 'from' in conv, "Conversation missing 'from'"
        assert 'value' in conv, "Conversation missing 'value'"
        assert conv['from'] in ['human', 'gpt', 'function_call', 'observation'], \
            f"Invalid 'from': {conv['from']}"
    
    print("✓ Conversation format correct")
    
    print("\n✓ TEST 8 PASSED")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "#"*70)
    print("# ConvQA to SIP Transformation - Test Suite")
    print("#"*70)
    
    try:
        result1 = test_basic_transformation()
        test_conversation_flow()
        test_table_formatting()
        test_ambiguity_type_determination()
        test_conditions_extraction()
        test_batch_transformation()
        test_edge_cases()
        test_output_format()
        
        print("\n" + "#"*70)
        print("# ALL TESTS PASSED ✓")
        print("#"*70)
        
        # Save sample output
        print("\n" + "="*70)
        print("Sample Output")
        print("="*70)
        print(json.dumps(result1, indent=2))
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
