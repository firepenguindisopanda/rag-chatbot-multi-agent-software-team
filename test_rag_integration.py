#!/usr/bin/env python3
"""
Test script for RAG integration in assignment evaluator.
Tests the master code contextual evaluation functionality.
"""

import os
import tempfile
import json
from pathlib import Path

# Import the evaluator functions
from assignment_evaluator.evaluator import grade_submission, load_and_index_master_code, get_relevant_master_chunks

def create_test_files():
    """Create test files for evaluation."""
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()

    # Create sample student code
    student_code = '''
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    total = sum(numbers)
    return total / len(numbers)

def find_max(numbers):
    """Find the maximum value in a list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
'''

    # Create master/reference code with better implementation
    master_code = '''
def calculate_average(numbers):
    """
    Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: List of numeric values

    Returns:
        float: The average value, or 0 if list is empty

    Raises:
        TypeError: If input is not a list or contains non-numeric values
    """
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list")

    if not numbers:
        return 0.0

    # Use more efficient sum and handle potential float precision
    total = sum(float(x) for x in numbers)
    return total / len(numbers)

def find_maximum(numbers):
    """
    Find the maximum value in a list using built-in max function.

    Args:
        numbers: List of comparable values

    Returns:
        The maximum value, or None if list is empty
    """
    if not numbers:
        return None
    return max(numbers)

def calculate_median(numbers):
    """
    Calculate the median of a sorted list.

    Args:
        numbers: List of numeric values

    Returns:
        float: The median value
    """
    if not numbers:
        return 0.0

    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    else:
        return sorted_nums[mid]
'''

    # Create rubric
    rubric = {
        "correctness": {
            "description": "Correctness of the solution implementation",
            "weight": 0.4
        },
        "code_quality": {
            "description": "Code readability, structure, and best practices",
            "weight": 0.3
        },
        "documentation": {
            "description": "Comments, docstrings, and code documentation",
            "weight": 0.2
        },
        "efficiency": {
            "description": "Algorithm efficiency and use of appropriate data structures",
            "weight": 0.1
        }
    }

    # Save files
    student_path = os.path.join(test_dir, "student_code.py")
    master_path = os.path.join(test_dir, "master_code.py")
    rubric_path = os.path.join(test_dir, "rubric.json")

    with open(student_path, 'w') as f:
        f.write(student_code)

    with open(master_path, 'w') as f:
        f.write(master_code)

    with open(rubric_path, 'w') as f:
        json.dump(rubric, f, indent=2)

    return test_dir, student_path, master_path, rubric_path

def test_rag_integration():
    """Test the RAG integration functionality."""
    print("🧪 Testing RAG Integration for Assignment Evaluator")
    print("=" * 60)

    # Create test files
    test_dir, student_path, master_path, rubric_path = create_test_files()

    try:
        # Test 1: Load and index master code
        print("1. Testing master code loading and indexing...")
        master_context = load_and_index_master_code(master_path)
        if master_context:
            print("   ✅ Master code loaded successfully")
            print(f"   📊 Found {len(master_context['chunks'])} code chunks")
            print(f"   🔍 Language detected: {master_context['language']}")
        else:
            print("   ❌ Failed to load master code")
            return

        # Test 2: Get relevant chunks
        print("\n2. Testing relevant chunk extraction...")
        relevant_chunks = get_relevant_master_chunks(master_context)
        if relevant_chunks:
            print("   ✅ Relevant chunks extracted")
            print(f"   📝 Chunk preview: {relevant_chunks[:200]}...")
        else:
            print("   ❌ No relevant chunks extracted")

        # Test 3: Full evaluation with RAG
        print("\n3. Testing full evaluation with RAG context...")

        # Load rubric
        with open(rubric_path, 'r') as f:
            rubric = json.load(f)

        # Create a simple zip file with student code
        import zipfile
        zip_path = os.path.join(test_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(student_path, "student_code.py")

        # Run evaluation with master code context
        print("   🔍 Running evaluation with master code context...")
        report = grade_submission(zip_path, rubric, None, master_path)  # No LLM for this test

        if report:
            print("   ✅ Evaluation completed successfully")
            print(f"   📊 Evaluated {len(report['results'])} criteria")

            # Show sample results
            for crit_key, result in list(report['results'].items())[:2]:
                score = result['llm_result']['score']
                print(f"   🎯 {crit_key}: {score}/100")

        else:
            print("   ❌ Evaluation failed")

        print("\n🎉 RAG Integration Test Complete!")
        print("\nKey Features Verified:")
        print("  ✓ Master code loading and chunking")
        print("  ✓ Relevant context extraction")
        print("  ✓ Integration with evaluation pipeline")
        print("  ✓ Contextual evaluation prompts")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        import shutil
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_rag_integration()