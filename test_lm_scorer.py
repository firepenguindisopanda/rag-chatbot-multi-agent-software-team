#!/usr/bin/env python3
"""
Test script for the enhanced LM Scorer implementation.
Tests the new multi-method evaluation approach with different programming languages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_code_detector.tools import LMScorer

def test_lm_scorer():
    """Test the LM Scorer with various code samples."""

    # Initialize the LM scorer
    lm_scorer = LMScorer()

    # Test cases with different languages and characteristics
    test_cases = [
        {
            "name": "Simple C++ Calculator",
            "language": "cpp",
            "code": '''
#include <iostream>
#include <vector>

class Calculator {
private:
    int value;
public:
    Calculator(int v = 0) : value(v) {}
    int add(int x) { return value + x; }
    int multiply(int x) { return value * x; }
};

int main() {
    Calculator calc(10);
    std::cout << "Result: " << calc.add(5) << std::endl;
    return 0;
}
'''
        },
        {
            "name": "JavaScript Function",
            "language": "javascript",
            "code": '''
function calculateTotal(items) {
    let total = 0;
    for (let item of items) {
        total += item.price * item.quantity;
    }
    return total;
}

const items = [
    { name: 'Apple', price: 1.50, quantity: 3 },
    { name: 'Banana', price: 0.75, quantity: 2 }
];

console.log('Total:', calculateTotal(items));
'''
        },
        {
            "name": "Java Class",
            "language": "java",
            "code": '''
public class Student {
    private String name;
    private int age;
    private List<Double> grades;

    public Student(String name, int age) {
        this.name = name;
        this.age = age;
        this.grades = new ArrayList<>();
    }

    public void addGrade(double grade) {
        if (grade >= 0 && grade <= 100) {
            grades.add(grade);
        }
    }

    public double getAverageGrade() {
        if (grades.isEmpty()) return 0.0;
        return grades.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}
'''
        },
        {
            "name": "Simple Python Script",
            "language": "python",
            "code": '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    # Print first 10 Fibonacci numbers
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()
'''
        }
    ]

    print("🧪 Testing Enhanced LM Scorer Implementation")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['name']}")
        print("-" * 40)

        try:
            # Run the LM scorer
            result = lm_scorer._run(test_case['code'], test_case['language'])

            # Display results
            print(f"Language: {result.get('language', 'N/A')}")
            print(".3f")
            print(".3f")
            print(f"Evaluation Methods: {result.get('evaluation_methods', 0)}")

            # Show component scores
            component_scores = result.get('component_scores', {})
            if component_scores:
                print("\nComponent Scores:")
                for method, score in component_scores.items():
                    print(".3f")

            # Show error if any
            if 'error' in result:
                print(f"❌ Error: {result['error']}")

        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ LM Scorer testing completed!")

if __name__ == "__main__":
    test_lm_scorer()