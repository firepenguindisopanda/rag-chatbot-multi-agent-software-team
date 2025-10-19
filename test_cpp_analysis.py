#!/usr/bin/env python3

from ai_code_detector.tools import ASTFeatureExtractor, StaticAnalyzerWrapper

# Test C++ analysis
cpp_code = '''
#include <iostream>
using namespace std;

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Calculator calc;
    cout << calc.add(5, 3) << endl;
    return 0;
}
'''

def test_cpp_analysis():
    ast_extractor = ASTFeatureExtractor()
    static_analyzer = StaticAnalyzerWrapper()

    print('Testing C++ AST Features:')
    ast_result = ast_extractor._run(cpp_code, 'cpp')
    print(ast_result)

    print('\nTesting C++ Static Analysis:')
    static_result = static_analyzer._run(cpp_code, 'cpp')
    print(static_result)

if __name__ == "__main__":
    test_cpp_analysis()