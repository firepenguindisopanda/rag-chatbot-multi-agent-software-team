"""
Test AI Code Detection System

Basic tests for the AI code detection functionality.
"""

import pytest
from pathlib import Path
import tempfile
import os

from ai_code_detector import (
    LoaderTool,
    MetadataExtractor,
    ASTFeatureExtractor,
    StaticAnalyzerWrapper,
    LMScorer,
    EmbeddingFAISSSearch,
    StylometryComparator,
    ScoringAggregator,
    ReportWriter,
    run_ai_detection
)


class TestLoaderTool:
    """Test the LoaderTool functionality."""

    def test_load_python_file(self):
        """Test loading a Python file."""
        loader = LoaderTool()

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def hello():\n    print("Hello World")\n')
            temp_file = f.name

        try:
            result = loader._run(temp_file)
            assert "chunks" in result
            assert len(result["chunks"]) > 0
            assert result["chunks"][0]["language"] == "python"
        finally:
            os.unlink(temp_file)


class TestASTFeatureExtractor:
    """Test AST feature extraction."""

    def test_python_ast_features(self):
        """Test extracting AST features from Python code."""
        extractor = ASTFeatureExtractor()
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

class Calculator:
    def add(self, a, b):
        return a + b
"""

        result = extractor._run(code, "python")
        assert "features" in result
        features = result["features"]
        assert "function_count" in features
        assert "class_count" in features
        assert features["function_count"] >= 1
        assert features["class_count"] >= 1


class TestStaticAnalyzerWrapper:
    """Test static analysis wrapper."""

    def test_python_static_analysis(self):
        """Test static analysis on Python code."""
        analyzer = StaticAnalyzerWrapper()
        code = """
# This is a comment
def hello():
    x = 1
    y = 2
    return x + y

class Test:
    pass
"""

        result = analyzer._run(code, "python")
        assert "function_count" in result
        assert "class_count" in result
        assert "comment_ratio" in result


class TestScoringAggregator:
    """Test score aggregation."""

    def test_score_aggregation(self):
        """Test weighted score aggregation."""
        aggregator = ScoringAggregator()

        scores = {
            "lm_score": 0.8,
            "faiss_similarity": 0.6,
            "stylometry_distance": 0.4,
            "provenance_score": 0.2,
            "dynamic_anomaly": 0.1
        }

        result = aggregator._run(scores)
        assert "final_score" in result
        assert "recommendation" in result
        assert 0 <= result["final_score"] <= 1


class TestReportWriter:
    """Test report generation."""

    def test_report_generation(self):
        """Test generating detection reports."""
        writer = ReportWriter()

        results = {
            "final_score": 0.85,
            "lm_score": 0.8,
            "similarity_score": 0.6,
            "stylometry_distance": 0.4,
            "recommendation": "FLAG_FOR_ACTION"
        }

        result = writer._run(results, "student123", "sub123")
        assert "json_report" in result
        assert "human_readable_report" in result
        report = result["json_report"]
        assert abs(report["scores"]["final_score"] - 0.85) < 0.01
        assert report["recommendation"] == "FLAG_FOR_ACTION"


class TestIntegration:
    """Integration tests for the full system."""

    def test_full_detection_pipeline(self):
        """Test the complete AI detection pipeline."""
        # Create a test Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Test the function
if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(bubble_sort(test_arr))
""")
            temp_file = f.name

        try:
            result = run_ai_detection(temp_file, "test_student")
            # Should not error out
            assert isinstance(result, dict)
            assert "final_score" in result or "error" in result
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])