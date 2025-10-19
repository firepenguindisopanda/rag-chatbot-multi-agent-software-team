"""
AI-Generated Code Detection System

This module implements an ensemble approach to detect AI-generated code submissions
using static analysis, language model scoring, embedding similarity, and behavioral testing.

Following design principles:
- Cohesion & Single Responsibility: Each tool has one clear purpose
- Encapsulation & Abstraction: Hide implementation details behind clean interfaces
- Loose Coupling & Modularity: Tools are independent and composable
- Reusability & Extensibility: Open for extension via plugin patterns
- Portability & Configurability: Cross-platform, configurable thresholds
- Defensibility & Security: Safe execution, input validation
- Maintainability & Testability: Well-structured, unit testable components
- Simplicity: KISS, DRY, YAGNI - avoid unnecessary complexity
- Observability & Error Handling: Structured logging, graceful failures
"""

from .tools import (
    LoaderTool,
    MetadataExtractor,
    ASTFeatureExtractor,
    StaticAnalyzerWrapper,
    LMScorer,
    EmbeddingFAISSSearch,
    StylometryComparator,
    SandboxTester,
    ScoringAggregator,
    ReportWriter
)
from .agent import AICodeDetectionAgent, run_ai_detection

__all__ = [
    'LoaderTool',
    'MetadataExtractor',
    'ASTFeatureExtractor',
    'StaticAnalyzerWrapper',
    'LMScorer',
    'EmbeddingFAISSSearch',
    'StylometryComparator',
    'SandboxTester',
    'ScoringAggregator',
    'ReportWriter',
    'AICodeDetectionAgent',
    'run_ai_detection'
]