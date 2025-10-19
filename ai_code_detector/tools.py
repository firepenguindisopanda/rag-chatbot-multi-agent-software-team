"""
AI Code Detection Tools

Implementation of LangChain tools for detecting AI-generated code.
Each tool follows single responsibility principle and is designed for modularity.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile
import zipfile
import os
import json
from datetime import datetime
import re

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

# Regex patterns for code analysis (shared constants)
CPP_FUNCTION_PATTERN = r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
JS_FUNCTION_PATTERN = r'\bfunction\s+\w+\s*\('
JS_ARROW_FUNCTION_PATTERN = r'\w+\s*=\s*\([^)]*\)\s*=>'
JAVA_METHOD_PATTERN = r'\b(?:public|private|protected)?\s*\w+\s+\w+\s*\([^)]*\)'
PYTHON_FUNCTION_PATTERN = r'\bdef\s+\w+\s*\('
CLASS_PATTERN = r'\b(?:class|struct|interface)\s+\w+'
CONTROL_STRUCTURE_PATTERN = r'\b(?:if|for|while|switch)\s*\('
CPP_TEMPLATE_PATTERN = r'\btemplate\s*<'
CPP_STD_PATTERN = r'\bstd::'
JS_ASYNC_PATTERN = r'\basync\s+|\bawait\s+'
JS_MODULE_PATTERN = r'\bimport\s+|\bexport\s+'
JAVA_STREAM_PATTERN = r'\bStream\b|\bCollectors\b'
JAVA_OVERRIDE_PATTERN = r'\b@Override\b'

# Check tree-sitter availability per language
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import tree_sitter_cpp
    TREE_SITTER_CPP_AVAILABLE = True
except ImportError:
    TREE_SITTER_CPP_AVAILABLE = False

try:
    import tree_sitter_javascript
    TREE_SITTER_JS_AVAILABLE = True
except ImportError:
    TREE_SITTER_JS_AVAILABLE = False

try:
    import tree_sitter_java
    TREE_SITTER_JAVA_AVAILABLE = True
except ImportError:
    TREE_SITTER_JAVA_AVAILABLE = False

try:
    import tree_sitter_python
    TREE_SITTER_PYTHON_AVAILABLE = True
except ImportError:
    TREE_SITTER_PYTHON_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class LoaderToolInput(BaseModel):
    """Input schema for LoaderTool."""
    submission_path: str = Field(description="Path to submission file or directory")
    language: Optional[str] = Field(default=None, description="Programming language hint")


class LoaderTool(BaseTool):
    """Tool to load and tag code submissions with metadata."""

    name: str = "loader_tool"
    description: str = "Load code submission and extract language-tagged code chunks with metadata"
    args_schema: type[BaseModel] = LoaderToolInput

    def _run(self, submission_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Load submission and return structured code chunks."""
        try:
            # Handle ZIP files (common for submissions)
            if submission_path.endswith('.zip'):
                with tempfile.TemporaryDirectory() as tmp:
                    files = self._unzip_submission(submission_path, tmp)
                    chunks = []
                    for fp in files:
                        file_chunks = self._process_file(fp, language)
                        chunks.extend(file_chunks)
                    return {
                        "chunks": chunks,
                        "metadata": {
                            "source_type": "zip",
                            "original_path": submission_path,
                            "extracted_files": len(files)
                        }
                    }
            else:
                # Handle single file or directory
                chunks = self._process_file(submission_path, language)
                return {
                    "chunks": chunks,
                    "metadata": {
                        "source_type": "file",
                        "original_path": submission_path
                    }
                }
        except Exception as e:
            return {
                "error": f"Failed to load submission: {str(e)}",
                "chunks": [],
                "metadata": {}
            }

    def _unzip_submission(self, zip_path: str, dest_dir: str) -> List[str]:
        """Unzip submission safely."""
        if os.path.getsize(zip_path) > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("ZIP file too large (max 50MB)")

        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if ".." in name or name.startswith("/"):
                    raise ValueError(f"Potentially dangerous file path: {name}")
            z.extractall(dest_dir)

        files = []
        for p in Path(dest_dir).rglob("*"):
            if p.is_file() and p.stat().st_size <= 10 * 1024 * 1024:  # 10MB per file
                if p.suffix in {".py", ".java", ".js", ".cpp", ".c", ".ts"}:
                    files.append(str(p))
        return files

    def _process_file(self, file_path: str, language_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a single file into chunks."""
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return []

            detected_lang = language_hint or self._detect_language(file_path)

            # Simple chunking - could be enhanced with AST-aware chunking
            lines = content.splitlines()
            chunks = []
            chunk_size = 200

            for i in range(0, len(lines), chunk_size):
                chunk_text = "\n".join(lines[i:i+chunk_size])
                chunks.append({
                    "file_path": file_path,
                    "language": detected_lang,
                    "text": chunk_text,
                    "start_line": i + 1,
                    "end_line": min(i + chunk_size, len(lines)),
                    "total_lines": len(lines)
                })

            return chunks
        except Exception as e:
            return [{
                "file_path": file_path,
                "language": "unknown",
                "text": "",
                "error": str(e),
                "start_line": 1,
                "end_line": 1,
                "total_lines": 0
            }]

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".js": "javascript",
            ".ts": "typescript",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust"
        }
        return lang_map.get(ext, "unknown")


class MetadataExtractorInput(BaseModel):
    """Input schema for MetadataExtractor."""
    file_path: str = Field(description="Path to the file to extract metadata from")
    include_git_history: bool = Field(default=True, description="Whether to include git history")


class MetadataExtractor(BaseTool):
    """Tool to extract metadata from code files including timestamps and git history."""

    name: str = "metadata_extractor"
    description: str = "Extract timestamps, file metadata, and git history from code files"
    args_schema: type[BaseModel] = MetadataExtractorInput

    def _run(self, file_path: str, include_git_history: bool = True) -> Dict[str, Any]:
        """Extract metadata from file."""
        try:
            path = Path(file_path)
            stat = path.stat()

            metadata = {
                "file_path": str(path),
                "file_size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_extension": path.suffix,
                "language": self._detect_language(str(path))
            }

            if include_git_history:
                git_metadata = self._extract_git_metadata(str(path))
                metadata.update(git_metadata)

            return metadata
        except Exception as e:
            return {
                "file_path": file_path,
                "error": f"Failed to extract metadata: {str(e)}"
            }

    def _extract_git_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract git-related metadata."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "log", "--follow", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso", "--", file_path],
                capture_output=True, text=True, cwd=Path(file_path).parent
            )

            if result.returncode == 0 and result.stdout.strip():
                commits = []
                for line in result.stdout.strip().split('\n'):
                    if '|' in line:
                        parts = line.split('|', 4)
                        if len(parts) >= 5:
                            commits.append({
                                "hash": parts[0],
                                "author": parts[1],
                                "email": parts[2],
                                "date": parts[3],
                                "message": parts[4]
                            })

                return {
                    "git_commits": commits,
                    "commit_count": len(commits),
                    "first_commit_date": commits[-1]["date"] if commits else None,
                    "last_commit_date": commits[0]["date"] if commits else None,
                    "authors": list(set(c["author"] for c in commits)),
                    "avg_commit_size": self._calculate_avg_commit_size(file_path)
                }
            else:
                return {"git_commits": [], "commit_count": 0}
        except Exception:
            return {"git_commits": [], "commit_count": 0}

    def _calculate_avg_commit_size(self, file_path: str) -> float:
        """Calculate average commit size in lines."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "log", "--follow", "--pretty=format:%H", "--", file_path],
                capture_output=True, text=True, cwd=Path(file_path).parent
            )

            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                total_changes = 0
                for commit in commits[:10]:  # Sample first 10 commits
                    if commit:
                        diff_result = subprocess.run(
                            ["git", "show", "--stat", commit, "--", file_path],
                            capture_output=True, text=True, cwd=Path(file_path).parent
                        )
                        if diff_result.returncode == 0:
                            # Parse diff stat for line changes
                            for line in diff_result.stdout.split('\n'):
                                if 'insertion' in line or 'deletion' in line:
                                    numbers = re.findall(r'\d+', line)
                                    total_changes += sum(int(n) for n in numbers)
                return total_changes / max(len(commits), 1)
        except Exception:
            pass
        return 0.0

    def _detect_language(self, file_path: str) -> str:
        """Detect language from extension."""
        ext = Path(file_path).suffix.lower()
        lang_map = {".py": "python", ".java": "java", ".js": "javascript", ".cpp": "cpp", ".c": "c", ".ts": "typescript"}
        return lang_map.get(ext, "unknown")


class ASTFeatureExtractorInput(BaseModel):
    """Input schema for ASTFeatureExtractor."""
    code: str = Field(description="Code text to analyze")
    language: str = Field(description="Programming language")


class ASTFeatureExtractor(BaseTool):
    """Tool to extract AST-based features from code using tree-sitter."""

    name: str = "ast_feature_extractor"
    description: str = "Extract AST statistics and features from code using tree-sitter"
    args_schema: type[BaseModel] = ASTFeatureExtractorInput

    def _run(self, code: str, language: str) -> Dict[str, Any]:
        """Extract AST features from code."""
        try:
            if language == "python":
                return self._extract_python_features_basic(code)
            elif language in ["cpp", "c", "c++"]:
                return self._extract_cpp_features(code)
            elif language == "javascript":
                return self._extract_javascript_features(code)
            elif language == "java":
                return self._extract_java_features(code)
            else:
                return self._extract_generic_features(code, language)
        except Exception as e:
            return {"error": f"AST extraction failed: {str(e)}", "features": {}}

    def _extract_python_features_basic(self, code: str) -> Dict[str, Any]:
        """Extract basic Python AST features without tree-sitter."""
        try:
            import ast

            tree = ast.parse(code)

            # Count different node types
            node_counts = {}
            def count_nodes(node):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
                for child in ast.iter_child_nodes(node):
                    count_nodes(child)
            count_nodes(tree)

            # Extract specific features
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

            features = {
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports),
                "total_ast_nodes": sum(node_counts.values()),
                "ast_depth": self._calculate_ast_depth(tree),
                "node_type_counts": node_counts
            }

            return {"features": features}
        except Exception as e:
            return {"error": f"Python AST extraction failed: {str(e)}", "features": {}}

    def _extract_cpp_features(self, code: str) -> Dict[str, Any]:
        """Extract C++ AST features using tree-sitter."""
        try:
            # Try tree-sitter first
            if TREE_SITTER_CPP_AVAILABLE:
                try:
                    import tree_sitter
                    import tree_sitter_cpp
                    parser = tree_sitter.Parser()
                    parser.language = tree_sitter.Language(tree_sitter_cpp.language())
                    tree = parser.parse(bytes(code, "utf8"))
                    return self._extract_tree_sitter_features(tree, code, "cpp")
                except ImportError:
                    pass

            # Fallback to basic regex-based analysis
            return self._extract_cpp_features_basic(code)
        except Exception as e:
            return {"error": f"C++ AST extraction failed: {str(e)}", "features": {}}

    def _extract_javascript_features(self, code: str) -> Dict[str, Any]:
        """Extract JavaScript AST features using tree-sitter."""
        try:
            # Try tree-sitter first
            if TREE_SITTER_JS_AVAILABLE:
                try:
                    import tree_sitter
                    import tree_sitter_javascript
                    parser = tree_sitter.Parser()
                    parser.language = tree_sitter.Language(tree_sitter_javascript.language())
                    tree = parser.parse(bytes(code, "utf8"))
                    return self._extract_tree_sitter_features(tree, code, "javascript")
                except ImportError:
                    pass

            # Fallback to basic analysis
            return self._extract_generic_features(code, "javascript")
        except Exception as e:
            return {"error": f"JavaScript AST extraction failed: {str(e)}", "features": {}}

    def _extract_java_features(self, code: str) -> Dict[str, Any]:
        """Extract Java AST features using tree-sitter."""
        try:
            # Try tree-sitter first
            if TREE_SITTER_JAVA_AVAILABLE:
                try:
                    import tree_sitter
                    import tree_sitter_java
                    parser = tree_sitter.Parser()
                    parser.language = tree_sitter.Language(tree_sitter_java.language())
                    tree = parser.parse(bytes(code, "utf8"))
                    return self._extract_tree_sitter_features(tree, code, "java")
                except ImportError:
                    pass

            # Fallback to basic analysis
            return self._extract_generic_features(code, "java")
        except Exception as e:
            return {"error": f"Java AST extraction failed: {str(e)}", "features": {}}

    def _extract_tree_sitter_features(self, tree, code: str, language: str) -> Dict[str, Any]:
        """Extract features using tree-sitter AST."""
        try:
            features = {
                "total_ast_nodes": self._count_nodes(tree.root_node),
                "ast_depth": self._calculate_depth(tree.root_node),
                "language": language
            }

            # Language-specific node counting
            if language == "cpp":
                features.update(self._count_cpp_nodes(tree))
            elif language == "javascript":
                features.update(self._count_js_nodes(tree))
            elif language == "java":
                features.update(self._count_java_nodes(tree))

            # Common features for all languages
            features.update(self._extract_common_features(code, language))

            return {"features": features}
        except Exception as e:
            return {"error": f"Tree-sitter feature extraction failed: {str(e)}", "features": {}}

    def _extract_cpp_features_basic(self, code: str) -> Dict[str, Any]:
        """Basic C++ feature extraction using regex."""
        import re

        features = {"language": "cpp"}

        # Count functions
        function_matches = re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*{', code)
        features["function_count"] = len(function_matches)

        # Count classes/structs
        class_matches = re.findall(r'\b(?:class|struct)\s+\w+', code)
        features["class_count"] = len(class_matches)

        # Count includes
        include_matches = re.findall(r'#include\s+[<"]\w+[\.>"]', code)
        features["include_count"] = len(include_matches)

        # Comment ratio
        lines = code.splitlines()
        comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
        features["comment_ratio"] = comment_lines / len(lines) if lines else 0

        # Variable declarations
        var_matches = re.findall(r'\b(?:int|float|double|char|string|bool)\s+\w+', code)
        features["variable_declarations"] = len(var_matches)

        return {"features": features}

    def _extract_common_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract features common to all languages."""
        lines = code.splitlines()
        features = {
            "total_lines": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
        }

        # Comment analysis based on language
        if language == "cpp":
            comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
        elif language == "javascript":
            comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
        elif language == "java":
            comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
        else:
            comment_lines = len([line for line in lines if any(comment in line for comment in ['//', '#', '/*', '*/'])])

        features["comment_lines"] = comment_lines
        return features

    def _count_nodes(self, node) -> int:
        """Count total AST nodes."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _calculate_depth(self, node) -> int:
        """Calculate AST depth."""
        if not node.children:
            return 0
        return 1 + max(self._calculate_depth(child) for child in node.children)

    def _count_nodes_by_type(self, node, counts):
        """Recursively count nodes by type."""
        node_type = node.type
        counts[node_type] = counts.get(node_type, 0) + 1
        for child in node.children:
            self._count_nodes_by_type(child, counts)
        return counts

    def _count_cpp_nodes(self, tree) -> Dict[str, Any]:
        """Count C++ specific AST nodes using tree-sitter."""
        counts = self._count_nodes_by_type(tree.root_node, {})

        # C++ specific constructs
        cpp_features = {
            "function_definitions": counts.get("function_definition", 0),
            "class_specs": counts.get("class_specifier", 0),
            "struct_specs": counts.get("struct_specifier", 0),
            "if_statements": counts.get("if_statement", 0),
            "for_loops": counts.get("for_statement", 0),
            "while_loops": counts.get("while_statement", 0),
            "template_declarations": counts.get("template_declaration", 0),
            "namespace_definitions": counts.get("namespace_definition", 0),
            "include_directives": counts.get("#include", 0),
            "using_directives": counts.get("using_declaration", 0),
            "typedefs": counts.get("type_definition", 0),
        }

        # Count variable declarations
        var_declarations = 0
        def count_var_decls(node):
            nonlocal var_declarations
            if node.type in ["declaration", "parameter_declaration", "field_declaration"]:
                var_declarations += 1
            for child in node.children:
                count_var_decls(child)
        count_var_decls(tree.root_node)
        cpp_features["variable_declarations"] = var_declarations

        return cpp_features

    def _count_js_nodes(self, tree) -> Dict[str, Any]:
        """Count JavaScript specific AST nodes using tree-sitter."""
        counts = self._count_nodes_by_type(tree.root_node, {})

        # JavaScript specific constructs
        js_features = {
            "function_declarations": counts.get("function_declaration", 0),
            "arrow_functions": counts.get("arrow_function", 0),
            "class_declarations": counts.get("class_declaration", 0),
            "if_statements": counts.get("if_statement", 0),
            "for_loops": counts.get("for_statement", 0),
            "while_loops": counts.get("while_statement", 0),
            "try_statements": counts.get("try_statement", 0),
            "catch_clauses": counts.get("catch_clause", 0),
            "import_statements": counts.get("import_statement", 0),
            "export_statements": counts.get("export_statement", 0),
        }

        # Count variable declarations by type
        var_counts = {"var": 0, "let": 0, "const": 0}
        def count_var_types(node):
            if node.type == "variable_declaration":
                # Check the kind of declaration
                for child in node.children:
                    if child.type in ["var", "let", "const"]:
                        var_counts[child.type] += 1
                        break
            for child in node.children:
                count_var_types(child)
        count_var_types(tree.root_node)

        js_features.update({
            "var_declarations": var_counts["var"],
            "let_declarations": var_counts["let"],
            "const_declarations": var_counts["const"],
        })

        return js_features

    def _count_java_nodes(self, tree) -> Dict[str, Any]:
        """Count Java specific AST nodes using tree-sitter."""
        counts = self._count_nodes_by_type(tree.root_node, {})

        # Java specific constructs
        java_features = {
            "method_declarations": counts.get("method_declaration", 0),
            "class_declarations": counts.get("class_declaration", 0),
            "interface_declarations": counts.get("interface_declaration", 0),
            "if_statements": counts.get("if_statement", 0),
            "for_loops": counts.get("for_statement", 0),
            "while_loops": counts.get("while_statement", 0),
            "try_statements": counts.get("try_statement", 0),
            "catch_clauses": counts.get("catch_clause", 0),
            "import_declarations": counts.get("import_declaration", 0),
            "package_declarations": counts.get("package_declaration", 0),
        }

        # Count modifiers
        modifier_counts = {"public": 0, "private": 0, "protected": 0, "static": 0, "final": 0}
        def count_modifiers(node):
            if node.type == "modifiers":
                for child in node.children:
                    if child.type in modifier_counts:
                        modifier_counts[child.type] += 1
            for child in node.children:
                count_modifiers(child)
        count_modifiers(tree.root_node)

        java_features.update(modifier_counts)

        # Count variable declarations
        var_declarations = 0
        def count_var_decls(node):
            nonlocal var_declarations
            if node.type in ["local_variable_declaration", "field_declaration"]:
                var_declarations += 1
            for child in node.children:
                count_var_decls(child)
        count_var_decls(tree.root_node)
        java_features["variable_declarations"] = var_declarations

        return java_features

    def _calculate_ast_depth(self, tree) -> int:
        """Calculate the maximum depth of the AST."""
        def get_depth(node, current_depth=0):
            if not hasattr(node, 'body') or not node.body:
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in node.body)
        return get_depth(tree)


# Placeholder implementations for remaining tools
# These will be implemented in subsequent steps

class StaticAnalyzerWrapperInput(BaseModel):
    """Input schema for StaticAnalyzerWrapper."""
    code: str = Field(description="Code text to analyze")
    language: str = Field(description="Programming language")


class StaticAnalyzerWrapper(BaseTool):
    """Wrapper for static analysis tools (linters, style checkers)."""

    name: str = "static_analyzer"
    description: str = "Run static analysis tools on code to extract quality metrics"
    args_schema: type[BaseModel] = StaticAnalyzerWrapperInput

    def _run(self, code: str, language: str) -> Dict[str, Any]:
        """Run static analysis on code."""
        try:
            results = {}

            if language == "python":
                results = self._analyze_python(code)
            elif language in ["cpp", "c", "c++"]:
                results = self._analyze_cpp(code)
            elif language == "javascript":
                results = self._analyze_javascript(code)
            elif language == "java":
                results = self._analyze_java(code)
            else:
                results = {"error": f"Static analysis not implemented for {language}"}

            return results

        except Exception as e:
            return {"error": f"Static analysis failed: {str(e)}"}

    def _analyze_python(self, code: str) -> Dict[str, Any]:
        """Run Python-specific static analysis."""
        try:
            import ast

            # Parse AST
            tree = ast.parse(code)

            # Basic metrics
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

            # Calculate basic complexity (simplified)
            total_nodes = len(list(ast.walk(tree)))
            avg_function_length = sum(len(func.body) for func in functions) / len(functions) if functions else 0

            # Comment ratio
            lines = code.splitlines()
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            comment_ratio = comment_lines / len(lines) if lines else 0

            return {
                "total_ast_nodes": total_nodes,
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports),
                "avg_function_length": avg_function_length,
                "comment_ratio": comment_ratio,
                "language": "python"
            }

        except Exception as e:
            return {"error": f"Python analysis failed: {str(e)}"}

    def _analyze_cpp(self, code: str) -> Dict[str, Any]:
        """Run C++ static analysis."""
        try:
            import re
            import subprocess
            import tempfile
            import os

            # Basic regex-based analysis
            features = {"language": "cpp"}

            # Count various C++ constructs using global patterns for consistency
            features["function_count"] = len(re.findall(CPP_FUNCTION_PATTERN, code))
            features["class_count"] = len(re.findall(CLASS_PATTERN, code))
            features["include_count"] = len(re.findall(r'#include\s+[<"]\w+[\.>"]', code))
            features["template_count"] = len(re.findall(CPP_TEMPLATE_PATTERN, code))
            features["namespace_count"] = len(re.findall(r'\bnamespace\s+\w+', code))

            # Comment analysis
            lines = code.splitlines()
            comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
            features["comment_ratio"] = comment_lines / len(lines) if lines else 0

            # Try to run cppcheck if available
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                    f.write(code)
                    temp_file = f.name

                try:
                    result = subprocess.run(['cppcheck', '--enable=all', '--xml', temp_file],
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        # Parse cppcheck output for issues
                        features["cppcheck_warnings"] = result.stdout.count('<error')
                        features["cppcheck_style_issues"] = result.stdout.count('style')
                    else:
                        features["cppcheck_warnings"] = 0
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    features["cppcheck_warnings"] = -1  # Not available
                finally:
                    os.unlink(temp_file)
            except Exception:
                features["cppcheck_warnings"] = -1

            return features

        except Exception as e:
            return {"error": f"C++ analysis failed: {str(e)}"}

    def _analyze_javascript(self, code: str) -> Dict[str, Any]:
        """Run JavaScript static analysis."""
        try:
            import re

            features = {"language": "javascript"}

            # Count JavaScript constructs using global patterns for consistency
            features["function_count"] = len(re.findall(JS_FUNCTION_PATTERN, code)) + len(re.findall(JS_ARROW_FUNCTION_PATTERN, code))
            features["class_count"] = len(re.findall(CLASS_PATTERN, code))
            features["import_count"] = len(re.findall(r'\bimport\s+.*\bfrom\b', code))
            features["export_count"] = len(re.findall(r'\bexport\s+', code))
            features["arrow_functions"] = len(re.findall(r'\([^)]*\)\s*=>', code))

            # Comment analysis
            lines = code.splitlines()
            comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
            features["comment_ratio"] = comment_lines / len(lines) if lines else 0

            # Variable declarations
            features["var_count"] = code.count('var ')
            features["let_count"] = code.count('let ')
            features["const_count"] = code.count('const ')

            return features

        except Exception as e:
            return {"error": f"JavaScript analysis failed: {str(e)}"}

    def _analyze_java(self, code: str) -> Dict[str, Any]:
        """Run Java static analysis."""
        try:
            import re

            features = {"language": "java"}

            # Count Java constructs using global patterns for consistency
            features["method_count"] = len(re.findall(JAVA_METHOD_PATTERN, code))
            features["class_count"] = len(re.findall(CLASS_PATTERN, code))
            features["interface_count"] = len(re.findall(r'\binterface\s+\w+', code))
            features["import_count"] = len(re.findall(r'\bimport\s+.*;', code))
            features["package_count"] = len(re.findall(r'\bpackage\s+.*;', code))

            # Comment analysis
            lines = code.splitlines()
            comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
            features["comment_ratio"] = comment_lines / len(lines) if lines else 0

            # Access modifiers
            features["public_count"] = code.count('public ')
            features["private_count"] = code.count('private ')
            features["protected_count"] = code.count('protected ')

            return features

        except Exception as e:
            return {"error": f"Java analysis failed: {str(e)}"}


class LMScorerInput(BaseModel):
    """Input schema for LMScorer."""
    code: str = Field(description="Code text to score")
    language: str = Field(description="Programming language")


class LMScorer(BaseTool):
    """Advanced language model scorer for computing perplexity-like scores using multiple evaluation methods."""

    name: str = "lm_scorer"
    description: str = "Compute comprehensive language model scores for code using NVIDIA LLM with multiple evaluation methods"
    args_schema: type[BaseModel] = LMScorerInput

    def _run(self, code: str, language: str) -> Dict[str, Any]:
        """Compute comprehensive LM-based scores for code."""
        try:
            # Get LLM instance
            llm = self._get_llm()
            if not llm:
                return {"error": "LLM not available", "lm_score": 0.5, "confidence": 0.0}

            # Compute multiple evaluation scores
            scores = self._compute_comprehensive_scores(code, language, llm)

            # Combine scores with weights
            final_score = self._combine_scores(scores)

            return {
                "lm_score": final_score,
                "normalized_score": min(max(final_score, 0), 1),  # Ensure 0-1 range
                "language": language,
                "component_scores": scores,
                "confidence": self._calculate_confidence(scores),
                "evaluation_methods": len(scores)
            }
        except Exception as e:
            return {
                "error": f"LM scoring failed: {str(e)}",
                "lm_score": 0.5,
                "confidence": 0.0
            }

    def _get_llm(self):
        """Get LLM instance following existing pattern."""
        try:
            from rag_pdf_server import load_nvidia_api_key
            api_key = load_nvidia_api_key()
            if api_key:
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                return ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=api_key)
        except Exception:
            pass
        return None

    def _compute_comprehensive_scores(self, code: str, language: str, llm) -> Dict[str, float]:
        """Compute multiple evaluation scores for comprehensive assessment."""
        scores = {}

        # Limit code length for API efficiency
        code_sample = code[:2000]  # Use larger sample for better evaluation

        # 1. Naturalness Score - How human-like does the code appear?
        scores["naturalness"] = self._evaluate_naturalness(code_sample, language, llm)

        # 2. Coherence Score - How logically consistent is the code structure?
        scores["coherence"] = self._evaluate_coherence(code_sample, language, llm)

        # 3. Typicality Score - How typical is this code for the language?
        scores["typicality"] = self._evaluate_typicality(code_sample, language, llm)

        # 4. Complexity-Appropriate Score - Does the complexity match the task?
        scores["complexity_fit"] = self._evaluate_complexity_fit(code_sample, language, llm)

        return scores

    def _evaluate_naturalness(self, code: str, language: str, llm) -> float:
        """Evaluate how natural and human-written the code appears."""
        try:
            prompt = f"""Evaluate how natural and human-written this {language} code looks.
Rate on a scale of 0-100, where:
- 0-20: Very unnatural, clearly AI-generated (robotic, overly verbose, unnatural patterns)
- 21-40: Unnatural, likely AI-generated (generic, lacks personality, formulaic)
- 41-60: Uncertain (could be either, no strong indicators)
- 61-80: Natural, likely human-written (shows thought process, reasonable style)
- 81-100: Very natural, definitely human-written (idiomatic, thoughtful, human-like)

Consider:
- Variable naming patterns (creative vs generic like 'temp', 'var1')
- Code structure and flow (logical progression vs mechanical)
- Comment style and quality (insightful vs generic)
- Overall readability and approach (intuitive vs algorithmic)

Code to evaluate:
```
{code}
```

Provide only a numerical score:"""

            response = llm.invoke(prompt).content
            score = self._extract_numerical_score(response)
            return score / 100.0  # Convert to 0-1 scale

        except Exception:
            return 0.5

    def _evaluate_coherence(self, code: str, language: str, llm) -> float:
        """Evaluate logical consistency and coherence of code structure."""
        try:
            prompt = f"""Evaluate the logical coherence and structure of this {language} code.
Rate on a scale of 0-100, where:
- 0-20: Very incoherent (random structure, no logical flow, disconnected parts)
- 21-40: Incoherent (poor organization, missing logical connections)
- 41-60: Moderately coherent (some logical flow but inconsistencies)
- 61-80: Coherent (good logical structure, reasonable organization)
- 81-100: Highly coherent (excellent structure, clear logical progression)

Consider:
- Function/method organization and purpose
- Variable usage patterns and scope
- Control flow logic and error handling
- Code modularity and separation of concerns

Code to evaluate:
```
{code}
```

Provide only a numerical score:"""

            response = llm.invoke(prompt).content
            score = self._extract_numerical_score(response)
            return score / 100.0

        except Exception:
            return 0.5

    def _evaluate_typicality(self, code: str, language: str, llm) -> float:
        """Evaluate how typical this code is for the given programming language."""
        try:
            language_patterns = self._get_language_patterns(language)

            prompt = f"""Evaluate how typical this code is for {language} programming.
Rate on a scale of 0-100, where:
- 0-20: Very atypical (uses wrong idioms, unusual patterns for the language)
- 21-40: Atypical (unusual approaches, not following language conventions)
- 41-60: Moderately typical (some standard patterns but unusual elements)
- 61-80: Typical (follows language conventions and idioms)
- 81-100: Highly typical (excellent use of language-specific patterns and idioms)

Language-specific considerations for {language}:
{language_patterns}

Code to evaluate:
```
{code}
```

Provide only a numerical score:"""

            response = llm.invoke(prompt).content
            score = self._extract_numerical_score(response)
            return score / 100.0

        except Exception:
            return 0.5

    def _evaluate_complexity_fit(self, code: str, language: str, llm) -> float:
        """Evaluate if the code complexity is appropriate for the apparent task."""
        try:
            # Estimate code complexity
            complexity_indicators = self._assess_code_complexity(code, language)

            prompt = f"""Evaluate if the complexity of this {language} code is appropriate for the apparent task.
Rate on a scale of 0-100, where:
- 0-20: Overly complex (unnecessarily complicated for the task, over-engineering)
- 21-40: Too complex (more complex than needed, could be simplified)
- 41-60: Moderately appropriate (reasonable complexity for the task)
- 61-80: Well-balanced (good complexity-to-task ratio)
- 81-100: Ideally balanced (perfect complexity match for the task)

Code complexity assessment:
- Lines of code: {complexity_indicators['lines']}
- Functions/methods: {complexity_indicators['functions']}
- Classes/structs: {complexity_indicators['classes']}
- Control structures: {complexity_indicators['control_structures']}
- Language features used: {complexity_indicators['language_features']}

Code to evaluate:
```
{code}
```

Provide only a numerical score:"""

            response = llm.invoke(prompt).content
            score = self._extract_numerical_score(response)
            return score / 100.0

        except Exception:
            return 0.5

    def _get_language_patterns(self, language: str) -> str:
        """Get language-specific evaluation patterns."""
        patterns = {
            "cpp": """
- Use of pointers, references, and memory management
- Template usage and generic programming
- Object-oriented patterns (inheritance, polymorphism)
- STL container usage and algorithms
- Exception handling with try/catch
- RAII (Resource Acquisition Is Initialization) patterns""",

            "javascript": """
- Use of modern ES6+ features (arrow functions, destructuring, async/await)
- Callback patterns vs Promises vs async/await
- DOM manipulation and event handling
- Module patterns (CommonJS, ES modules, IIFE)
- Functional programming elements
- Error handling with try/catch and Promises""",

            "java": """
- Object-oriented design patterns and principles
- Exception handling and checked exceptions
- Collection framework usage
- Stream API and functional programming
- Access modifiers and encapsulation
- Interface vs abstract class usage""",

            "python": """
- Use of list comprehensions and generator expressions
- Duck typing and dynamic features
- Context managers and decorators
- Pythonic idioms and conventions (EAFP, iterators)
- Standard library usage
- Exception handling patterns"""
        }

        return patterns.get(language.lower(), "- Standard programming practices\n- Language-appropriate patterns\n- Idiomatic code structure")

    def _assess_code_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Assess basic code complexity metrics."""
        lines = code.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('#')])

        # Count functions/methods (rough approximation)
        if language.lower() == 'cpp':
            functions = len(re.findall(CPP_FUNCTION_PATTERN, code))
        elif language.lower() == 'javascript':
            functions = len(re.findall(JS_FUNCTION_PATTERN, code)) + len(re.findall(JS_ARROW_FUNCTION_PATTERN, code))
        elif language.lower() == 'java':
            functions = len(re.findall(JAVA_METHOD_PATTERN, code))
        else:
            functions = len(re.findall(PYTHON_FUNCTION_PATTERN, code))  # Python default

        # Count classes/structs
        classes = len(re.findall(CLASS_PATTERN, code))

        # Count control structures
        control_structures = len(re.findall(CONTROL_STRUCTURE_PATTERN, code))

        # Assess language features (rough count)
        language_features = 0
        if language.lower() == 'cpp':
            language_features = len(re.findall(CPP_TEMPLATE_PATTERN, code)) + len(re.findall(CPP_STD_PATTERN, code))
        elif language.lower() == 'javascript':
            language_features = len(re.findall(JS_ASYNC_PATTERN, code)) + len(re.findall(JS_MODULE_PATTERN, code))
        elif language.lower() == 'java':
            language_features = len(re.findall(JAVA_STREAM_PATTERN, code)) + len(re.findall(JAVA_OVERRIDE_PATTERN, code))

        return {
            "lines": lines_of_code,
            "functions": functions,
            "classes": classes,
            "control_structures": control_structures,
            "language_features": language_features
        }

    def _extract_numerical_score(self, response: str) -> float:
        """Extract numerical score from LLM response."""
        import re
        # Look for numbers in the response
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            # Take the first number found
            score = float(numbers[0])
            # Ensure it's in valid range
            return max(0, min(100, score))
        return 50.0  # Default neutral score

    def _combine_scores(self, scores: Dict[str, float]) -> float:
        """Combine multiple evaluation scores into final LM score."""
        # Weights for different evaluation methods
        weights = {
            "naturalness": 0.40,      # Most important - human-like appearance
            "coherence": 0.30,        # Logical structure
            "typicality": 0.20,       # Language-appropriate patterns
            "complexity_fit": 0.10    # Appropriate complexity
        }

        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0

        for method, score in scores.items():
            if method in weights:
                weight = weights[method]
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        # Convert to AI-likelihood score (invert: high naturalness = low AI likelihood)
        naturalness_score = weighted_sum / total_weight
        ai_likelihood = 1.0 - naturalness_score

        return ai_likelihood

    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate confidence in the scoring based on score variance."""
        if len(scores) < 2:
            return 0.5

        # Calculate variance in scores
        score_values = list(scores.values())
        mean_score = sum(score_values) / len(score_values)
        variance = sum((score - mean_score) ** 2 for score in score_values) / len(score_values)

        # Lower variance = higher confidence
        # Scale variance to confidence (0-1 scale)
        confidence = max(0, min(1, 1.0 - variance * 4))  # Scale factor of 4 for reasonable range

        return confidence


class EmbeddingFAISSSearchInput(BaseModel):
    """Input schema for EmbeddingFAISSSearch."""
    code: str = Field(description="Code text to search for")
    language: str = Field(description="Programming language")
    search_type: str = Field(default="ai_corpus", description="Type of search: ai_corpus, student_history, or both")


class EmbeddingFAISSSearch(BaseTool):
    """FAISS-based embedding search for code similarity."""

    name: str = "embedding_faiss_search"
    description: str = "Search code embeddings for similarity to AI corpus and student history"
    args_schema: type[BaseModel] = EmbeddingFAISSSearchInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Don't set instance attributes that aren't fields
        self._embeddings = None
        self._ai_index = None
        self._student_indices = {}

    def _run(self, code: str, language: str, search_type: str = "ai_corpus") -> Dict[str, Any]:
        """Search for similar code in embeddings."""
        if not FAISS_AVAILABLE:
            return {"error": "FAISS not available", "similarity_score": 0.0}

        try:
            # Initialize embeddings if needed
            if not self._embeddings:
                self._embeddings = self._get_embeddings()

            if not self._embeddings:
                return {"error": "Embeddings not available", "similarity_score": 0.0}

            # Get embedding for input code
            query_embedding = self._embeddings.embed_query(code)

            results = {}

            if search_type in ["ai_corpus", "both"]:
                ai_similarity = self._search_ai_corpus(query_embedding)
                results["ai_corpus_similarity"] = ai_similarity

            if search_type in ["student_history", "both"]:
                student_similarity = self._search_student_history(query_embedding, "default_student")
                results["student_history_similarity"] = student_similarity

            # Overall similarity score (max of AI and student similarities)
            similarities = [v for v in results.values() if isinstance(v, (int, float))]
            overall_similarity = max(similarities) if similarities else 0.0

            return {
                "similarity_score": overall_similarity,
                "search_results": results,
                "search_type": search_type
            }

        except Exception as e:
            return {"error": f"Embedding search failed: {str(e)}", "similarity_score": 0.0}

    def _get_embeddings(self):
        """Get NVIDIA embeddings instance."""
        try:
            from rag_pdf_server import load_nvidia_api_key
            api_key = load_nvidia_api_key()
            if api_key:
                from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
                return NVIDIAEmbeddings(api_key=api_key)
        except Exception:
            pass
        return None

    def _search_ai_corpus(self, query_embedding: List[float]) -> float:
        """Search AI-generated code corpus."""
        # Placeholder: in real implementation, load pre-built FAISS index
        # For now, return random similarity
        if not self._ai_index:
            # Simulate loading index
            self._ai_index = "placeholder"

        # Placeholder similarity calculation
        # In real implementation: self.ai_index.search(query_embedding, k=5)
        return 0.3  # Placeholder

    def _search_student_history(self, query_embedding: List[float], student_id: str) -> float:
        """Search student's historical submissions."""
        # Placeholder implementation
        return 0.1  # Placeholder


class StylometryComparatorInput(BaseModel):
    """Input schema for StylometryComparator."""
    features: Dict[str, Any] = Field(description="Stylometric features from AST/static analysis")
    student_id: str = Field(description="Student identifier for baseline comparison")


class StylometryComparator(BaseTool):
    """Compare stylometric features to student baseline."""

    name: str = "stylometry_comparator"
    description: str = "Compare code style metrics to student's historical patterns"
    args_schema: type[BaseModel] = StylometryComparatorInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._student_baselines = {}  # In real implementation, load from database/file

    def _run(self, features: Dict[str, Any], student_id: str) -> Dict[str, Any]:
        """Compare features to student baseline."""
        try:
            baseline = self._get_student_baseline(student_id)

            if not baseline:
                # No baseline available, use general population statistics
                distance = self._calculate_population_distance(features)
                confidence = 0.5
            else:
                distance = self._calculate_distance(features, baseline)
                confidence = self._calculate_confidence(distance, baseline)

            return {
                "stylometry_distance": distance,
                "confidence": confidence,
                "baseline_available": baseline is not None,
                "comparison_metrics": self._get_comparison_metrics(features, baseline) if baseline else {}
            }

        except Exception as e:
            return {"error": f"Stylometry comparison failed: {str(e)}", "stylometry_distance": 0.5}

    def _get_student_baseline(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student's historical stylometric baseline."""
        # Placeholder: in real implementation, load from persistent storage
        return self._student_baselines.get(student_id)

    def _calculate_distance(self, features: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Calculate distance between current features and baseline."""
        distance = 0.0
        count = 0

        for key, value in features.items():
            if key in baseline and isinstance(value, (int, float)) and isinstance(baseline[key], (int, float)):
                # Normalized distance
                baseline_val = baseline[key]
                if baseline_val != 0:
                    diff = abs(value - baseline_val) / abs(baseline_val)
                    distance += diff
                    count += 1

        return distance / count if count > 0 else 0.5

    def _calculate_population_distance(self, features: Dict[str, Any]) -> float:
        """Calculate distance from general population statistics."""
        # Placeholder: use general AI-detection heuristics
        ai_indicators = {
            "comment_ratio": 0.3,  # AI often has high comment ratios
            "var_name_diversity": 0.1,  # AI uses fewer unique variable names
            "generic_func_ratio": 0.5,  # AI uses generic function names
        }

        distance = 0.0
        for key, ai_value in ai_indicators.items():
            if key in features:
                current = features[key]
                distance += abs(current - ai_value)

        return distance / len(ai_indicators)

    def _calculate_confidence(self, distance: float, baseline: Dict[str, Any]) -> float:
        """Calculate confidence in the distance measurement."""
        # Higher confidence with more baseline data
        sample_size = baseline.get("sample_count", 1)
        return min(sample_size / 10.0, 1.0)  # Max confidence at 10 samples

    def _get_comparison_metrics(self, features: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed comparison metrics."""
        comparisons = {}
        for key in features:
            if key in baseline:
                comparisons[key] = {
                    "current": features[key],
                    "baseline": baseline[key],
                    "difference": features[key] - baseline[key]
                }
        return comparisons


class SandboxTester(BaseTool):
    """Run code in sandbox for behavioral testing."""
    name: str = "sandbox_tester"
    description: str = "Execute code safely and analyze behavior"

    def _run(self, code: str) -> Dict[str, Any]:
        # Placeholder implementation
        return {"execution_results": "Not implemented yet"}


class ScoringAggregator(BaseTool):
    """Aggregate scores from multiple detectors."""
    name: str = "scoring_aggregator"
    description: str = "Combine detection scores into final assessment"

    def _run(self, scores: Dict[str, float]) -> Dict[str, Any]:
        # Weighted ensemble scoring
        weights = {
            "lm_score": 0.35,
            "faiss_similarity": 0.25,
            "stylometry_distance": 0.15,
            "provenance_score": 0.15,
            "dynamic_anomaly": 0.10
        }

        final_score = sum(scores.get(k, 0) * w for k, w in weights.items())

        return {
            "final_score": final_score,
            "threshold_high": 0.75,
            "threshold_review": 0.40,
            "recommendation": self._get_recommendation(final_score)
        }

    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on score."""
        if score >= 0.75:
            return "High confidence AI-generated - flag for action"
        elif score >= 0.40:
            return "Review required"
        else:
            return "Likely human-generated"


class ReportWriterInput(BaseModel):
    """Input schema for ReportWriter."""
    detection_results: Dict[str, Any] = Field(description="Results from all detection tools")
    student_id: str = Field(description="Student identifier")
    submission_id: str = Field(description="Submission identifier")


class ReportWriter(BaseTool):
    """Generate human-readable and JSON reports with explanations."""

    name: str = "report_writer"
    description: str = "Generate comprehensive detection reports with evidence and recommendations"
    args_schema: type[BaseModel] = ReportWriterInput

    def _run(self, detection_results: Dict[str, Any], student_id: str, submission_id: str) -> Dict[str, Any]:
        """Generate detection report."""
        try:
            # Extract key metrics
            final_score = detection_results.get("final_score", 0.5)
            threshold_high = detection_results.get("threshold_high", 0.75)
            threshold_review = detection_results.get("threshold_review", 0.40)

            # Determine recommendation
            if final_score >= threshold_high:
                recommendation = "FLAG_FOR_ACTION"
                confidence_level = "HIGH"
            elif final_score >= threshold_review:
                recommendation = "REVIEW_REQUIRED"
                confidence_level = "MEDIUM"
            else:
                recommendation = "LIKELY_HUMAN"
                confidence_level = "LOW"

            # Generate explanations
            explanations = self._generate_explanations(detection_results)

            # Create report
            report = {
                "report_metadata": {
                    "student_id": student_id,
                    "submission_id": submission_id,
                    "timestamp": datetime.now().isoformat(),
                    "detector_version": "1.0"
                },
                "scores": {
                    "final_score": final_score,
                    "confidence_level": confidence_level,
                    "thresholds": {
                        "high_confidence": threshold_high,
                        "review_required": threshold_review
                    }
                },
                "recommendation": recommendation,
                "evidence": explanations,
                "component_scores": self._extract_component_scores(detection_results)
            }

            # Generate human-readable summary
            human_readable = self._generate_human_readable(report)

            return {
                "json_report": report,
                "human_readable_report": human_readable,
                "recommendation": recommendation
            }

        except Exception as e:
            return {"error": f"Report generation failed: {str(e)}"}

    def _generate_explanations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate natural language explanations for the score."""
        explanations = []

        # LM Score explanation
        lm_score = results.get("lm_score", 0.5)
        if lm_score > 0.7:
            explanations.append({
                "component": "Language Model Analysis",
                "evidence": "Code appears highly unnatural based on language model evaluation",
                "score_contribution": lm_score * 0.35,
                "severity": "HIGH"
            })
        elif lm_score < 0.3:
            explanations.append({
                "component": "Language Model Analysis",
                "evidence": "Code appears natural and human-like",
                "score_contribution": lm_score * 0.35,
                "severity": "LOW"
            })

        # Similarity explanation
        similarity = results.get("similarity_score", 0.0)
        if similarity > 0.5:
            explanations.append({
                "component": "Embedding Similarity",
                "evidence": f"High similarity ({similarity:.2f}) to known AI-generated code",
                "score_contribution": similarity * 0.25,
                "severity": "HIGH"
            })

        # Stylometry explanation
        stylometry_dist = results.get("stylometry_distance", 0.5)
        if stylometry_dist > 0.7:
            explanations.append({
                "component": "Stylometric Analysis",
                "evidence": "Code style significantly differs from student's baseline",
                "score_contribution": stylometry_dist * 0.15,
                "severity": "MEDIUM"
            })

        return explanations

    def _extract_component_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract individual component scores."""
        return {
            "lm_score": results.get("lm_score", 0.5),
            "similarity_score": results.get("similarity_score", 0.0),
            "stylometry_distance": results.get("stylometry_distance", 0.5),
            "provenance_score": results.get("provenance_score", 0.0),
            "dynamic_anomaly": results.get("dynamic_anomaly", 0.0)
        }

    def _generate_human_readable(self, report: Dict[str, Any]) -> str:
        """Generate human-readable report."""
        scores = report["scores"]
        recommendation = report["recommendation"]
        evidence = report["evidence"]

        report_text = f"""
AI-GENERATED CODE DETECTION REPORT

Student ID: {report['report_metadata']['student_id']}
Submission ID: {report['report_metadata']['submission_id']}
Generated: {report['report_metadata']['timestamp']}

FINAL SCORE: {scores['final_score']:.3f}
CONFIDENCE LEVEL: {scores['confidence_level']}
RECOMMENDATION: {recommendation}

THRESHOLDS:
- High Confidence (Flag): >= {scores['thresholds']['high_confidence']}
- Review Required: >= {scores['thresholds']['review_required']}

EVIDENCE SUMMARY:
"""

        for item in evidence:
            report_text += f"- {item['component']}: {item['evidence']} (Contribution: {item['score_contribution']:.3f})\n"

        report_text += "\nCOMPONENT SCORES:\n"
        for comp, score in report["component_scores"].items():
            report_text += f"- {comp}: {score:.3f}\n"

        return report_text