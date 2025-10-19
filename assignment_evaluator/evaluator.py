from pathlib import Path
from typing import Dict, Any, List
import zipfile
import tempfile
import json
import os

# Remove the direct import - we'll pass LLM as parameter instead
# Import existing LLM and vector store from the main application
def get_llm():
    """Get LLM instance, with fallback for testing."""
    try:
        from rag_pdf_server import llm
        return llm
    except (ImportError, AttributeError):
        # Fallback for testing
        return None

def unzip_submission(zip_path: str, dest_dir: str) -> List[str]:
    """Unzip submission and return list of file paths (code files only)."""
    # Security: Check file size (max 50MB)
    if os.path.getsize(zip_path) > 50 * 1024 * 1024:
        raise ValueError("ZIP file too large (max 50MB)")

    with zipfile.ZipFile(zip_path, "r") as z:
        # Security: Check for dangerous file names
        for name in z.namelist():
            if ".." in name or name.startswith("/"):
                raise ValueError(f"Potentially dangerous file path: {name}")

        z.extractall(dest_dir)

    files = []
    for p in Path(dest_dir).rglob("*"):
        if p.is_file():
            # Security: Check individual file sizes (max 10MB each)
            if p.stat().st_size > 10 * 1024 * 1024:
                continue  # Skip large files

            # Only include code files
            if p.suffix in {".py", ".java", ".js", ".cpp", ".c", ".ts"}:
                files.append(str(p))

    return files

def prepare_code_chunks(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Read files and produce chunks with metadata for embedding/indexing."""
    chunks = []
    for fp in file_paths:
        text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        # simple chunking; replace with smart chunking preserving function boundaries
        lines = text.splitlines()
        chunk_size = 200
        for i in range(0, len(lines), chunk_size):
            chunk_text = "\n".join(lines[i:i+chunk_size])
            chunks.append({"path": fp, "text": chunk_text, "start_line": i+1})
    return chunks

def grade_submission(zip_path: str, rubric: Dict[str, Any], llm=None, master_code_path: str = None) -> Dict[str, Any]:
    """High-level grading: index code, run rubric checks via RAG/LLM and static heuristics."""
    # Get LLM if not provided
    if llm is None:
        llm = get_llm()

    with tempfile.TemporaryDirectory() as tmp:
        files = unzip_submission(zip_path, tmp)

        # Run static checks
        static_results = run_static_checks(files)

        # Extract grading criteria from complex rubric structure
        grading_criteria = extract_grading_criteria(rubric)

        # Initialize master code context if provided
        master_context = None
        if master_code_path and os.path.exists(master_code_path):
            master_context = load_and_index_master_code(master_code_path)
            # Add relevant chunks to context for use in prompts
            if master_context:
                master_context['relevant_chunks'] = get_relevant_master_chunks(master_context)

        # Evaluate each grading criterion
        results = {}
        for crit_key, crit in grading_criteria.items():
            results[crit_key] = evaluate_criterion(crit, files, static_results.get(crit_key), llm, master_context)

        # AI-detection for whole submission
        ai_flags = detect_ai_generated(files, llm)

        # Build per-file metrics so callers (and Google Sheets export) can persist one row per file
        from pathlib import Path as _Path
        import re as _re

        def _extract_student_id(filename: str) -> str:
            """Extract leading 9-digit student id from filename, or empty string."""
            m = _re.match(r"^(\d{9})", filename)
            return m.group(1) if m else ""

        file_metrics: Dict[str, Dict[str, Any]] = {}
        # For efficiency, if no LLM is available, avoid per-criterion LLM calls and use aggregated result
        for fp in files:
            name = _Path(fp).name
            metrics: Dict[str, Any] = {}
            metrics['student_id'] = _extract_student_id(name)
            metrics['filename'] = name

            # Stylometric features + ai confidence for this file
            try:
                code_text = _Path(fp).read_text(encoding='utf-8', errors='ignore')
                features = calculate_stylometric_features(code_text)
                metrics['ai_confidence'] = assess_ai_confidence(features)
            except Exception:
                metrics['ai_confidence'] = 0.0

            # Per-criterion scoring for this file (LLM when available)
            per_scores = {}
            for crit_key, crit in grading_criteria.items():
                try:
                    if llm is not None:
                        # evaluate criterion with only this file as context
                        eval_res = evaluate_criterion(crit, [fp], None, llm, master_context)
                        score = eval_res.get('llm_result', {}).get('score', None)
                    else:
                        # Fallback: use aggregated result (if present) or default
                        agg = results.get(crit_key, {})
                        score = agg.get('llm_result', {}).get('score') if agg else None
                except Exception:
                    score = None
                per_scores[f'criterion_{crit_key}'] = score

            # Average per-file score (mean of available criterion scores)
            score_vals = [v for v in per_scores.values() if isinstance(v, (int, float))]
            if score_vals:
                metrics['avg_score'] = sum(score_vals) / len(score_vals)
            else:
                metrics['avg_score'] = None

            # Merge per-criterion scores into metrics
            metrics.update(per_scores)
            file_metrics[name] = metrics

        # Also include list of basenames to support the existing Sheets helper which expects a list
        files_list = [_Path(fp).name for fp in files]

        return {
            "results": results,
            "ai_flags": ai_flags,
            "static_summary": static_results,
            "files": files_list,
            "file_metrics": file_metrics,
        }


def extract_grading_criteria(rubric: Dict[str, Any]) -> Dict[str, Any]:
    """Extract actual grading criteria from complex rubric structures."""
    criteria = {}

    for key, value in rubric.items():
        if is_grading_criterion(key, value):
            criteria[key] = create_criterion_structure(key, value)

    return criteria


def is_grading_criterion(key: str, value: Any) -> bool:
    """Check if a rubric entry is an actual grading criterion vs metadata."""
    # Skip metadata fields
    if key in ['total_points', 'partial_credit_policy', 'grading_notes']:
        return False

    # Skip non-dict values
    if not isinstance(value, dict):
        return False

    # Check if this looks like a grading criterion
    return 'weight' in value or 'subcriteria' in value


def create_criterion_structure(key: str, value: Dict[str, Any]) -> Dict[str, Any]:
    """Create a simplified criterion structure for LLM evaluation."""
    criterion = {
        'description': value.get('description', f'Criterion: {key}'),
        'weight': value.get('weight', 0),
        'original_key': key
    }

    # If it has subcriteria, include them in the description
    if 'subcriteria' in value and isinstance(value['subcriteria'], dict):
        subcriterion_text = build_subcriteria_description(value['subcriteria'])
        if subcriterion_text:
            criterion['description'] += "\n\nSubcriteria:\n" + subcriterion_text

    return criterion


def build_subcriteria_description(subcriteria: Dict[str, Any]) -> str:
    """Build a formatted description of subcriteria."""
    descriptions = []
    for sub_key, sub_value in subcriteria.items():
        if isinstance(sub_value, dict) and 'description' in sub_value:
            points = sub_value.get('points', 'N/A')
            descriptions.append(f"- {sub_value['description']} ({points} points)")

    return "\n".join(descriptions)


def evaluate_criterion(criterion: Dict[str, Any], files: List[str], static_result: Any = None, llm=None, master_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluate a single rubric criterion using LLM or fallback."""
    if llm is not None:
        llm_result = evaluate_with_llm(criterion, files, llm, master_context)
    else:
        llm_result = {"score": 50, "explanation": "LLM not available - using default score"}

    return {
        "rubric": criterion,
        "llm_result": llm_result,
        "static": static_result
    }


def evaluate_with_llm(criterion: Dict[str, Any], files: List[str], llm, master_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Use LLM to evaluate a rubric criterion."""
    prompt = build_evaluation_prompt(criterion, files, master_context)

    try:
        response = llm.invoke(prompt).content
        return parse_llm_response(response)
    except Exception as e:
        return {"score": 0, "explanation": f"LLM evaluation failed: {str(e)}"}


def build_evaluation_prompt(criterion: Dict[str, Any], files: List[str], master_context: Dict[str, Any] = None) -> str:
    """Build the LLM prompt for rubric evaluation."""
    code_samples = []
    for f in files[:3]:  # Limit to first 3 files
        try:
            content = Path(f).read_text(encoding='utf-8', errors='ignore')[:500]
            code_samples.append(f"- {Path(f).name}: {content}...")
        except Exception:
            continue

    base_prompt = f"""Evaluate the following code submission for the criterion: {criterion.get('description', 'General quality')}

Please analyze the code and provide:
1. A score from 0-100 (where 100 is excellent)
2. A brief explanation of your evaluation

Code files in submission:
{chr(10).join(code_samples)}"""

    # Add master code context if available
    if master_context and master_context.get('relevant_chunks'):
        master_code_section = f"""

For context, here are relevant excerpts from the instructor's model solution that may be helpful for comparison:

{master_context['relevant_chunks']}

When evaluating, consider how the student's approach compares to the model solution, but remember that different valid approaches should still receive appropriate credit."""

        base_prompt += master_code_section

    return base_prompt + "\n\nScore and explanation:"


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse score and explanation from LLM response."""
    lines = response.split('\n')
    score = 50  # default
    explanation = response

    # Try to extract score from response
    for line in lines:
        if 'score' in line.lower() or line.strip().isdigit():
            try:
                score = int(''.join(filter(str.isdigit, line)))
                score = max(0, min(100, score))
                break
            except ValueError:
                pass

    return {"score": score, "explanation": explanation}

def run_static_checks(file_paths: List[str]) -> Dict[str, Any]:
    """Run simple static heuristics (placeholder). Return mapping criterion -> findings."""
    findings = {}
    # Examples: missing docstrings, very uniform variable names, suspicious large comment blocks
    for fp in file_paths:
        text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        if "TODO" in text:
            findings.setdefault("notes", []).append({"file": fp, "note": "Contains TODOs"})
    return findings

def assess_ai_confidence(features: Dict[str, float]) -> float:
    """Assess AI confidence based on stylometric features."""
    confidence = 0.0

    # High comment ratio often indicates AI-generated code
    if features["comment_ratio"] > 0.3:
        confidence += 0.3

    # Very consistent line lengths may indicate AI
    if features["avg_line_length"] > 80:
        confidence += 0.2

    # Low variable name diversity may indicate AI
    if features["var_name_diversity"] < 0.1:
        confidence += 0.2

    # High ratio of generic function names
    if features["generic_func_ratio"] > 0.5:
        confidence += 0.2

    # High string literal ratio
    if features["string_literal_ratio"] > 0.1:
        confidence += 0.1

    return min(confidence, 1.0)


def detect_ai_generated(file_paths: List[str], llm=None) -> Dict[str, Any]:
    """Combine heuristics + LLM classifier to flag likely AI-generated code."""
    flags = {"suspected_files": [], "reasons": [], "confidence_scores": {}}

    for fp in file_paths:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            # Calculate stylometric features
            features = calculate_stylometric_features(text)
            heuristic_confidence = assess_ai_confidence(features)

            # Use LLM for additional analysis if available
            llm_confidence = 0.0
            llm_reasoning = ""
            if llm is not None:
                llm_confidence, llm_reasoning = analyze_with_llm_for_ai_detection(text, llm)

            # Combine heuristic and LLM confidence (weighted average)
            combined_confidence = (heuristic_confidence * 0.7) + (llm_confidence * 0.3)

            if combined_confidence > 0.7:  # High confidence threshold
                flags["suspected_files"].append(fp)
                flags["reasons"].append({
                    "file": fp,
                    "reason": f"AI-generated code detected (confidence: {combined_confidence:.2f})",
                    "features": features,
                    "heuristic_confidence": heuristic_confidence,
                    "llm_confidence": llm_confidence,
                    "llm_reasoning": llm_reasoning
                })

            flags["confidence_scores"][fp] = combined_confidence

        except Exception:
            # Skip files that can't be read
            continue

    return flags


def analyze_with_llm_for_ai_detection(code: str, llm) -> tuple[float, str]:
    """Use LLM to analyze code for AI generation patterns."""
    try:
        prompt = f"""Analyze this code snippet and determine if it appears to be AI-generated. Consider:

1. Code structure and patterns
2. Comment quality and style
3. Variable naming conventions
4. Code organization and flow
5. Signs of template or boilerplate usage

Rate your confidence that this code is AI-generated on a scale of 0-100, where:
- 0-20: Definitely human-written
- 21-40: Likely human-written
- 41-60: Uncertain
- 61-80: Likely AI-generated
- 81-100: Definitely AI-generated

Code to analyze:
```
{code[:2000]}  # Limit code length
```

Provide your confidence score and brief reasoning:"""

        response = llm.invoke(prompt).content

        # Extract confidence score from response
        confidence = 50  # default
        lines = response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['confidence', 'score', 'rating']):
                try:
                    # Extract numbers from the line
                    numbers = [int(s) for s in line.split() if s.isdigit()]
                    if numbers:
                        confidence = numbers[0]
                        break
                except ValueError:
                    continue

        confidence = max(0, min(100, confidence)) / 100.0  # Convert to 0-1 scale
        return confidence, response

    except Exception as e:
        return 0.0, f"LLM analysis failed: {str(e)}"


def calculate_stylometric_features(code: str) -> Dict[str, float]:
    """Calculate stylometric features that may indicate AI-generated code."""
    features = {}

    # Comment to code ratio
    comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
    total_lines = len(code.split('\n'))
    features["comment_ratio"] = comment_lines / max(total_lines, 1)

    # Average line length
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    features["avg_line_length"] = sum(len(line) for line in lines) / max(len(lines), 1)

    # Variable name entropy (diversity)
    import re
    var_names = re.findall(r'\b\w+\b', code)
    unique_vars = len(set(var_names))
    features["var_name_diversity"] = unique_vars / max(len(var_names), 1)

    # Function name patterns (AI often uses generic names)
    func_names = re.findall(r'def\s+(\w+)', code)
    generic_funcs = sum(1 for name in func_names if any(word in name.lower() for word in
                        ['function', 'method', 'process', 'handle', 'calculate', 'compute']))
    features["generic_func_ratio"] = generic_funcs / max(len(func_names), 1)

    # String literal patterns
    strings = re.findall(r'["\']([^"\']*)["\']', code)
    features["string_literal_ratio"] = len(strings) / max(len(code.split()), 1)

    return features


def load_and_index_master_code(master_code_path: str) -> Dict[str, Any]:
    """Load and index master code for contextual evaluation."""
    try:
        # Read the master code file
        with open(master_code_path, 'r', encoding='utf-8', errors='ignore') as f:
            master_code = f.read()

        # Split into logical chunks (functions, classes, etc.)
        chunks = split_code_into_chunks(master_code)

        return {
            'full_code': master_code,
            'chunks': chunks,
            'filename': Path(master_code_path).name,
            'language': detect_code_language(master_code_path)
        }
    except Exception as e:
        print(f"Warning: Could not load master code from {master_code_path}: {e}")
        return None


def split_code_into_chunks(code: str) -> List[Dict[str, str]]:
    """Split code into meaningful chunks for context."""
    chunks = []

    # Split by functions/classes (basic approach)
    lines = code.split('\n')
    current_chunk = []
    current_type = "general"
    current_name = "main"

    for i, line in enumerate(lines):
        line = line.strip()

        # Detect function definitions
        if line.startswith('def ') and ':' in line:
            # Save previous chunk if it exists
            if current_chunk:
                chunks.append({
                    'type': current_type,
                    'name': current_name,
                    'content': '\n'.join(current_chunk),
                    'line_start': max(1, i - len(current_chunk)),
                    'line_end': i
                })

            # Start new function chunk
            func_name = line.split('def ')[1].split('(')[0].strip()
            current_chunk = [line]
            current_type = "function"
            current_name = func_name

        # Detect class definitions
        elif line.startswith('class ') and ':' in line:
            # Save previous chunk if it exists
            if current_chunk:
                chunks.append({
                    'type': current_type,
                    'name': current_name,
                    'content': '\n'.join(current_chunk),
                    'line_start': max(1, i - len(current_chunk)),
                    'line_end': i
                })

            # Start new class chunk
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            current_chunk = [line]
            current_type = "class"
            current_name = class_name

        else:
            # Add line to current chunk
            current_chunk.append(line)

    # Add final chunk
    if current_chunk:
        chunks.append({
            'type': current_type,
            'name': current_name,
            'content': '\n'.join(current_chunk),
            'line_start': max(1, len(lines) - len(current_chunk)),
            'line_end': len(lines)
        })

    return chunks


def detect_code_language(filepath: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(filepath).suffix.lower()
    language_map = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust'
    }
    return language_map.get(ext, 'unknown')


def get_relevant_master_chunks(master_context: Dict[str, Any], max_chunks: int = 2) -> str:
    """Get relevant chunks from master code based on student submission."""
    if not master_context or not master_context.get('chunks'):
        return ""

    # Simple relevance heuristic: include key functions/classes
    relevant_chunks = []

    # Prioritize functions and classes over general code
    prioritized_chunks = [chunk for chunk in master_context['chunks']
                         if chunk['type'] in ['function', 'class']]

    # Take the most substantial chunks
    prioritized_chunks.sort(key=lambda x: len(x['content']), reverse=True)

    for chunk in prioritized_chunks[:max_chunks]:
        chunk_text = f"**{chunk['type'].title()}: {chunk['name']}**\n```python\n{chunk['content']}\n```"
        relevant_chunks.append(chunk_text)

    if relevant_chunks:
        return "\n\n".join(relevant_chunks)
    else:
        # Fallback: include first substantial chunk
        substantial_chunks = [chunk for chunk in master_context['chunks']
                            if len(chunk['content']) > 50]
        if substantial_chunks:
            chunk = substantial_chunks[0]
            return f"**Reference Implementation:**\n```python\n{chunk['content'][:1000]}\n```"

    return ""