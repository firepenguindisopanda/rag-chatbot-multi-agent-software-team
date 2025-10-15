import json
import yaml
from typing import Dict, Any

def load_rubric(path: str) -> Dict[str,Any]:
    """Load rubric JSON or YAML describing criteria and weights."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)

def validate_rubric(rubric: Dict[str,Any]) -> bool:
    """Basic sanity checks for rubric structure."""
    if not isinstance(rubric, dict):
        return False

    # Find criteria that have weights (actual grading criteria vs metadata)
    criteria_weights = []
    for key, value in rubric.items():
        # Skip metadata fields that aren't grading criteria
        if key in ['total_points', 'partial_credit_policy', 'grading_notes']:
            continue
        # Skip non-dict values (like integers, strings)
        if not isinstance(value, dict):
            continue
        # Check if this looks like a grading criterion
        if 'weight' in value or 'subcriteria' in value:
            weight = value.get('weight', 0)
            if isinstance(weight, (int, float)):
                criteria_weights.append(weight)

    # Require at least one criterion with weight
    return len(criteria_weights) > 0 and sum(criteria_weights) > 0