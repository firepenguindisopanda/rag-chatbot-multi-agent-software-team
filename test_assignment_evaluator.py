import tempfile, zipfile, json
from assignment_evaluator.evaluator import grade_submission

def make_sample_zip(tmp_zip_path: str):
    with zipfile.ZipFile(tmp_zip_path, "w") as z:
        z.writestr("solution.py", "def add(a,b):\n    return a + b\n")
    return tmp_zip_path

def test_grade_stub():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
        p = tf.name
    make_sample_zip(p)
    rubric = {"correctness": {"description": "Correctness of solution", "weight": 1.0}}
    report = grade_submission(p, rubric)
    assert "results" in report