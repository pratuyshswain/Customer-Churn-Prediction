import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_import_has_no_output(tmp_path):
    result = subprocess.run([sys.executable, "-c", "import churn_prediction"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout == ""

def test_pipeline_generates_charts(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "churn_prediction.py")], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=120)
    assert "Accuracy:" in result.stdout
    for name in ("confusion_matrix.png", "feature_importance.png"):
        assert (tmp_path / name).stat().st_size > 0
