import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_train_subset_epoch_smoke():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ml" / "train.py"),
            "--model",
            "cnn_small",
            "--epochs",
            "1",
            "--subset",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
