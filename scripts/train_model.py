"""Thin CLI wrapper for model training.

Run from the repository root:
    python scripts/train_model.py --data-path data/text.csv --out-dir runs_emotion_cls_2
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_xai.train import main


if __name__ == "__main__":
    main()
