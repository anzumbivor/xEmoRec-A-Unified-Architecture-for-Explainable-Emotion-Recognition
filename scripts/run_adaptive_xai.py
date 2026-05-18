"""Thin CLI wrapper for the latest adaptive-XAI faithfulness analysis.

Run from the repository root after training:
    python scripts/run_adaptive_xai.py \
        --data-path data/text.csv \
        --model-path runs_emotion_cls_2/best_model_cls.pt \
        --out-dir runs_emotion_cls_2/xai_faithfulness_v6_pubready_figures
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_xai.faithfulness import main


if __name__ == "__main__":
    main()
