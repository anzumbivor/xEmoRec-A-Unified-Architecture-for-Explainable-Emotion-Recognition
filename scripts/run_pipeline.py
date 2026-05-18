"""Run training followed by the latest adaptive-XAI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_xai.faithfulness import AdaptiveXAIConfig, run_adaptive_xai
from emotion_xai.train import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model and run adaptive XAI in one command.")
    parser.add_argument("--data-path", default="text.csv")
    parser.add_argument("--out-dir", default="runs_emotion_cls_2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-pool-size", type=int, default=600)
    parser.add_argument("--m-steps-screen", type=int, default=16)
    parser.add_argument("--m-steps-deep", type=int, default=32)
    args = parser.parse_args()

    train_cfg = TrainingConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    train_result = run_training(train_cfg)

    xai_out = Path(args.out_dir) / "xai_faithfulness_v6_pubready_figures"
    xai_cfg = AdaptiveXAIConfig(
        data_path=args.data_path,
        model_path=str(train_result["best_model_path"]),
        out_dir=str(xai_out),
        max_length=args.max_length,
        global_seed=args.seed,
        split_random_state=args.seed,
        eval_pool_size=args.eval_pool_size,
        m_steps_screen=args.m_steps_screen,
        m_steps_deep=args.m_steps_deep,
    )
    run_adaptive_xai(xai_cfg)


if __name__ == "__main__":
    main()
