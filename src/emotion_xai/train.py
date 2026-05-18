"""Training entry point for the RoBERTa-CNN-BiLSTM-CLS-attention model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from emotion_xai.data import TextDataset, load_text_label_csv, stratified_text_split
from emotion_xai.model import ImprovedDeepEmotionModel, count_parameters
from emotion_xai.plotting import save_confusion_matrix, save_training_curves
from emotion_xai.reproducibility import set_seed
from emotion_xai.xai import build_score_bundle, explain_instance


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    data_path: str = "text.csv"
    out_dir: str = "runs_emotion_cls_2"
    roberta_name: str = "roberta-base"
    max_length: int = 64
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    dropout: float = 0.3
    test_size: float = 0.10
    seed: int = 42
    num_workers: int = 0
    deterministic: bool = False
    example_count: int = 10
    ig_steps_for_examples: int = 16


def train_one_epoch(
    model: ImprovedDeepEmotionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Run one training epoch and return loss, accuracy, macro-F1, and time."""
    model.train()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    start_time = time.time()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, epoch_time


@torch.no_grad()
def evaluate(
    model: ImprovedDeepEmotionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
    """Evaluate the model and return aggregate metrics plus predictions."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    start_time = time.time()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)["logits"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, np.asarray(all_labels), np.asarray(all_preds), epoch_time


def save_example_explanations(
    model: ImprovedDeepEmotionModel,
    tokenizer: RobertaTokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    out_path: Path,
    max_length: int,
    example_count: int,
    ig_steps: int,
) -> None:
    """Save a small human-inspection CSV with attention, IG, and hybrid scores."""
    rows = []
    for text, label in list(zip(texts, labels))[:example_count]:
        rec = explain_instance(
            model=model,
            tokenizer=tokenizer,
            text=text,
            true_label=label,
            device=device,
            m_steps=ig_steps,
            max_length=max_length,
        )
        bundle = build_score_bundle(rec, norm="sum", alpha_fixed=0.5, alpha_min=0.65, alpha_max=0.95)
        rows.append(
            {
                "text": rec.text,
                "true_label": rec.true_label,
                "pred_label": rec.pred_label,
                "pred_conf": rec.pred_conf,
                "tokens": " ".join(rec.tokens_full),
                "attention_scores": " ".join(f"{v:.6f}" for v in bundle["attention_scores"]),
                "ig_scores": " ".join(f"{v:.6f}" for v in bundle["ig_scores"]),
                "hybrid_scores_alpha_0_5": " ".join(f"{v:.6f}" for v in bundle["hybrid_fixed_scores"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_training(config: TrainingConfig) -> dict[str, float | str]:
    """Train the model and write all reproducibility artifacts to disk."""
    set_seed(config.seed, deterministic=config.deterministic)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    df = load_text_label_csv(config.data_path)
    split = stratified_text_split(df, test_size=config.test_size, random_state=config.seed)
    num_classes = len(sorted(df["label"].unique()))

    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_name)
    train_dataset = TextDataset(split.train_texts, split.train_labels, tokenizer, config.max_length)
    val_dataset = TextDataset(split.eval_texts, split.eval_labels, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedDeepEmotionModel(
        num_classes=num_classes,
        dropout=config.dropout,
        roberta_name=config.roberta_name,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f">>> Device: {device}")
    print(f">>> Total parameters: {total_params:,}; trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_time": [],
        "val_time": [],
    }
    best_val_f1 = -1.0
    best_path = out_dir / "best_model_cls.pt"
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, y_true, y_pred, val_time = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)
        history["train_time"].append(tr_time)
        history["val_time"].append(val_time)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train loss={tr_loss:.4f}, acc={tr_acc:.4f}, f1={tr_f1:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f">>> Saved new best checkpoint: {best_path} (val_f1={val_f1:.4f})")

    # Re-evaluate the best checkpoint so saved metrics correspond to the best model.
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state, strict=False)
    val_loss, val_acc, val_f1, y_true, y_pred, _ = evaluate(model, val_loader, criterion, device)

    pd.DataFrame(history).to_csv(out_dir / "history_cls.csv", index=False)
    save_training_curves(history, out_dir)
    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix_val_cls.png", sorted(df["label"].unique()))

    pd.DataFrame(
        {
            "text": split.eval_texts,
            "true_label": y_true,
            "pred_label": y_pred,
        }
    ).to_csv(out_dir / "validation_predictions.csv", index=False)

    with open(out_dir / "classification_report_val.json", "w", encoding="utf-8") as f:
        json.dump(classification_report(y_true, y_pred, output_dict=True), f, indent=2)

    save_example_explanations(
        model=model,
        tokenizer=tokenizer,
        texts=split.eval_texts,
        labels=split.eval_labels,
        device=device,
        out_path=out_dir / "example_explanations_cls_attention_ig_hybrid.csv",
        max_length=config.max_length,
        example_count=config.example_count,
        ig_steps=config.ig_steps_for_examples,
    )

    elapsed_min = (time.time() - start) / 60.0
    result = {
        "best_model_path": str(best_path),
        "best_val_f1_during_training": float(best_val_f1),
        "best_checkpoint_val_loss": float(val_loss),
        "best_checkpoint_val_acc": float(val_acc),
        "best_checkpoint_val_f1": float(val_f1),
        "total_minutes": float(elapsed_min),
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f">>> Training complete. Outputs saved to: {out_dir}")
    return result


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the emotion-recognition model.")
    parser.add_argument("--data-path", default="text.csv")
    parser.add_argument("--out-dir", default="runs_emotion_cls_2")
    parser.add_argument("--roberta-name", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--example-count", type=int, default=10)
    parser.add_argument("--ig-steps-for-examples", type=int, default=16)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
