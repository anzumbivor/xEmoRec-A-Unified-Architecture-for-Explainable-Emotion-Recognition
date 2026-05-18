"""Plot helpers for training and XAI outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_training_curves(history: dict[str, list[float]], out_dir: str | Path) -> None:
    """Save loss, accuracy, and macro-F1 curves from the training history."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plots = [
        ("train_loss", "val_loss", "Loss", "loss_curve_cls.png"),
        ("train_acc", "val_acc", "Accuracy", "accuracy_curve_cls.png"),
        ("train_f1", "val_f1", "Macro-F1", "macro_f1_curve_cls.png"),
    ]
    for train_key, val_key, ylabel, filename in plots:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history[train_key], label="Train")
        plt.plot(epochs, history[val_key], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / filename)
        plt.close()


def save_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    out_path: str | Path,
    class_names: Iterable[str | int] | None = None,
) -> None:
    """Save a confusion matrix without requiring seaborn."""
    cm = confusion_matrix(list(y_true), list(y_pred))
    labels = list(class_names) if class_names is not None else list(range(cm.shape[0]))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
