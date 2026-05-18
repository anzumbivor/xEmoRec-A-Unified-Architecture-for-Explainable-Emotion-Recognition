"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


REQUIRED_COLUMNS = {"text", "label"}


@dataclass(frozen=True)
class DatasetSplit:
    """Container for a stratified train/validation or train/test split."""

    train_texts: list[str]
    eval_texts: list[str]
    train_labels: list[int]
    eval_labels: list[int]


def load_text_label_csv(data_path: str | Path) -> pd.DataFrame:
    """Load a CSV file and validate the expected `text` and `label` columns."""
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {sorted(REQUIRED_COLUMNS)}; missing {sorted(missing)}")

    out = df.copy()
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def stratified_text_split(
    df: pd.DataFrame,
    test_size: float = 0.10,
    random_state: int = 42,
) -> DatasetSplit:
    """Create the same stratified split pattern used in the original notebooks."""
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        df["text"].astype(str).tolist(),
        df["label"].astype(int).tolist(),
        test_size=test_size,
        stratify=df["label"].astype(int).tolist(),
        random_state=random_state,
    )
    return DatasetSplit(
        train_texts=list(train_texts),
        eval_texts=list(eval_texts),
        train_labels=list(train_labels),
        eval_labels=list(eval_labels),
    )


def stratified_subsample_indices(labels: Iterable[int], max_n: int | None, seed: int) -> pd.Index | list[int]:
    """Return a stratified subset of row indices for expensive XAI evaluation."""
    import numpy as np

    y = np.asarray(list(labels))
    idx = np.arange(len(y))
    if max_n is None or max_n >= len(y):
        return idx
    chosen, _ = train_test_split(idx, train_size=max_n, stratify=y, random_state=seed)
    return np.sort(chosen)


class TextDataset(Dataset):
    """PyTorch dataset that tokenizes one text example at a time.

    This intentionally mirrors the notebook logic so the saved model is
    reproducible from the original CSV with only `text` and `label` columns.
    """

    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ) -> None:
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }
