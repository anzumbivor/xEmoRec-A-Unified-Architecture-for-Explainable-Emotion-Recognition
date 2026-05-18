"""Attention, Integrated Gradients, and hybrid token-attribution utilities.

This module reflects the latest XAI notebook version supplied by the user
(`Adaptive_XAI_modified_V2_Fig_Regenerated.ipynb`) and is used by the adaptive
faithfulness pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import PreTrainedTokenizerBase

from emotion_xai.model import ImprovedDeepEmotionModel, forward_from_embeddings


@dataclass(frozen=True)
class ExplanationRecord:
    """Serializable container for one explained text instance."""

    text: str
    true_label: int
    pred_label: int
    pred_conf: float
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens_full: list[str]
    visible_tokens: list[str]
    deletable_indices: np.ndarray
    deletable_mask: np.ndarray
    num_deletable: int
    attention_raw_scores: np.ndarray
    ig_raw_scores: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation with safe fallbacks for short/constant vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    corr, _ = spearmanr(a, b)
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def entropy_normalized(x: np.ndarray, eps: float = 1e-12) -> float:
    """Return entropy normalized to [0, 1] for a non-negative score vector."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, a_min=0.0, a_max=None)
    total = x.sum()
    if total <= eps:
        return 1.0
    p = x / total
    h = -(p * np.log(p + eps)).sum()
    hmax = np.log(len(p) + eps)
    return 1.0 if hmax <= eps else float(h / hmax)


def normalize_with_mask(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    mode: str = "sum",
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize attribution scores only across valid/deletable token positions.

    Supported modes mirror the latest notebook: `sum`, `minmax`, `rank`, and
    `softmax`. Invalid positions are set to zero.
    """
    scores = np.asarray(scores, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    out = np.zeros_like(scores, dtype=np.float64)
    if valid_mask.sum() == 0:
        return out

    vals = np.abs(scores[valid_mask].astype(np.float64))
    if mode == "sum":
        vals = np.clip(vals, a_min=0.0, a_max=None)
    elif mode == "minmax":
        vmin = vals.min()
        vmax = vals.max()
        vals = np.ones_like(vals) if (vmax - vmin) <= eps else (vals - vmin) / (vmax - vmin)
    elif mode == "rank":
        order = np.argsort(vals)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float64)
        vals = ranks
    elif mode == "softmax":
        vals = np.exp(vals - vals.max())
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    vals = np.clip(vals, a_min=0.0, a_max=None)
    total = vals.sum()
    vals = np.ones_like(vals) / len(vals) if total <= eps else vals / total
    out[valid_mask] = vals
    return out


def adaptive_alpha_from_scores(
    attn_scores: np.ndarray,
    ig_scores: np.ndarray,
    alpha_min: float = 0.10,
    alpha_max: float = 0.60,
) -> tuple[float, float, float]:
    """Compute the adaptive fusion weight from Attention/IG agreement.

    The latest notebook combines two signals: Attention–IG rank agreement and
    attention concentration. The resulting alpha is clipped to the configured
    range.
    """
    corr = safe_spearman(attn_scores, ig_scores)
    corr01 = 0.5 * (corr + 1.0)
    attn_entropy = entropy_normalized(attn_scores)
    attn_concentration = 1.0 - attn_entropy
    raw = 0.60 * corr01 + 0.40 * attn_concentration
    alpha = alpha_min + (alpha_max - alpha_min) * raw
    return float(np.clip(alpha, alpha_min, alpha_max)), corr, attn_entropy


def build_deletable_mask(
    input_ids_1d: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """Identify real non-special tokens that can be removed during perturbation."""
    mask = attention_mask_1d.bool().clone()
    special_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    for special_id in special_ids:
        if special_id is not None:
            mask &= input_ids_1d != special_id
    return mask


def prepare_model_for_ig(model: ImprovedDeepEmotionModel) -> list[tuple[torch.nn.Module, bool]]:
    """Temporarily enable gradients through trainable layers while disabling dropout."""
    old_modes = [
        (model, model.training),
        (model.bilstm, model.bilstm.training),
        (model.dropout, model.dropout.training),
        (model.attention, model.attention.training),
        (model.layernorm, model.layernorm.training),
        (model.fc, model.fc.training),
    ]
    model.train()
    model.bilstm.train()
    model.attention.train()
    model.fc.train()
    model.layernorm.train()
    model.dropout.eval()
    return old_modes


def restore_modes(old_modes: list[tuple[torch.nn.Module, bool]]) -> None:
    """Restore PyTorch train/eval states after Integrated Gradients."""
    for module, mode in old_modes:
        module.train(mode)


def compute_attention_token_scores(
    model_out: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Map CLS-query attention over pooled positions back to original tokens."""
    attn_weights = model_out["attn_weights"]
    pooled_mask = model_out["pooled_mask"][0]
    emb_mask = model_out["emb_mask"][0]

    attn = attn_weights[0]
    if attn.dim() == 2:
        pooled_scores = attn.squeeze(0)
    elif attn.dim() == 3:
        pooled_scores = attn.mean(dim=0).squeeze(0)
    else:
        raise ValueError(f"Unexpected attention weight shape: {attn.shape}")

    pooled_scores = pooled_scores * pooled_mask.float()
    pooled_scores = pooled_scores / (pooled_scores.sum() + 1e-8)

    token_scores = torch.zeros(emb_mask.size(0), device=device)
    for pooled_idx in range(pooled_scores.size(0)):
        if not pooled_mask[pooled_idx]:
            continue
        t1 = 2 * pooled_idx
        t2 = 2 * pooled_idx + 1
        if t1 < token_scores.size(0):
            token_scores[t1] += pooled_scores[pooled_idx]
        if t2 < token_scores.size(0):
            token_scores[t2] += pooled_scores[pooled_idx]
    return token_scores.detach().cpu().numpy()


def compute_ig_token_scores(
    model: ImprovedDeepEmotionModel,
    embeddings: torch.Tensor,
    emb_mask: torch.Tensor,
    pred_class: int,
    device: torch.device,
    m_steps: int = 16,
) -> np.ndarray:
    """Compute absolute-sum Integrated Gradients scores at the embedding layer."""
    baseline = torch.zeros_like(embeddings).to(device)
    scaled_embs = []
    for step in range(1, m_steps + 1):
        alpha = float(step) / m_steps
        scaled_embs.append(baseline + alpha * (embeddings - baseline))

    scaled_embs = torch.cat(scaled_embs, dim=0)
    scaled_masks = emb_mask.unsqueeze(0).repeat(m_steps, 1).to(device)
    scaled_embs.requires_grad_(True)
    grads_accum = torch.zeros_like(scaled_embs)

    old_modes = prepare_model_for_ig(model)
    try:
        # Avoid CuDNN RNN backward edge cases in IG loops.
        with torch.backends.cudnn.flags(enabled=False):
            for i in range(m_steps):
                emb_i = scaled_embs[i : i + 1]
                mask_i = scaled_masks[i : i + 1]
                logits_i = forward_from_embeddings(model, emb_i, mask_i)
                logit_c = logits_i[0, pred_class]
                grads = torch.autograd.grad(
                    logit_c,
                    emb_i,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                grads_accum[i] = grads[0]
    finally:
        restore_modes(old_modes)
        model.eval()

    avg_grads = grads_accum.mean(dim=0)
    ig = (embeddings[0] - baseline[0]) * avg_grads
    return ig.abs().sum(dim=-1).detach().cpu().numpy()


def explain_instance(
    model: ImprovedDeepEmotionModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    true_label: int,
    device: torch.device,
    m_steps: int = 16,
    max_length: int = 64,
) -> ExplanationRecord:
    """Generate attention and IG token scores for one text instance."""
    enc = tokenizer(
        str(text),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids, attention_mask, return_attn=True, return_embeddings=True)

    probs = F.softmax(out["logits"], dim=-1)[0]
    pred_class = int(probs.argmax().item())
    pred_conf = float(probs[pred_class].item())

    input_ids_1d = input_ids[0].detach().cpu()
    attention_mask_1d = attention_mask[0].detach().cpu()
    deletable_mask_t = build_deletable_mask(input_ids_1d, attention_mask_1d, tokenizer)
    deletable_mask = deletable_mask_t.numpy().astype(bool)

    attention_scores = compute_attention_token_scores(out, device=device)
    ig_scores = compute_ig_token_scores(
        model=model,
        embeddings=out["embeddings"],
        emb_mask=out["emb_mask"][0],
        pred_class=pred_class,
        device=device,
        m_steps=m_steps,
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids_1d.tolist())
    deletable_indices = np.where(deletable_mask)[0]
    visible_tokens = [tokens[i] for i in deletable_indices]

    return ExplanationRecord(
        text=str(text),
        true_label=int(true_label),
        pred_label=pred_class,
        pred_conf=pred_conf,
        input_ids=input_ids_1d.clone(),
        attention_mask=attention_mask_1d.clone(),
        tokens_full=tokens,
        visible_tokens=visible_tokens,
        deletable_indices=deletable_indices,
        deletable_mask=deletable_mask,
        num_deletable=int(len(deletable_indices)),
        attention_raw_scores=attention_scores,
        ig_raw_scores=ig_scores,
    )


def build_score_bundle(
    record: ExplanationRecord,
    norm: str = "sum",
    alpha_fixed: float = 0.85,
    alpha_min: float = 0.65,
    alpha_max: float = 0.95,
) -> dict[str, np.ndarray | float | str]:
    """Build normalized attention, IG, fixed-hybrid, and adaptive-hybrid scores."""
    mask = record.deletable_mask
    attn = normalize_with_mask(record.attention_raw_scores, mask, mode=norm)
    ig = normalize_with_mask(record.ig_raw_scores, mask, mode=norm)

    alpha_adaptive, corr, attn_entropy = adaptive_alpha_from_scores(
        attn[mask], ig[mask], alpha_min=alpha_min, alpha_max=alpha_max
    )

    fixed = normalize_with_mask(alpha_fixed * attn + (1.0 - alpha_fixed) * ig, mask, mode="sum")
    adaptive = normalize_with_mask(
        alpha_adaptive * attn + (1.0 - alpha_adaptive) * ig,
        mask,
        mode="sum",
    )

    return {
        "attention_scores": attn,
        "ig_scores": ig,
        "hybrid_fixed_scores": fixed,
        "hybrid_adaptive_scores": adaptive,
        "alpha_fixed": alpha_fixed,
        "alpha_adaptive": alpha_adaptive,
        "spearman_attn_ig": corr,
        "attn_entropy": attn_entropy,
        "norm": norm,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
    }
