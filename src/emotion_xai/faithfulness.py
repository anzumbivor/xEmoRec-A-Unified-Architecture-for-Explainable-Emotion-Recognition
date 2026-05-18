"""Adaptive XAI faithfulness pipeline.

This script modularizes the latest notebook:
`Adaptive_XAI_modified_V2_Fig_Regenerated.ipynb`.

It performs two phases:
1. Screening: evaluates attention, IG, random, fixed hybrids, and adaptive hybrids.
2. Deep evaluation: reruns the final shortlist with more IG steps and exports tables,
   statistical tests, token-level outputs, and publication-ready figures.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.integrate import trapezoid
from scipy.stats import friedmanchisquare, wilcoxon
from transformers import RobertaTokenizer

from emotion_xai.data import load_text_label_csv, stratified_subsample_indices, stratified_text_split
from emotion_xai.model import ImprovedDeepEmotionModel
from emotion_xai.reproducibility import set_seed
from emotion_xai.xai import ExplanationRecord, build_score_bundle, explain_instance


@dataclass
class AdaptiveXAIConfig:
    """Configuration for the adaptive-XAI faithfulness experiment."""

    model_path: str = "runs_emotion_cls_2/best_model_cls.pt"
    data_path: str = "text.csv"
    out_dir: str = "runs_emotion_cls_2/xai_faithfulness_v6_pubready_figures"
    roberta_name: str = "roberta-base"
    test_size: float = 0.10
    split_random_state: int = 42
    global_seed: int = 42
    max_length: int = 64
    m_steps_screen: int = 16
    m_steps_deep: int = 32
    eval_pool_size: int | None = 600
    n_runs: int = 7
    run_sample_size: int = 400
    fractions: list[float] = field(default_factory=lambda: np.linspace(0.1, 1.0, 10).tolist())
    topk_frac_comp: float = 0.20
    topk_frac_sweep: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30])
    random_restarts: int = 5
    fixed_alpha_sweep: list[float] = field(default_factory=lambda: [0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
    normalization_sweep: list[str] = field(default_factory=lambda: ["sum", "minmax", "rank", "softmax"])
    adaptive_alpha_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.10, 0.60), (0.65, 0.95), (0.70, 0.95), (0.75, 0.95)]
    )
    primary_baselines: list[str] = field(default_factory=lambda: ["attention", "ig", "random"])
    bootstrap_b: int = 2000
    bootstrap_seed: int = 123
    dpi: int = 180


def method_tag(method_type: str, **kwargs: float | str) -> str:
    """Create a unique, parseable method identifier."""
    if method_type in {"attention", "ig", "random"}:
        return method_type
    if method_type == "hybrid_fixed":
        return f"hybrid_fixed__norm={kwargs['norm']}__alpha={float(kwargs['alpha']):.2f}"
    if method_type == "hybrid_adaptive":
        return (
            f"hybrid_adaptive__norm={kwargs['norm']}"
            f"__amin={float(kwargs['alpha_min']):.2f}__amax={float(kwargs['alpha_max']):.2f}"
        )
    raise ValueError(f"Unknown method_type: {method_type}")


def parse_method_tag(tag: str) -> dict[str, float | str]:
    """Parse method identifiers created by `method_tag`."""
    if tag in {"attention", "ig", "random"}:
        return {"method_type": tag}
    parts = tag.split("__")
    out: dict[str, float | str] = {"method_type": parts[0]}
    for part in parts[1:]:
        key, value = part.split("=")
        out[key] = value if key == "norm" else float(value)
    return out


def make_candidate_specs(config: AdaptiveXAIConfig) -> list[dict[str, float | str]]:
    """Build all candidate methods used during the screening stage."""
    specs: list[dict[str, float | str]] = []
    for base in config.primary_baselines:
        specs.append({"name": base, "method_type": base})

    for norm in config.normalization_sweep:
        for alpha in config.fixed_alpha_sweep:
            specs.append(
                {
                    "name": method_tag("hybrid_fixed", norm=norm, alpha=alpha),
                    "method_type": "hybrid_fixed",
                    "norm": norm,
                    "alpha": alpha,
                }
            )
        for alpha_min, alpha_max in config.adaptive_alpha_ranges:
            specs.append(
                {
                    "name": method_tag(
                        "hybrid_adaptive",
                        norm=norm,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                    ),
                    "method_type": "hybrid_adaptive",
                    "norm": norm,
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                }
            )
    return specs


def bootstrap_mean_ci(x: Iterable[float], n_boot: int = 2000, seed: int = 123, ci: int = 95) -> tuple[float, float]:
    """Bootstrap a confidence interval for the mean."""
    values = np.asarray(list(x), dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    low = np.percentile(means, (100 - ci) / 2)
    high = np.percentile(means, 100 - (100 - ci) / 2)
    return float(low), float(high)


def holm_bonferroni(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    """Apply Holm-Bonferroni correction within one family of tests."""
    out = df.copy().sort_values(p_col).reset_index(drop=True)
    m = len(out)
    adjusted = []
    running_max = 0.0
    for i, p in enumerate(out[p_col].tolist()):
        adjusted_p = (m - i) * p
        running_max = max(running_max, adjusted_p)
        adjusted.append(min(running_max, 1.0))
    out["p_holm"] = adjusted
    return out


def load_model_and_tokenizer(config: AdaptiveXAIConfig, num_classes: int, device: torch.device) -> tuple[ImprovedDeepEmotionModel, RobertaTokenizer]:
    """Load tokenizer and trained checkpoint for XAI evaluation."""
    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_name)
    model = ImprovedDeepEmotionModel(num_classes=num_classes, dropout=0.3, roberta_name=config.roberta_name).to(device)
    state = torch.load(config.model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(">>> Warning: checkpoint did not load perfectly.")
        print(">>> Missing keys:", missing)
        print(">>> Unexpected keys:", unexpected)
    model.eval()
    return model, tokenizer


def predict_prob_for_class(
    model: ImprovedDeepEmotionModel,
    input_ids_1d: torch.Tensor,
    class_idx: int,
    pad_id: int,
    device: torch.device,
) -> float:
    """Predict P(class_idx) after token masking/removal."""
    if input_ids_1d.dim() == 1:
        input_ids = input_ids_1d.unsqueeze(0).to(device)
    else:
        input_ids = input_ids_1d.to(device)
    attention_mask = (input_ids != pad_id).long()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)["logits"]
        probs = F.softmax(logits, dim=-1)
    return float(probs[0, class_idx].item())


def get_scores_for_method(record: ExplanationRecord, method_name: str) -> tuple[np.ndarray | None, dict]:
    """Return token scores and score bundle for a named explanation method."""
    parsed = parse_method_tag(method_name)
    method_type = parsed["method_type"]

    if method_type == "attention":
        bundle = build_score_bundle(record, norm="sum", alpha_fixed=0.85, alpha_min=0.65, alpha_max=0.95)
        return bundle["attention_scores"], bundle
    if method_type == "ig":
        bundle = build_score_bundle(record, norm="sum", alpha_fixed=0.85, alpha_min=0.65, alpha_max=0.95)
        return bundle["ig_scores"], bundle
    if method_type == "hybrid_fixed":
        bundle = build_score_bundle(
            record,
            norm=str(parsed["norm"]),
            alpha_fixed=float(parsed["alpha"]),
            alpha_min=0.65,
            alpha_max=0.95,
        )
        return bundle["hybrid_fixed_scores"], bundle
    if method_type == "hybrid_adaptive":
        bundle = build_score_bundle(
            record,
            norm=str(parsed["norm"]),
            alpha_fixed=0.85,
            alpha_min=float(parsed["amin"]),
            alpha_max=float(parsed["amax"]),
        )
        return bundle["hybrid_adaptive_scores"], bundle
    if method_type == "random":
        return None, {}
    raise ValueError(f"Unknown method: {method_name}")


def rank_positions(record: ExplanationRecord, method_name: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Rank deletable token positions from most to least important."""
    deletable = record.deletable_indices
    if method_name == "random":
        rng = rng or np.random.default_rng()
        order = deletable.copy()
        rng.shuffle(order)
        return order
    scores, _ = get_scores_for_method(record, method_name)
    if scores is None:
        raise ValueError("Non-random method returned no scores.")
    return deletable[np.argsort(-scores[deletable])]


def make_removed_input(input_ids_1d: torch.Tensor, remove_positions: Iterable[int], pad_id: int) -> torch.Tensor:
    """Replace selected token positions with the tokenizer pad token."""
    x = input_ids_1d.clone()
    positions = [int(p) for p in remove_positions]
    if positions:
        x[positions] = pad_id
    return x


def make_kept_only_input(
    input_ids_1d: torch.Tensor,
    deletable_positions: Iterable[int],
    keep_positions: Iterable[int],
    pad_id: int,
) -> torch.Tensor:
    """Mask every deletable token except the selected kept positions."""
    x = input_ids_1d.clone()
    keep = {int(p) for p in keep_positions}
    for p in deletable_positions:
        if int(p) not in keep:
            x[int(p)] = pad_id
    return x


def compute_metrics_for_record(
    record: ExplanationRecord,
    model: ImprovedDeepEmotionModel,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    fractions: list[float],
    topk_frac: float = 0.2,
    random_restarts: int = 5,
    method_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[float, dict[str, dict[str, float | np.ndarray]]]:
    """Compute deletion AUC, comprehensiveness, and sufficiency for one record."""
    method_names = method_names or ["attention", "ig", "random"]
    pad_id = tokenizer.pad_token_id
    input_ids_1d = record.input_ids.clone()
    pred_class = int(record.pred_label)
    deletable = record.deletable_indices
    n_del = len(deletable)
    base_prob = predict_prob_for_class(model, input_ids_1d, pred_class, pad_id, device)

    out: dict[str, dict[str, float | np.ndarray]] = {}
    x_axis = np.concatenate([[0.0], np.asarray(fractions, dtype=np.float64)])

    for method in method_names:
        if n_del == 0:
            curve = np.full(len(x_axis), base_prob, dtype=np.float64)
            out[method] = {"curve": curve, "auc": float(trapezoid(curve, x=x_axis)), "comprehensiveness": 0.0, "sufficiency": 0.0}
            continue

        if method == "random":
            curves, comps, suffs = [], [], []
            for restart in range(random_restarts):
                rng = np.random.default_rng(seed + restart + int(record.true_label) * 1000 + len(record.text))
                order = rank_positions(record, "random", rng=rng)
                curve = [base_prob]
                for frac in fractions:
                    k = max(1, int(math.ceil(frac * n_del)))
                    masked_ids = make_removed_input(input_ids_1d, order[:k], pad_id)
                    curve.append(predict_prob_for_class(model, masked_ids, pred_class, pad_id, device))

                k_top = max(1, int(math.ceil(topk_frac * n_del)))
                top_pos = order[:k_top]
                removed_ids = make_removed_input(input_ids_1d, top_pos, pad_id)
                kept_ids = make_kept_only_input(input_ids_1d, deletable, top_pos, pad_id)
                comps.append(base_prob - predict_prob_for_class(model, removed_ids, pred_class, pad_id, device))
                suffs.append(base_prob - predict_prob_for_class(model, kept_ids, pred_class, pad_id, device))
                curves.append(curve)
            curve_arr = np.mean(np.asarray(curves, dtype=np.float64), axis=0)
            comp = float(np.mean(comps))
            suff = float(np.mean(suffs))
        else:
            order = rank_positions(record, method_name=method)
            curve = [base_prob]
            for frac in fractions:
                k = max(1, int(math.ceil(frac * n_del)))
                masked_ids = make_removed_input(input_ids_1d, order[:k], pad_id)
                curve.append(predict_prob_for_class(model, masked_ids, pred_class, pad_id, device))

            k_top = max(1, int(math.ceil(topk_frac * n_del)))
            top_pos = order[:k_top]
            removed_ids = make_removed_input(input_ids_1d, top_pos, pad_id)
            kept_ids = make_kept_only_input(input_ids_1d, deletable, top_pos, pad_id)
            comp = base_prob - predict_prob_for_class(model, removed_ids, pred_class, pad_id, device)
            suff = base_prob - predict_prob_for_class(model, kept_ids, pred_class, pad_id, device)
            curve_arr = np.asarray(curve, dtype=np.float64)

        out[method] = {
            "curve": curve_arr,
            "auc": float(trapezoid(curve_arr, x=x_axis)),
            "comprehensiveness": float(comp),
            "sufficiency": float(suff),
        }
    return base_prob, out


def summarize_metrics_df(
    metrics_df: pd.DataFrame,
    method_names: list[str],
    bootstrap_b: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Summarize per-instance metrics for each method."""
    rows = []
    for method in method_names:
        auc_vals = metrics_df[f"{method}_auc"].values
        comp_vals = metrics_df[f"{method}_comp"].values
        suff_vals = metrics_df[f"{method}_suff"].values
        ci_low, ci_high = bootstrap_mean_ci(auc_vals, n_boot=bootstrap_b, seed=bootstrap_seed)
        rows.append(
            {
                "method": method,
                "auc_mean": float(np.mean(auc_vals)),
                "auc_std": float(np.std(auc_vals, ddof=1)),
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "comp_mean": float(np.mean(comp_vals)),
                "suff_mean": float(np.mean(suff_vals)),
            }
        )
    return pd.DataFrame(rows).sort_values(["auc_mean", "comp_mean"], ascending=[True, False]).reset_index(drop=True)


def repeated_stratified_auc_runs(
    metrics_df: pd.DataFrame,
    labels_arr: np.ndarray,
    method_names: list[str],
    n_runs: int,
    run_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat stratified subsampling to test ranking stability."""
    rows = []
    for run_seed in range(n_runs):
        idx = stratified_subsample_indices(labels_arr, run_sample_size, seed=1000 + run_seed)
        sub = metrics_df.iloc[idx]
        for method in method_names:
            rows.append({"run_seed": 1000 + run_seed, "method": method, "mean_auc": float(sub[f"{method}_auc"].mean())})
    runs_df = pd.DataFrame(rows)
    agg = (
        runs_df.groupby("method", as_index=False)
        .agg(repeated_mean_auc=("mean_auc", "mean"), repeated_std_auc=("mean_auc", "std"))
        .sort_values("repeated_mean_auc")
        .reset_index(drop=True)
    )
    return runs_df, agg


def pairwise_tests(metrics_df: pd.DataFrame, method_names: list[str]) -> pd.DataFrame:
    """Run Wilcoxon signed-rank tests with Holm correction for each metric."""
    pairs = []
    for metric_key, suffix in [("auc", "auc"), ("comprehensiveness", "comp"), ("sufficiency", "suff")]:
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                a, b = method_names[i], method_names[j]
                xa = metrics_df[f"{a}_{suffix}"].values
                xb = metrics_df[f"{b}_{suffix}"].values
                try:
                    stat, p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    stat, p = np.nan, 1.0
                pairs.append(
                    {
                        "metric": metric_key,
                        "method_a": a,
                        "method_b": b,
                        "mean_diff_a_minus_b": float(np.mean(xa - xb)),
                        "statistic": float(stat) if not np.isnan(stat) else np.nan,
                        "p_value": float(p),
                    }
                )
    pairwise_df = pd.DataFrame(pairs)
    return pd.concat([holm_bonferroni(pairwise_df[pairwise_df["metric"] == m]) for m in pairwise_df["metric"].unique()]).reset_index(drop=True)


def build_explanation_cache(
    test_df: pd.DataFrame,
    model: ImprovedDeepEmotionModel,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    max_length: int,
    m_steps: int,
    label: str,
) -> list[ExplanationRecord]:
    """Compute attention and IG explanations for every selected test instance."""
    records: list[ExplanationRecord] = []
    for i, row in test_df.iterrows():
        if i % 25 == 0:
            print(f"    {label}: processed {i}/{len(test_df)}")
        records.append(
            explain_instance(
                model=model,
                tokenizer=tokenizer,
                text=row["text"],
                true_label=int(row["label"]),
                device=device,
                m_steps=m_steps,
                max_length=max_length,
            )
        )
    return records


def evaluate_methods(
    records: list[ExplanationRecord],
    model: ImprovedDeepEmotionModel,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    method_names: list[str],
    config: AdaptiveXAIConfig,
    include_text: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Evaluate faithfulness metrics for a method list over cached records."""
    metric_rows = []
    curve_store: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in records:
        _, per_method = compute_metrics_for_record(
            record=rec,
            model=model,
            tokenizer=tokenizer,
            device=device,
            fractions=config.fractions,
            topk_frac=config.topk_frac_comp,
            random_restarts=config.random_restarts,
            method_names=method_names,
            seed=config.global_seed,
        )
        default_bundle = build_score_bundle(rec, norm="sum", alpha_fixed=0.85, alpha_min=0.65, alpha_max=0.95)
        row = {
            "true_label": rec.true_label,
            "pred_label": rec.pred_label,
            "pred_conf": rec.pred_conf,
            "num_deletable": rec.num_deletable,
            "spearman_attn_ig": default_bundle["spearman_attn_ig"],
            "alpha_adaptive_default": default_bundle["alpha_adaptive"],
        }
        if include_text:
            row["text"] = rec.text
        for method_name, metrics in per_method.items():
            row[f"{method_name}_auc"] = metrics["auc"]
            row[f"{method_name}_comp"] = metrics["comprehensiveness"]
            row[f"{method_name}_suff"] = metrics["sufficiency"]
            curve_store[method_name].append(metrics["curve"])
        metric_rows.append(row)
    return pd.DataFrame(metric_rows), curve_store


def run_topk_sweep(
    records: list[ExplanationRecord],
    model: ImprovedDeepEmotionModel,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    final_methods: list[str],
    config: AdaptiveXAIConfig,
) -> pd.DataFrame:
    """Evaluate comprehensiveness/sufficiency sensitivity to top-k fraction."""
    rows = []
    for topk_frac in config.topk_frac_sweep:
        temp_rows = []
        for rec in records:
            _, per_method = compute_metrics_for_record(
                rec,
                model=model,
                tokenizer=tokenizer,
                device=device,
                fractions=config.fractions,
                topk_frac=topk_frac,
                random_restarts=config.random_restarts,
                method_names=final_methods,
                seed=config.global_seed,
            )
            row = {"true_label": rec.true_label, "topk_frac": topk_frac}
            for method_name, metrics in per_method.items():
                row[f"{method_name}_comp"] = metrics["comprehensiveness"]
                row[f"{method_name}_suff"] = metrics["sufficiency"]
            temp_rows.append(row)
        temp_df = pd.DataFrame(temp_rows)
        for method in final_methods:
            rows.append(
                {
                    "topk_frac": topk_frac,
                    "method": method,
                    "comp_mean": float(temp_df[f"{method}_comp"].mean()),
                    "suff_mean": float(temp_df[f"{method}_suff"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_publication_plots(
    out_dir: Path,
    metrics_df: pd.DataFrame,
    curve_store: dict[str, list[np.ndarray]],
    classwise_pivot: pd.DataFrame,
    topk_df: pd.DataFrame,
    final_methods: list[str],
    best_fixed_method: str,
    best_adaptive_method: str,
    config: AdaptiveXAIConfig,
) -> dict[str, str]:
    """Save publication-ready figures regenerated by the latest XAI notebook."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.frameon": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    labels = {
        "attention": "Attention",
        "ig": "Integrated Gradients",
        best_fixed_method: "Hybrid (Fixed α)",
        best_adaptive_method: "Hybrid (Adaptive α)",
        "random": "Random",
    }
    colors = {
        "attention": "#1f77b4",
        "ig": "#ff7f0e",
        best_fixed_method: "#2ca02c",
        best_adaptive_method: "#d62728",
        "random": "#7f7f7f",
    }

    def apply_axis_style(ax, grid_axis: str = "y") -> None:
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def styled_boxplot(ax, data, xlabels, ylabel, title, box_colors) -> None:
        bp = ax.boxplot(
            data,
            labels=xlabels,
            showmeans=True,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            boxprops={"linewidth": 1.2},
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
            flierprops={"marker": "o", "markerfacecolor": "gray", "markeredgecolor": "gray", "markersize": 3, "alpha": 0.45},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        apply_axis_style(ax)

    saved: dict[str, str] = {}
    x = np.concatenate([[0.0], np.asarray(config.fractions, dtype=np.float64)])

    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=config.dpi)
    for method in final_methods:
        curves = np.asarray(curve_store[method])
        mean_curve = curves.mean(axis=0)
        sem_curve = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        ax.plot(x, mean_curve, marker="o", markersize=5, linewidth=2.2, color=colors[method], label=labels[method])
        ax.fill_between(x, mean_curve - 1.96 * sem_curve, mean_curve + 1.96 * sem_curve, color=colors[method], alpha=0.14)
    ax.set_xlabel("Fraction of Tokens Removed")
    ax.set_ylabel("Predicted Probability of Original Class")
    ax.set_title("Deletion Curves (MoRF) with 95% Confidence Intervals", pad=10)
    ax.set_xlim(0.0, 1.0)
    apply_axis_style(ax, "both")
    ax.legend(loc="best")
    path = out_dir / "fig01_deletion_curves_morf_ci_pubready.png"
    save(path); saved["deletion_curves"] = str(path)

    for filename, suffix, ylabel, title in [
        ("fig02_deletion_auc_boxplot_pubready.png", "auc", "Deletion AUC (Lower is Better)", "Per-Instance Deletion AUC Distribution"),
        ("fig03_comprehensiveness_boxplot_pubready.png", "comp", f"Comprehensiveness at Top {int(config.topk_frac_comp * 100)}% (Higher is Better)", "Per-Instance Comprehensiveness Distribution"),
        ("fig04_sufficiency_boxplot_pubready.png", "suff", f"Sufficiency Gap at Top {int(config.topk_frac_comp * 100)}% (Lower is Better)", "Per-Instance Sufficiency Distribution"),
    ]:
        fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=config.dpi)
        styled_boxplot(
            ax,
            [metrics_df[f"{m}_{suffix}"].values for m in final_methods],
            [labels[m] for m in final_methods],
            ylabel,
            title,
            [colors[m] for m in final_methods],
        )
        ax.tick_params(axis="x", rotation=15)
        path = out_dir / filename
        save(path); saved[filename] = str(path)

    for filename, column, xlabel, title, color in [
        ("fig05_attention_ig_spearman_histogram_pubready.png", "spearman_attn_ig", "Spearman Correlation Between Attention and IG", "Attention–IG Rank Correlation Distribution", "#4c78a8"),
        ("fig06_adaptive_alpha_histogram_pubready.png", "alpha_adaptive", "Adaptive α", "Adaptive Weight Distribution", "#e45756"),
    ]:
        vals = metrics_df[column].values
        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=config.dpi)
        ax.hist(vals, bins=20, color=color, alpha=0.82, edgecolor="white")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.8, label=f"Mean = {vals.mean():.3f}")
        ax.axvline(np.median(vals), color="black", linestyle=":", linewidth=1.8, label=f"Median = {np.median(vals):.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Instances")
        ax.set_title(title, pad=10)
        apply_axis_style(ax)
        ax.legend(loc="best")
        path = out_dir / filename
        save(path); saved[filename] = str(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=config.dpi)
    im = ax.imshow(classwise_pivot.values, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Deletion AUC")
    ax.set_xticks(range(classwise_pivot.shape[1]))
    ax.set_xticklabels([labels.get(c, c) for c in classwise_pivot.columns], rotation=20, ha="right")
    ax.set_yticks(range(classwise_pivot.shape[0]))
    ax.set_yticklabels(classwise_pivot.index)
    ax.set_xlabel("Explanation Method")
    ax.set_ylabel("True Label")
    ax.set_title("Class-Wise Mean Deletion AUC Heatmap", pad=10)
    path = out_dir / "fig07_classwise_deletion_auc_heatmap_pubready.png"
    save(path); saved["classwise_auc_heatmap"] = str(path)

    for filename, metric_col, ylabel, title in [
        ("fig08_topk_comprehensiveness_sweep_pubready.png", "comp_mean", "Mean Comprehensiveness", "Top-k Sensitivity of Comprehensiveness"),
        ("fig09_topk_sufficiency_sweep_pubready.png", "suff_mean", "Mean Sufficiency Gap", "Top-k Sensitivity of Sufficiency"),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=config.dpi)
        for method in final_methods:
            sub = topk_df[topk_df["method"] == method]
            ax.plot(sub["topk_frac"], sub[metric_col], marker="o", markersize=5, linewidth=2.0, color=colors[method], label=labels[method])
        ax.set_xlabel("Top-k Fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        apply_axis_style(ax, "both")
        ax.legend(loc="best")
        path = out_dir / filename
        save(path); saved[filename] = str(path)

    return saved


def export_top_tokens(
    out_dir: Path,
    records: list[ExplanationRecord],
    methods: list[str],
    labels: dict[str, str],
    topn: int = 10,
) -> None:
    """Export top tokens per instance for qualitative inspection."""
    token_rows = []
    for idx, rec in enumerate(records):
        for method in methods:
            scores, bundle = get_scores_for_method(rec, method)
            if scores is None:
                continue
            order = rec.deletable_indices[np.argsort(-scores[rec.deletable_indices])][:topn]
            pairs = [(rec.tokens_full[i], float(scores[i])) for i in order]
            token_rows.append(
                {
                    "id": idx,
                    "true_label": rec.true_label,
                    "pred_label": rec.pred_label,
                    "method": labels.get(method, method),
                    "method_full": method,
                    "alpha_adaptive": bundle.get("alpha_adaptive", np.nan) if method.startswith("hybrid_adaptive") else np.nan,
                    "top_tokens": json.dumps(pairs, ensure_ascii=False),
                    "text": rec.text,
                }
            )
    pd.DataFrame(token_rows).to_csv(out_dir / "top_tokens_per_instance.csv", index=False)


def run_adaptive_xai(config: AdaptiveXAIConfig) -> dict[str, str]:
    """Execute the full adaptive-XAI experiment and save all outputs."""
    set_seed(config.global_seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "adaptive_xai_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    df = load_text_label_csv(config.data_path)
    num_classes = len(sorted(df["label"].unique()))
    split = stratified_text_split(df, test_size=config.test_size, random_state=config.split_random_state)
    test_df = pd.DataFrame({"text": split.eval_texts, "label": split.eval_labels})
    if config.eval_pool_size is not None and config.eval_pool_size < len(test_df):
        keep_idx = stratified_subsample_indices(test_df["label"].values, config.eval_pool_size, seed=config.global_seed)
        test_df = test_df.iloc[keep_idx].reset_index(drop=True)
    test_df.to_csv(out_dir / "xai_eval_instances.csv", index=False)
    print(f">>> Test instances available for XAI analysis: {len(test_df)}")

    model, tokenizer = load_model_and_tokenizer(config, num_classes=num_classes, device=device)
    candidates = make_candidate_specs(config)
    candidate_names = [str(spec["name"]) for spec in candidates]
    print(f">>> Candidate methods in screening stage: {len(candidate_names)}")

    print(">>> Building explanation cache for screening...")
    records_screen = build_explanation_cache(test_df, model, tokenizer, device, config.max_length, config.m_steps_screen, "screening")
    print(">>> Screening candidate methods...")
    screening_metrics, _ = evaluate_methods(records_screen, model, tokenizer, device, candidate_names, config, include_text=False)
    screening_metrics.to_csv(out_dir / "screening_metrics_all_candidates.csv", index=False)
    screening_summary = summarize_metrics_df(screening_metrics, candidate_names, config.bootstrap_b, config.bootstrap_seed)
    screening_summary.to_csv(out_dir / "screening_summary_all_candidates.csv", index=False)

    fixed_summary = screening_summary[screening_summary["method"].str.startswith("hybrid_fixed")].reset_index(drop=True)
    adaptive_summary = screening_summary[screening_summary["method"].str.startswith("hybrid_adaptive")].reset_index(drop=True)
    best_fixed_method = str(fixed_summary.iloc[0]["method"])
    best_adaptive_method = str(adaptive_summary.iloc[0]["method"])
    print(f">>> Best fixed hybrid from screening: {best_fixed_method}")
    print(f">>> Best adaptive hybrid from screening: {best_adaptive_method}")

    print(">>> Rebuilding explanation cache for deep evaluation...")
    records_deep = build_explanation_cache(test_df, model, tokenizer, device, config.max_length, config.m_steps_deep, "deep")
    final_methods = ["attention", "ig", best_fixed_method, best_adaptive_method, "random"]
    print(">>> Final shortlist:", final_methods)

    final_metrics, curve_store = evaluate_methods(records_deep, model, tokenizer, device, final_methods, config, include_text=True)
    # Keep the column name used by the latest notebook for the selected adaptive method.
    adaptive_parsed = parse_method_tag(best_adaptive_method)
    alpha_vals = []
    for rec in records_deep:
        bundle = build_score_bundle(
            rec,
            norm=str(adaptive_parsed["norm"]),
            alpha_fixed=0.85,
            alpha_min=float(adaptive_parsed["amin"]),
            alpha_max=float(adaptive_parsed["amax"]),
        )
        alpha_vals.append(bundle["alpha_adaptive"])
    final_metrics["alpha_adaptive"] = alpha_vals
    final_metrics.to_csv(out_dir / "final_metrics_shortlist.csv", index=False)

    summary = summarize_metrics_df(final_metrics, final_methods, config.bootstrap_b, config.bootstrap_seed)
    summary.to_csv(out_dir / "summary_overall.csv", index=False)

    runs_df, runs_agg = repeated_stratified_auc_runs(final_metrics, final_metrics["true_label"].values, final_methods, config.n_runs, config.run_sample_size)
    runs_df.to_csv(out_dir / "repeated_runs_aggregate.csv", index=False)
    runs_agg.to_csv(out_dir / "repeated_runs_summary.csv", index=False)

    friedman_stat, friedman_p = friedmanchisquare(*[final_metrics[f"{m}_auc"].values for m in final_methods])
    pd.DataFrame([{"statistic": friedman_stat, "p_value": friedman_p}]).to_csv(out_dir / "friedman_auc.csv", index=False)

    pairwise = pairwise_tests(final_metrics, final_methods)
    pairwise.to_csv(out_dir / "pairwise_wilcoxon_with_holm.csv", index=False)

    print(">>> Running top-k sweep for final shortlist...")
    topk_df = run_topk_sweep(records_deep, model, tokenizer, device, final_methods, config)
    topk_df.to_csv(out_dir / "topk_sweep_summary.csv", index=False)

    classwise_rows = []
    for cls in sorted(final_metrics["true_label"].unique()):
        sub = final_metrics[final_metrics["true_label"] == cls]
        for method in final_methods:
            vals = sub[f"{method}_auc"].values
            classwise_rows.append({"true_label": cls, "method": method, "mean_auc": float(np.mean(vals)), "std_auc": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)})
    classwise_df = pd.DataFrame(classwise_rows)
    classwise_df.to_csv(out_dir / "classwise_auc.csv", index=False)
    classwise_pivot = classwise_df.pivot(index="true_label", columns="method", values="mean_auc")
    classwise_pivot.to_csv(out_dir / "classwise_auc_pivot.csv")

    median_conf = final_metrics["pred_conf"].median()
    median_len = final_metrics["num_deletable"].median()
    subsets = [
        ("low_confidence", final_metrics["pred_conf"] <= median_conf),
        ("high_confidence", final_metrics["pred_conf"] > median_conf),
        ("short_text", final_metrics["num_deletable"] <= median_len),
        ("long_text", final_metrics["num_deletable"] > median_len),
    ]
    subset_rows = []
    for subset_name, mask in subsets:
        sub = final_metrics[mask]
        for method in final_methods:
            subset_rows.append({"subset": subset_name, "method": method, "mean_auc": float(sub[f"{method}_auc"].mean()), "mean_comp": float(sub[f"{method}_comp"].mean()), "mean_suff": float(sub[f"{method}_suff"].mean()), "n": int(len(sub))})
    pd.DataFrame(subset_rows).to_csv(out_dir / "subset_analysis.csv", index=False)

    print(">>> Saving publication-ready plots...")
    save_publication_plots(out_dir, final_metrics, curve_store, classwise_pivot, topk_df, final_methods, best_fixed_method, best_adaptive_method, config)

    labels = {"attention": "Attention", "ig": "Integrated Gradients", best_fixed_method: "Hybrid (Fixed α)", best_adaptive_method: "Hybrid (Adaptive α)", "random": "Random"}
    export_top_tokens(out_dir, records_deep, ["attention", "ig", best_fixed_method, best_adaptive_method], labels)

    best_auc_method = str(summary.iloc[0]["method"])
    report_lines = [
        "Explainability / Faithfulness Summary",
        "=" * 60,
        f"Instances analyzed: {len(final_metrics)}",
        f"Device: {device}",
        f"IG steps screening: {config.m_steps_screen}",
        f"IG steps deep evaluation: {config.m_steps_deep}",
        f"Deletion fractions: {config.fractions}",
        f"Top-k default for comp/suff: {config.topk_frac_comp}",
        f"Top-k sweep: {config.topk_frac_sweep}",
        "",
        "Best screening methods:",
        f"  best fixed hybrid    : {best_fixed_method}",
        f"  best adaptive hybrid : {best_adaptive_method}",
        "",
        "Overall ranking by deletion AUC (lower is better):",
    ]
    for _, row in summary.iterrows():
        report_lines.append(
            f"  {labels.get(row['method'], row['method']):>22s} | AUC {row['auc_mean']:.4f} ± {row['auc_std']:.4f} "
            f"[95% CI {row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}] | Comp {row['comp_mean']:.4f} | Suff {row['suff_mean']:.4f}"
        )
    report_lines.extend(["", "Repeated stratified runs (mean of run-level means):"])
    for _, row in runs_agg.iterrows():
        report_lines.append(f"  {labels.get(row['method'], row['method']):>22s} | mean AUC {row['repeated_mean_auc']:.4f} ± {row['repeated_std_auc']:.4f}")
    report_lines.extend([
        "",
        f"Friedman test on per-instance AUCs: stat={friedman_stat:.4f}, p={friedman_p:.6g}",
        "",
        "Pairwise Wilcoxon tests (Holm-adjusted):",
    ])
    for _, row in pairwise.sort_values(["metric", "p_holm"]).iterrows():
        report_lines.append(
            f"  {row['metric']:>18s} | {labels.get(row['method_a'], row['method_a'])} vs {labels.get(row['method_b'], row['method_b'])} | "
            f"mean diff={row['mean_diff_a_minus_b']:.6f} | p={row['p_value']:.6g} | p_holm={row['p_holm']:.6g}"
        )
    report_lines.extend([
        "",
        f"Attention–IG Spearman: mean={final_metrics['spearman_attn_ig'].mean():.4f}, median={final_metrics['spearman_attn_ig'].median():.4f}",
        f"Adaptive alpha (best adaptive hybrid): mean={final_metrics['alpha_adaptive'].mean():.4f}, median={final_metrics['alpha_adaptive'].median():.4f}, min={final_metrics['alpha_adaptive'].min():.4f}, max={final_metrics['alpha_adaptive'].max():.4f}",
        "",
        f"Best AUC method: {labels.get(best_auc_method, best_auc_method)}",
    ])
    (out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\n>>> Done. All outputs saved to: {out_dir}")
    return {"out_dir": str(out_dir), "best_fixed_method": best_fixed_method, "best_adaptive_method": best_adaptive_method, "best_auc_method": best_auc_method}


def parse_args() -> AdaptiveXAIConfig:
    parser = argparse.ArgumentParser(description="Run adaptive XAI faithfulness analysis.")
    parser.add_argument("--model-path", default="runs_emotion_cls_2/best_model_cls.pt")
    parser.add_argument("--data-path", default="text.csv")
    parser.add_argument("--out-dir", default="runs_emotion_cls_2/xai_faithfulness_v6_pubready_figures")
    parser.add_argument("--roberta-name", default="roberta-base")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--m-steps-screen", type=int, default=16)
    parser.add_argument("--m-steps-deep", type=int, default=32)
    parser.add_argument("--eval-pool-size", type=int, default=600)
    parser.add_argument("--n-runs", type=int, default=7)
    parser.add_argument("--run-sample-size", type=int, default=400)
    parser.add_argument("--topk-frac-comp", type=float, default=0.20)
    parser.add_argument("--random-restarts", type=int, default=5)
    args = parser.parse_args()
    return AdaptiveXAIConfig(**vars(args))


def main() -> None:
    run_adaptive_xai(parse_args())


if __name__ == "__main__":
    main()
