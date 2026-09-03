#!/usr/bin/env python3
"""Build manuscript-facing tables and figures from frozen COVER-MTL results.

The script is intentionally read-only with respect to the formal experiment
directories.  All derived artifacts are written below ``paper_results``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_results"
TABLE_MAIN = OUT / "tables" / "main"
TABLE_ADD = OUT / "tables" / "additional"
FIG_MAIN = OUT / "figures" / "main"
FIG_ADD = OUT / "figures" / "additional"
DATA_OUT = OUT / "data"

METHODS = ["Pool", "STL", "HPS", "MMoE", "ARMUL", "FLARCC", "COVER"]
METHOD_LABEL = {
    "Pool": "Pool",
    "STL": "STL",
    "HPS": "HPS",
    "MMoE": "MMoE",
    "ARMUL": "ARMUL",
    "FLARCC": "FLARCC",
    "Average-Moment": "Average-Moment",
    "COVER": "COVER",
    "TissueMean": "Tissue Mean",
    "GlobalMean": "Global Mean",
}

# The focal method follows the PPCI deep blue.  Comparator colors are drawn
# from a colorblind-safe palette and are intentionally less visually dominant.
COLORS = {
    "COVER": "#005A9E",
    "HPS": "#6F6F6F",
    "Average-Moment": "#E69F00",
    "Pool": "#009E73",
    "STL": "#CC79A7",
    "MMoE": "#56B4E9",
    "ARMUL": "#D89000",
    "FLARCC": "#D55E00",
}
MARKERS = {
    "COVER": "o",
    "HPS": "s",
    "Average-Moment": "D",
    "Pool": "^",
    "STL": "v",
    "MMoE": "P",
    "ARMUL": "X",
    "FLARCC": "<",
}
LINESTYLES = {
    "COVER": "-",
    "HPS": "--",
    "Average-Moment": "-.",
    "Pool": "-",
    "STL": "--",
    "MMoE": ":",
    "ARMUL": "-.",
    "FLARCC": (0, (3, 1, 1, 1)),
}


def set_paper_style() -> None:
    """Match PPCI sizing while using the JMLR template's CMR typography."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Latin Modern Roman",
                "cmr10",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.titlesize": 21,
            "axes.labelsize": 21,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 19,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.grid": False,
        }
    )


def prepare_output_dirs() -> None:
    for path in [TABLE_MAIN, TABLE_ADD, FIG_MAIN, FIG_ADD, DATA_OUT]:
        path.mkdir(parents=True, exist_ok=True)


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="0.85", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_csvs(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in sorted(files):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No input files matched the requested formal result block.")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_metrics(relative_dir: str) -> pd.DataFrame:
    return read_csvs((ROOT / relative_dir).glob("**/metrics.csv"))


def summarize(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    grouped = data.groupby(list(group_cols), dropna=False)
    blocks = []
    for metric in metrics:
        block = grouped[metric].agg(["count", "mean", "std"]).reset_index()
        block = block.rename(
            columns={
                "count": f"{metric}_n",
                "mean": f"{metric}_mean",
                "std": f"{metric}_sd",
            }
        )
        block[f"{metric}_se"] = block[f"{metric}_sd"] / np.sqrt(block[f"{metric}_n"])
        blocks.append(block)
    result = blocks[0]
    for block in blocks[1:]:
        result = result.merge(block, on=list(group_cols), how="outer", validate="one_to_one")
    return result


def mean_sd(mean: float, sd: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def latex_escape(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("–", "--")
    )


def ranked_cell(
    frame: pd.DataFrame,
    method: str,
    setting: str,
    mean_col: str,
    sd_col: str,
    digits: int = 4,
) -> str:
    subset = frame.loc[frame["setting"].eq(setting)].sort_values(mean_col)
    row = subset.loc[subset["method"].eq(method)].iloc[0]
    value = mean_sd(float(row[mean_col]), float(row[sd_col]), digits)
    ranks = subset["method"].tolist()
    if method == ranks[0]:
        return rf"\textbf{{{value}}}"
    if len(ranks) > 1 and method == ranks[1]:
        return rf"\underline{{{value}}}"
    return value


def write_main_heterogeneity_table() -> pd.DataFrame:
    settings = [
        (
            "Homogeneous",
            "results/controls/within_0p200/homogeneous",
        ),
        (
            "Covariate only",
            "results/controls/within_0p200/covariate_only",
        ),
        (
            "Posterior only",
            "results/controls/within_0p200/posterior_only",
        ),
        (
            "Both--moderate",
            "results/main/within_0p200/both_overlap_aligned",
        ),
        (
            "Both--strong",
            "results/main/within_0p300/both_overlap_aligned",
        ),
    ]
    frames = []
    for label, relative_dir in settings:
        frame = read_metrics(relative_dir)
        frame["setting"] = label
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.loc[raw["method"].isin(METHODS)].copy()
    summary = summarize(
        raw,
        ["setting", "method"],
        [
            "excess_mse",
            "prediction_mse",
            "worst_task_excess_mse",
            "common_mse",
            "deviation_mse",
        ],
    )
    summary.to_csv(DATA_OUT / "heterogeneity_regimes_summary.csv", index=False)

    setting_order = [label for label, _ in settings]
    pivot = summary.pivot(index="method", columns="setting", values="excess_mse_mean")
    pivot = pivot.reindex(index=METHODS, columns=setting_order)
    pivot.to_csv(TABLE_MAIN / "table_main_heterogeneity_numeric.csv")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Homogeneous & Covariate only & Posterior only & Both--moderate & Both--strong \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [
            ranked_cell(
                summary,
                method,
                setting,
                "excess_mse_mean",
                "excess_mse_sd",
            )
            for setting in setting_order
        ]
        lines.append(f"{latex_escape(METHOD_LABEL[method])} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Task-balanced excess mean squared error across heterogeneity regimes. Entries are means (standard deviations) over 100 repetitions. The best and second-best results in each column are shown in bold and underlined, respectively.}",
        r"\label{tab:simulation-main}",
        r"\end{table}",
    ])
    (TABLE_MAIN / "table_main_heterogeneity.tex").write_text("\n".join(lines) + "\n")
    return summary


def plot_main_mechanism() -> pd.DataFrame:
    raw = read_metrics("results/mechanism")
    raw = raw.loc[raw["method"].isin(["HPS", "Average-Moment", "COVER"])].copy()
    summary = summarize(
        raw,
        ["axis", "value", "method"],
        [
            "excess_mse",
            "deviation_mse",
            "selected_coupling",
            "normalized_overlap_strength",
            "overlap_energy",
            "supported_contrast_fraction",
        ],
    )
    summary.to_csv(DATA_OUT / "mechanism_summary.csv", index=False)

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    method_order = ["HPS", "Average-Moment", "COVER"]
    for axis_name, ax in zip(["covariate", "posterior"], axes):
        block = summary.loc[summary["axis"].eq(axis_name)].copy()
        for method in method_order:
            method_block = block.loc[block["method"].eq(method)].sort_values("value")
            if axis_name == "covariate":
                x = method_block["normalized_overlap_strength_mean"].to_numpy()
            else:
                x = method_block["value"].to_numpy()
            y = method_block["excess_mse_mean"].to_numpy()
            ci = 1.96 * method_block["excess_mse_se"].to_numpy()
            linewidth = 2.8 if method == "COVER" else 2.0
            zorder = 6 if method == "COVER" else 3
            ax.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=linewidth,
                markersize=6,
                label=METHOD_LABEL[method],
                zorder=zorder,
            )
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                color=COLORS[method],
                alpha=0.14 if method == "COVER" else 0.08,
                linewidth=0,
                zorder=zorder - 1,
            )
        style_axis(ax)
    axes[0].set_title("Covariate overlap", loc="left", pad=8)
    axes[0].set_xlabel("Normalized covariate overlap")
    axes[0].set_ylabel("Excess MSE")
    axes[1].set_title("Posterior heterogeneity", loc="left", pad=8)
    axes[1].set_xlabel("Posterior strength")
    axes[1].set_ylabel("Excess MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, FIG_MAIN / "figure_main_mechanism")
    return summary


def plot_main_scaling() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_raw = read_metrics("results/scaling")
    task_raw = task_raw.loc[task_raw["method"].isin(METHODS)].copy()
    # The directory also retains earlier calibration axes.  The frozen formal
    # audit design contains only T = 24, 96, and 192.
    task_raw = task_raw.loc[task_raw["scaling_value"].isin([24, 96, 192])].copy()
    dim_raw = read_metrics("results/scaling")
    dim_raw = dim_raw.loc[dim_raw["method"].isin(METHODS)].copy()
    dim_raw = dim_raw.loc[dim_raw["scaling_value"].isin([50, 100, 200])].copy()
    # Legacy formal runs counted COVER's pairwise consensus coordinates as
    # model parameters and omitted the fixed task heads returned by ARMUL and
    # FLARCC. Normalize to the public-package convention: parameter_count
    # contains fitted quantities needed for prediction, while the consensus
    # coordinates are optimization auxiliaries reported separately in text.
    for frame in (task_raw, dim_raw):
        if "auxiliary_parameter_count" in frame:
            continue
        task_count = frame["num_tasks"].astype(int)
        representation_dim = frame["representation_dim"].astype(int)
        fixed_head = frame["method"].isin(("ARMUL", "FLARCC"))
        frame.loc[fixed_head, "parameter_count"] += (
            task_count[fixed_head] * representation_dim[fixed_head]
        )
        cover = frame["method"].eq("COVER") & frame["selected_coupling"].gt(0)
        frame.loc[cover, "parameter_count"] -= (
            task_count[cover]
            * (task_count[cover] - 1)
            // 2
            * representation_dim[cover]
        )
    task_summary = summarize(
        task_raw,
        ["scaling_value", "method"],
        ["excess_mse", "workflow_seconds", "peak_device_memory_mb", "parameter_count", "selected_coupling"],
    )
    dim_summary = summarize(
        dim_raw,
        ["scaling_value", "method"],
        ["excess_mse", "workflow_seconds", "peak_device_memory_mb", "parameter_count", "selected_coupling"],
    )
    task_medians = (
        task_raw.groupby(["scaling_value", "method"])["selected_coupling"]
        .median()
        .rename("selected_coupling_median")
        .reset_index()
    )
    dim_medians = (
        dim_raw.groupby(["scaling_value", "method"])["selected_coupling"]
        .median()
        .rename("selected_coupling_median")
        .reset_index()
    )
    task_summary = task_summary.merge(
        task_medians,
        on=["scaling_value", "method"],
        how="left",
        validate="one_to_one",
    )
    dim_summary = dim_summary.merge(
        dim_medians,
        on=["scaling_value", "method"],
        how="left",
        validate="one_to_one",
    )
    task_summary.to_csv(DATA_OUT / "scaling_tasks_summary.csv", index=False)
    dim_summary.to_csv(DATA_OUT / "scaling_dimensions_summary.csv", index=False)

    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for method in METHODS:
        for ax, block, y_col, se_col in [
            (axes[0], task_summary, "excess_mse_mean", "excess_mse_se"),
            (axes[1], dim_summary, "excess_mse_mean", "excess_mse_se"),
        ]:
            method_block = block.loc[block["method"].eq(method)].sort_values("scaling_value")
            x = method_block["scaling_value"].to_numpy()
            y = method_block[y_col].to_numpy()
            ci = 1.96 * method_block[se_col].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=ci,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.8 if method == "COVER" else 1.65,
                markersize=6 if method == "COVER" else 5,
                capsize=2.5,
                label=METHOD_LABEL[method],
                zorder=7 if method == "COVER" else 3,
            )
        method_block = task_raw.loc[task_raw["method"].eq(method)].copy()
        time_summary = (
            method_block.groupby("scaling_value")["workflow_seconds"]
            .agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reset_index()
            .sort_values("scaling_value")
        )
        x = time_summary["scaling_value"].to_numpy()
        y = time_summary["median"].to_numpy()
        yerr = np.vstack([y - time_summary["q1"].to_numpy(), time_summary["q3"].to_numpy() - y])
        axes[2].errorbar(
            x,
            y,
            yerr=yerr,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.8 if method == "COVER" else 1.65,
            markersize=6 if method == "COVER" else 5,
            capsize=2.5,
            label=METHOD_LABEL[method],
            zorder=7 if method == "COVER" else 3,
        )
    axes[0].set_title("Increasing task count", loc="left", pad=8)
    axes[0].set_xlabel("Number of tasks $T$")
    axes[0].set_ylabel("Excess MSE")
    axes[0].set_xticks([24, 96, 192])
    axes[1].set_title("Increasing input dimension", loc="left", pad=8)
    axes[1].set_xlabel("Input dimension $p$")
    axes[1].set_ylabel("Excess MSE")
    axes[1].set_xticks([50, 100, 200])
    axes[2].set_title("End-to-end computation", loc="left", pad=8)
    axes[2].set_xlabel("Number of tasks $T$")
    axes[2].set_ylabel("Workflow time (seconds)")
    axes[2].set_xticks([24, 96, 192])
    axes[2].set_yscale("log")
    for ax in axes:
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=7,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, FIG_MAIN / "figure_main_scaling")
    return task_summary, dim_summary


def write_main_gtex_table() -> pd.DataFrame:
    result_dir = ROOT / "experiments" / "gtex" / "results" / "repeated_cv_v1"
    response = pd.read_csv(result_dir / "response_mean_sd.csv")
    overall = pd.read_csv(result_dir / "overall_mean_sd.csv")
    baseline_response = pd.read_csv(result_dir / "mean_baseline_response_mean_sd.csv")
    baseline_overall = pd.read_csv(result_dir / "mean_baseline_overall_mean_sd.csv")

    baseline_response = baseline_response.loc[baseline_response["method"].isin(["TissueMean", "GlobalMean"])]
    baseline_overall = baseline_overall.loc[baseline_overall["method"].isin(["TissueMean", "GlobalMean"])]
    response_all = pd.concat([response, baseline_response], ignore_index=True, sort=False)
    overall_all = pd.concat([overall, baseline_overall], ignore_index=True, sort=False)
    overall_all["response"] = "Overall"
    combined = pd.concat([response_all, overall_all], ignore_index=True, sort=False)
    combined.to_csv(DATA_OUT / "gtex_main_summary.csv", index=False)

    order = ["GlobalMean", "TissueMean"] + METHODS
    columns = ["JAM2", "SH2D2A", "Overall"]
    numeric = combined.pivot(index="method", columns="response", values="standardized_mse_mean")
    numeric = numeric.reindex(index=order, columns=columns)
    numeric.to_csv(TABLE_MAIN / "table_main_gtex_numeric.csv")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & JAM2 & SH2D2A & Overall \\",
        r"\midrule",
    ]
    for index, method in enumerate(order):
        cells = []
        for response_name in columns:
            block = combined.loc[combined["response"].eq(response_name)].sort_values("standardized_mse_mean")
            row = block.loc[block["method"].eq(method)].iloc[0]
            value = mean_sd(row["standardized_mse_mean"], row["standardized_mse_sd"])
            ranks = block["method"].tolist()
            if method == ranks[0]:
                value = rf"\textbf{{{value}}}"
            elif len(ranks) > 1 and method == ranks[1]:
                value = rf"\underline{{{value}}}"
            cells.append(value)
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + r" \\")
        if index == 1:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Prediction performance in the GTEx brain-tissue analysis. Entries are standardized mean squared errors, reported as means (standard deviations) over 20 repeated five-fold cross-validation partitions.}",
        r"\label{tab:gtex-main}",
        r"\end{table}",
    ])
    (TABLE_MAIN / "table_main_gtex.tex").write_text("\n".join(lines) + "\n")
    return combined


def plot_main_gtex_overlap() -> None:
    source = ROOT / "experiments" / "gtex" / "results" / "repeated_cv_v1" / "pairwise_overlap_mean_sd.csv"
    data = pd.read_csv(source)
    tissues = sorted(set(data["left_tissue"]) | set(data["right_tissue"]))
    short = {
        "brain_amygdala": "Amygdala",
        "brain_anterior_cingulate_cortex_ba24": "ACC",
        "brain_caudate_basal_ganglia": "Caudate",
        "brain_cerebellar_hemisphere": "Cereb. hem.",
        "brain_cerebellum": "Cerebellum",
        "brain_cortex": "Cortex",
        "brain_frontal_cortex_ba9": "Frontal",
        "brain_hypothalamus": "Hypothalamus",
        "brain_nucleus_accumbens_basal_ganglia": "N. accumbens",
        "brain_putamen_basal_ganglia": "Putamen",
        "brain_spinal_cord_cervical_c-1": "Spinal cord",
    }
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.2))
    image = None
    for ax, response_name in zip(axes, ["JAM2", "SH2D2A"]):
        matrix = pd.DataFrame(np.eye(len(tissues)), index=tissues, columns=tissues)
        block = data.loc[data["response"].eq(response_name)]
        for row in block.itertuples(index=False):
            matrix.loc[row.left_tissue, row.right_tissue] = row.normalized_overlap_trace_mean
            matrix.loc[row.right_tissue, row.left_tissue] = row.normalized_overlap_trace_mean
        image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(response_name, loc="left", pad=8)
        labels = [short.get(tissue, tissue) for tissue in tissues]
        ax.set_xticks(np.arange(len(tissues)), labels=labels, rotation=55, ha="right")
        ax.set_yticks(np.arange(len(tissues)), labels=labels)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.89, bottom=0.29, top=0.90, wspace=0.30)
    fig.canvas.draw()
    right_position = axes[1].get_position()
    color_axis = fig.add_axes(
        [right_position.x1 + 0.010, right_position.y0, 0.018, right_position.height]
    )
    cbar = fig.colorbar(image, cax=color_axis)
    cbar.set_label("Normalized overlap")
    save_figure(fig, FIG_MAIN / "figure_main_gtex_overlap")


def write_secondary_metrics_table(heterogeneity: pd.DataFrame) -> None:
    settings = ["Homogeneous", "Covariate only", "Posterior only", "Both--moderate", "Both--strong"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Homogeneous & Covariate only & Posterior only & Both--moderate & Both--strong \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel A: Worst-task excess MSE}} \\",
    ]
    for method in METHODS:
        cells = []
        for setting in settings:
            row = heterogeneity.loc[
                heterogeneity["setting"].eq(setting) & heterogeneity["method"].eq(method)
            ].iloc[0]
            cells.append(mean_sd(row["worst_task_excess_mse_mean"], row["worst_task_excess_mse_sd"]))
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\midrule", r"\multicolumn{6}{l}{\textit{Panel B: Task-deviation MSE}} \\"])
    for method in METHODS:
        cells = []
        for setting in settings:
            row = heterogeneity.loc[
                heterogeneity["setting"].eq(setting) & heterogeneity["method"].eq(method)
            ].iloc[0]
            cells.append(mean_sd(row["deviation_mse_mean"], row["deviation_mse_sd"]))
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Secondary recovery metrics across heterogeneity regimes. Entries are means (standard deviations) over 100 repetitions.}",
        r"\label{tab:simulation-secondary}",
        r"\end{table}",
    ])
    (TABLE_ADD / "table_additional_secondary_metrics.tex").write_text("\n".join(lines) + "\n")


def write_random_alignment_table() -> pd.DataFrame:
    raw = read_metrics("results/random_alignment")
    raw = raw.loc[raw["method"].isin(METHODS)].copy()
    summary = summarize(
        raw,
        ["method"],
        ["excess_mse", "worst_task_excess_mse", "deviation_mse"],
    )
    summary.to_csv(DATA_OUT / "random_alignment_summary.csv", index=False)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Excess MSE & Worst-task excess MSE & Task-deviation MSE \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = summary.loc[summary["method"].eq(method)].iloc[0]
        cells = [
            mean_sd(row["excess_mse_mean"], row["excess_mse_sd"]),
            mean_sd(row["worst_task_excess_mse_mean"], row["worst_task_excess_mse_sd"]),
            mean_sd(row["deviation_mse_mean"], row["deviation_mse_sd"]),
        ]
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Performance under random alignment between covariate and posterior heterogeneity. Entries are means (standard deviations) over 100 repetitions.}",
        r"\label{tab:random-alignment}",
        r"\end{table}",
    ])
    (TABLE_ADD / "table_additional_random_alignment.tex").write_text("\n".join(lines) + "\n")
    return summary


def plot_additional_mechanism(mechanism: pd.DataFrame) -> None:
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    covariate = mechanism.loc[
        mechanism["axis"].eq("covariate") & mechanism["method"].eq("COVER")
    ].sort_values("value")
    x = covariate["value"].to_numpy()
    axes[0].plot(
        x,
        covariate["normalized_overlap_strength_mean"],
        color=COLORS["COVER"],
        marker="o",
        linewidth=2.6,
        label="Normalized overlap",
    )
    axes[0].plot(
        x,
        covariate["supported_contrast_fraction_mean"],
        color="#2F7D4A",
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label="Supported component contrast",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Weak-direction variance")
    axes[0].set_ylabel("Normalized diagnostic")
    axes[0].set_title("Overlap diagnostics", loc="left", pad=8)
    axes[0].legend(frameon=False, loc="best")
    style_axis(axes[0])

    posterior = mechanism.loc[mechanism["axis"].eq("posterior")]
    for method in ["HPS", "Average-Moment", "COVER"]:
        block = posterior.loc[posterior["method"].eq(method)].sort_values("value")
        axes[1].plot(
            block["value"],
            block["deviation_mse_mean"],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.6 if method == "COVER" else 2.0,
            label=METHOD_LABEL[method],
            zorder=6 if method == "COVER" else 3,
        )
    axes[1].set_xlabel("Posterior strength")
    axes[1].set_ylabel("Task-specific component MSE")
    axes[1].set_title("Task-specific component recovery", loc="left", pad=8)
    axes[1].legend(frameon=False, loc="best")
    style_axis(axes[1])
    fig.tight_layout()
    save_figure(fig, FIG_ADD / "figure_additional_mechanism_diagnostics")


def plot_sensitivity() -> pd.DataFrame:
    raw = read_metrics("results/sensitivity")
    raw = raw.loc[raw["method"].isin(["HPS", "COVER"])].copy()
    order = ["base", "d12", "d36", "narrow", "wide", "deep"]
    labels = ["Base", "$d=12$", "$d=36$", "Narrow", "Wide", "Deep"]
    raw["variant"] = pd.Categorical(raw["variant"], categories=order, ordered=True)
    raw.sort_values(["variant", "method", "initialization"]).to_csv(
        DATA_OUT / "sensitivity_all_initializations.csv", index=False
    )
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    rng = np.random.default_rng(20260831)
    offsets = {"HPS": -0.10, "COVER": 0.10}
    for method in ["HPS", "COVER"]:
        for index, variant in enumerate(order):
            values = raw.loc[
                raw["variant"].astype(str).eq(variant) & raw["method"].eq(method), "excess_mse"
            ].to_numpy()
            jitter = rng.normal(0.0, 0.018, size=len(values))
            axes[0].scatter(
                index + offsets[method] + jitter,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                s=42 if method == "COVER" else 34,
                alpha=0.95,
                label=METHOD_LABEL[method] if index == 0 else None,
                zorder=5 if method == "COVER" else 3,
            )
            axes[0].plot(
                [index + offsets[method] - 0.055, index + offsets[method] + 0.055],
                [np.mean(values), np.mean(values)],
                color=COLORS[method],
                linewidth=2.3,
                zorder=6,
            )
    cover = raw.loc[raw["method"].eq("COVER")]
    for index, variant in enumerate(order):
        values = cover.loc[cover["variant"].astype(str).eq(variant), "selected_coupling"].to_numpy()
        jitter = rng.normal(0.0, 0.025, size=len(values))
        axes[1].scatter(
            index + jitter,
            values,
            color=COLORS["COVER"],
            marker="o",
            s=42,
            alpha=0.95,
            zorder=5,
        )
        axes[1].plot(
            [index - 0.08, index + 0.08],
            [np.median(values), np.median(values)],
            color=COLORS["COVER"],
            linewidth=2.3,
            zorder=6,
        )
    axes[0].set_title("Prediction sensitivity", loc="left", pad=8)
    axes[0].set_ylabel("Excess MSE")
    axes[0].legend(frameon=False, loc="best")
    axes[1].set_title(r"Selected COVER $\lambda$", loc="left", pad=8)
    axes[1].set_ylabel(r"Selected $\lambda$")
    if np.all(cover["selected_coupling"].to_numpy() > 0):
        axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(np.arange(len(order)), labels=labels)
        style_axis(ax)
    fig.tight_layout()
    save_figure(fig, FIG_ADD / "figure_additional_sensitivity")
    return raw


def write_scaling_diagnostics_table(
    task_summary: pd.DataFrame,
    dim_summary: pd.DataFrame,
) -> None:
    task_max = float(task_summary["scaling_value"].max())
    dim_max = float(dim_summary["scaling_value"].max())
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf"& \multicolumn{{3}}{{c}}{{\(T={int(task_max)}\)}} & \multicolumn{{3}}{{c}}{{\(p={int(dim_max)}\)}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Memory (MB) & Pred. params. & Selected \(\lambda\) & Memory (MB) & Pred. params. & Selected \(\lambda\) \\",
        r"\midrule",
    ]
    for method in METHODS:
        task = task_summary.loc[
            task_summary["scaling_value"].eq(task_max) & task_summary["method"].eq(method)
        ].iloc[0]
        dim = dim_summary.loc[
            dim_summary["scaling_value"].eq(dim_max) & dim_summary["method"].eq(method)
        ].iloc[0]
        cells = [
            f"{task['peak_device_memory_mb_mean']:.1f}",
            f"{task['parameter_count_mean']:.0f}",
            f"{task['selected_coupling_median']:.2g}",
            f"{dim['peak_device_memory_mb_mean']:.1f}",
            f"{dim['parameter_count_mean']:.0f}",
            f"{dim['selected_coupling_median']:.2g}",
        ]
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Resource use at the largest task-count and input-dimension settings. Memory and predictive parameter counts are means; selected regularization parameters are medians over 100 repetitions.}",
        r"\label{tab:scaling-resources}",
        r"\end{table}",
    ])
    (TABLE_ADD / "table_additional_scaling_resources.tex").write_text("\n".join(lines) + "\n")


def plot_outlier() -> pd.DataFrame:
    raw = read_metrics("results/outlier")
    raw = raw.loc[raw["method"].eq("COVER")].copy()
    metrics = [
        "population_outlier_overlap_degree",
        "learned_outlier_overlap_degree",
        "outlier_excess_mse",
        "mean_inlier_excess_mse",
    ]
    summary = summarize(raw, ["overlap"], metrics).sort_values("overlap")
    summary.to_csv(DATA_OUT / "outlier_overlap_summary.csv", index=False)
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = summary["overlap"].to_numpy()
    axes[0].plot(
        x,
        summary["population_outlier_overlap_degree_mean"],
        color="#2F7D4A",
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label="Population",
    )
    axes[0].plot(
        x,
        summary["learned_outlier_overlap_degree_mean"],
        color=COLORS["COVER"],
        marker="o",
        linewidth=2.7,
        label="Learned",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Outlier covariate overlap")
    axes[0].set_ylabel("Normalized overlap")
    axes[0].set_title("Overlap-adaptive pooling", loc="left", pad=8)
    axes[0].legend(frameon=False, loc="best")
    style_axis(axes[0])

    for metric, label, color, marker, linestyle in [
        ("outlier_excess_mse", "Outlying task", COLORS["COVER"], "o", "-"),
        ("mean_inlier_excess_mse", "Inlier average", "#6F6F6F", "s", "--"),
    ]:
        mean_values = summary[f"{metric}_mean"].to_numpy()
        ci = 1.96 * summary[f"{metric}_se"].to_numpy()
        axes[1].plot(
            x,
            mean_values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.7 if metric == "outlier_excess_mse" else 2.1,
            label=label,
        )
        axes[1].fill_between(x, mean_values - ci, mean_values + ci, color=color, alpha=0.12, linewidth=0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Outlier covariate overlap")
    axes[1].set_ylabel("Excess MSE")
    axes[1].set_title("Task-specific prediction", loc="left", pad=8)
    axes[1].legend(frameon=False, loc="best")
    style_axis(axes[1])
    fig.tight_layout()
    save_figure(fig, FIG_ADD / "figure_additional_outlier")
    return summary


def write_theory_verification_table() -> pd.DataFrame:
    source = ROOT / "results" / "theory_verification" / "fixed_representation.csv"
    data = pd.read_csv(source)
    summary = (
        data.groupby("design")["relative_error"]
        .agg(couplings="count", median_relative_error="median", maximum_relative_error="max")
        .reset_index()
    )
    summary.to_csv(DATA_OUT / "theory_verification_summary.csv", index=False)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Second-moment design & Values of \(\lambda\) & Median relative error & Maximum relative error \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        design = {
            "near_singular": "Near-singular",
            "random_spd": "Random positive definite",
        }.get(str(row.design), str(row.design).replace("_", " ").title())
        lines.append(
            f"{design} & {int(row.couplings)} & {row.median_relative_error:.5f} & {row.maximum_relative_error:.5f} "
            + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Numerical verification of the fixed-representation risk identity.}",
        r"\label{tab:theory-verification}",
        r"\end{table}",
    ])
    (TABLE_ADD / "table_additional_theory_verification.tex").write_text("\n".join(lines) + "\n")
    return summary


def write_gtex_task_table() -> pd.DataFrame:
    source = ROOT / "experiments" / "gtex" / "results" / "repeated_cv_v1" / "per_repeat_tissue_metrics.csv"
    data = pd.read_csv(source)
    summary = (
        data.groupby(["response", "tissue", "method"])["standardized_mse"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "repeats", "mean": "mse_mean", "std": "mse_sd"})
    )
    summary.to_csv(DATA_OUT / "gtex_tissue_summary.csv", index=False)
    short = {
        "brain_amygdala": "Amygdala",
        "brain_anterior_cingulate_cortex_ba24": "Anterior cingulate cortex",
        "brain_caudate_basal_ganglia": "Caudate",
        "brain_cerebellar_hemisphere": "Cerebellar hemisphere",
        "brain_cerebellum": "Cerebellum",
        "brain_cortex": "Cortex",
        "brain_frontal_cortex_ba9": "Frontal cortex",
        "brain_hypothalamus": "Hypothalamus",
        "brain_nucleus_accumbens_basal_ganglia": "Nucleus accumbens",
        "brain_putamen_basal_ganglia": "Putamen",
        "brain_spinal_cord_cervical_c-1": "Spinal cord",
    }
    lines = [
        r"\begin{sidewaystable}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Tissue & Pool & STL & HPS & MMoE & ARMUL & FLARCC & COVER \\",
    ]
    for response_name in ["JAM2", "SH2D2A"]:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{response_name}}}}} \\")
        block = summary.loc[summary["response"].eq(response_name)]
        for tissue in sorted(block["tissue"].unique()):
            tissue_block = block.loc[block["tissue"].eq(tissue)].sort_values("mse_mean")
            ranks = tissue_block["method"].tolist()
            cells = []
            for method in METHODS:
                row = tissue_block.loc[tissue_block["method"].eq(method)].iloc[0]
                value = mean_sd(row["mse_mean"], row["mse_sd"])
                if method == ranks[0]:
                    value = rf"\textbf{{{value}}}"
                elif method == ranks[1]:
                    value = rf"\underline{{{value}}}"
                cells.append(value)
            lines.append(f"{latex_escape(short[tissue])} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Tissue-specific standardized mean squared errors in GTEx. Entries are means (standard deviations) over 20 repeated five-fold cross-validation partitions.}",
        r"\label{tab:gtex-tissue}",
        r"\end{sidewaystable}",
    ])
    (TABLE_ADD / "table_additional_gtex_tissues.tex").write_text("\n".join(lines) + "\n")
    return summary


def write_provenance() -> None:
    provenance = {
        "formal_input_directories": [
            "results/main",
            "results/controls",
            "results/mechanism",
            "results/scaling",
            "results/random_alignment",
            "results/sensitivity",
            "results/outlier",
            "results/theory_verification",
            "experiments/gtex/results/repeated_cv_v1",
        ],
        "excluded": [
            "exploratory and calibration simulations",
            "PDX experiments",
            "deprecated GTEx pilot, smoke, and E2E experiments",
        ],
        "primary_simulation_metric": "task-balanced excess MSE",
        "simulation_repetitions": 100,
        "gtex_protocol": "20 repeated five-fold cross-validation partitions",
        "method_order": METHODS,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def write_validation(
    heterogeneity: pd.DataFrame,
    mechanism: pd.DataFrame,
    task_scaling: pd.DataFrame,
    dimension_scaling: pd.DataFrame,
    random_alignment: pd.DataFrame,
    sensitivity: pd.DataFrame,
    outlier: pd.DataFrame,
    theory: pd.DataFrame,
    gtex_tissues: pd.DataFrame,
) -> None:
    checks: list[dict[str, object]] = []

    def record(name: str, condition: bool, observed: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "observed": observed})

    record(
        "heterogeneity regimes: 5 settings x 7 methods x 100 repetitions",
        len(heterogeneity) == 35 and heterogeneity["excess_mse_n"].eq(100).all(),
        {"rows": len(heterogeneity), "counts": sorted(heterogeneity["excess_mse_n"].unique().tolist())},
    )
    record(
        "mechanism: 10 settings x 3 methods x 100 repetitions",
        len(mechanism) == 30 and mechanism["excess_mse_n"].eq(100).all(),
        {"rows": len(mechanism), "counts": sorted(mechanism["excess_mse_n"].unique().tolist())},
    )
    record(
        "task scaling: 3 settings x 7 methods x 100 repetitions",
        len(task_scaling) == 21 and task_scaling["excess_mse_n"].eq(100).all(),
        {
            "rows": len(task_scaling),
            "settings": sorted(task_scaling["scaling_value"].unique().tolist()),
        },
    )
    record(
        "dimension scaling: 3 settings x 7 methods x 100 repetitions",
        len(dimension_scaling) == 21 and dimension_scaling["excess_mse_n"].eq(100).all(),
        {
            "rows": len(dimension_scaling),
            "settings": sorted(dimension_scaling["scaling_value"].unique().tolist()),
        },
    )
    record(
        "random alignment: 7 methods x 100 repetitions",
        len(random_alignment) == 7 and random_alignment["excess_mse_n"].eq(100).all(),
        {"rows": len(random_alignment), "counts": sorted(random_alignment["excess_mse_n"].unique().tolist())},
    )
    sensitivity_counts = sensitivity.groupby(["variant", "method"], observed=True).size()
    record(
        "sensitivity: 6 variants x 2 methods x 5 initializations",
        len(sensitivity_counts) == 12 and sensitivity_counts.eq(5).all(),
        {"groups": len(sensitivity_counts), "counts": sorted(sensitivity_counts.unique().tolist())},
    )
    record(
        "outlier diagnostic: 5 overlap settings x 100 repetitions",
        len(outlier) == 5 and outlier["outlier_excess_mse_n"].eq(100).all(),
        {"rows": len(outlier), "counts": sorted(outlier["outlier_excess_mse_n"].unique().tolist())},
    )
    record(
        "fixed-representation verification: 5 designs x 31 couplings",
        len(theory) == 5 and theory["couplings"].eq(31).all(),
        {"rows": len(theory), "couplings": sorted(theory["couplings"].unique().tolist())},
    )
    record(
        "GTEx tissue table: 2 responses x 11 tissues x 7 methods x 20 repeats",
        len(gtex_tissues) == 154 and gtex_tissues["repeats"].eq(20).all(),
        {"rows": len(gtex_tissues), "counts": sorted(gtex_tissues["repeats"].unique().tolist())},
    )
    report = {"complete": all(item["passed"] for item in checks), "checks": checks}
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        failures = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Paper-result validation failed: {failures}")


def main() -> None:
    prepare_output_dirs()
    heterogeneity = write_main_heterogeneity_table()
    mechanism = plot_main_mechanism()
    task_scaling, dimension_scaling = plot_main_scaling()
    write_main_gtex_table()
    plot_main_gtex_overlap()
    write_secondary_metrics_table(heterogeneity)
    random_alignment = write_random_alignment_table()
    plot_additional_mechanism(mechanism)
    sensitivity = plot_sensitivity()
    write_scaling_diagnostics_table(task_scaling, dimension_scaling)
    outlier = plot_outlier()
    theory = write_theory_verification_table()
    gtex_tissues = write_gtex_task_table()
    write_provenance()
    write_validation(
        heterogeneity,
        mechanism,
        task_scaling,
        dimension_scaling,
        random_alignment,
        sensitivity,
        outlier,
        theory,
        gtex_tissues,
    )
    print(f"Paper-facing artifacts written to {OUT}")


if __name__ == "__main__":
    main()
