"""
generate_paper_figures.py — Publication-Grade 300 DPI PNG & PDF Empirical Figures.

Reads completed metrics and statistical/entropy reports from
`logs/benchmarks/final_paper_150ep/` and generates:
1. figures/fig3_ablation_and_autonomy.{png,pdf}
2. figures/fig4_markov_transitions_comparison.{png,pdf}
3. figures/fig5_entropy_and_pareto.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#334155",
    "axes.grid": True,
    "grid.color": "#CBD5E1",
    "grid.alpha": 0.70,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# Shared editorial IEEE/Elsevier palette
TEAL_ACCENT = "#0F766E"
BLUE_ACCENT = "#1D4ED8"
SLATE_ACCENT = "#475569"
CRIMSON_ACCENT = "#B91C1C"
BRICK_ACCENT = "#B45309"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"
SUBTLE_BORDER = "#CBD5E1"

MARKOV_STATES = [
    "Start",
    "ScanNetwork",
    "FindServices",
    "ExploitService",
    "FindData",
    "ExfiltrateData",
    "Invalid",
]
STATE_TO_IDX = {s: i for i, s in enumerate(MARKOV_STATES)}


def _normalize_state(action_type: str) -> str:
    s = (action_type or "").strip()
    return s if s in STATE_TO_IDX else "Invalid"


def compute_transition_matrix(steps_csv: Path) -> np.ndarray:
    """Compute 7x7 row-normalized Markov transition matrix from a steps CSV."""
    mat = np.zeros((len(MARKOV_STATES), len(MARKOV_STATES)), dtype=float)
    if not steps_csv.exists():
        return mat

    episodes: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    with open(steps_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            episodes[int(row["episode_id"])].append(
                (int(row["step_num"]), _normalize_state(row["action_type"]))
            )

    for seq in episodes.values():
        seq.sort(key=lambda x: x[0])
        prev_idx = STATE_TO_IDX["Start"]
        for _, st in seq:
            cur_idx = STATE_TO_IDX[st]
            mat[prev_idx, cur_idx] += 1.0
            prev_idx = cur_idx

    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(row_sums > 0, mat / row_sums, 0.0)
    return probs


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("summary", {})


def plot_fig3_ablation_and_autonomy(suite_root: Path, out_dir: Path) -> None:
    """
    Three-panel empirical figure:
    (a) Full + Monitor On Win/Alert across all 7 configs
    (b) Three-Networks + Monitor On Win/Alert across all 7 configs
    (c) SLM Decision Autonomy Breakdown (FirstPass, +Reflexion Gain, Residual Override)
    """
    regimes = [
        ("(a) Full Topology (Monitor On)", suite_root / "regime1b_full_defender"),
        ("(b) Three-Networks (Monitor On)", suite_root / "regime2b_threenets_defender"),
    ]
    configs = [
        ("Rand", "Baseline_RandomAttacker_metrics.json"),
        ("Heur\n-Stl", "Baseline_PureHeuristic_metrics.json"),
        ("Heur\n+Stl", "Ablation_PureHeuristic_WithStealth_metrics.json"),
        ("Raw\n7B", "Ablation_SLM_7B_NF4_RawReAct_metrics.json"),
        ("Bel\n-Stl", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Prop\n1.5B", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Prop\n7B", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
    ]

    fig = plt.figure(figsize=(10.5, 3.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.42, 0.92], wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    def _fmt_pct(val: float) -> str:
        if 0.0 < val < 1.0:
            return f"{val:.2f}".rstrip("0").rstrip(".")
        return f"{val:.1f}"

    x = np.arange(len(configs))
    width = 0.36

    for idx_r, (title, rdir) in enumerate(regimes):
        ax = axes[idx_r]
        wins, win_cis, dets, det_cis = [], [], [], []
        for _, fname in configs:
            s = _load_summary(rdir / fname)
            wins.append(float(s.get("win_rate_percent", 0.0)))
            win_cis.append(float(s.get("win_rate_ci95", 0.0)))
            dets.append(float(s.get("detection_rate_percent", 0.0)))
            det_cis.append(float(s.get("detection_rate_ci95", 0.0)))

        b1 = ax.bar(
            x - width / 2,
            wins,
            width,
            yerr=win_cis,
            capsize=2.2,
            error_kw=dict(ecolor=TEXT_DARK, lw=0.90),
            label="Win Rate (%)",
            color=TEAL_ACCENT,
            alpha=0.93,
            zorder=2.5,
        )
        b2 = ax.bar(
            x + width / 2,
            dets,
            width,
            yerr=det_cis,
            capsize=2.2,
            error_kw=dict(ecolor=TEXT_DARK, lw=0.90),
            label="Alert Rate (%)",
            color=CRIMSON_ACCENT,
            alpha=0.88,
            zorder=2.5,
        )

        # Place bar value labels strictly above the upper 95% CI error cap with anti-collision nudge
        for i_bar, (bar1, ci1, bar2, ci2) in enumerate(zip(b1, win_cis, b2, det_cis)):
            h1, h2 = bar1.get_height(), bar2.get_height()
            top1, top2 = h1 + ci1 + 2.2, h2 + ci2 + 2.2
            close_pair = abs(top1 - top2) < 9.0
            dx1 = -0.05 if close_pair else 0.0
            dx2 = +0.05 if close_pair else 0.0
            ax.text(
                bar1.get_x() + bar1.get_width() / 2 + dx1,
                top1,
                _fmt_pct(h1),
                ha="right" if close_pair else "center",
                va="bottom",
                fontsize=6.6,
                color=TEAL_ACCENT,
                fontweight="bold",
                zorder=3.5,
            )
            ax.text(
                bar2.get_x() + bar2.get_width() / 2 + dx2,
                top2,
                _fmt_pct(h2),
                ha="left" if close_pair else "center",
                va="bottom",
                fontsize=6.6,
                color=CRIMSON_ACCENT,
                fontweight="bold",
                zorder=3.5,
            )

        ax.set_title(title, fontweight="bold", fontsize=9.2, pad=8, color=TEXT_DARK)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in configs], fontsize=7.6, color=TEXT_DARK)
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=7.8)
        ax.grid(axis="y", linestyle="--", color=SUBTLE_BORDER, alpha=0.70, zorder=0.5)
        ax.set_axisbelow(True)
        if idx_r == 0:
            ax.set_ylabel("Percentage (%) with 95% CI", fontsize=8.8, color=TEXT_DARK)
        ax.legend(
            loc="upper right",
            fontsize=7.3,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor=SUBTLE_BORDER,
            framealpha=0.96,
        )

    # Panel (c): Stacked Autonomy Breakdown on Three-Networks + Monitor On
    ax_c = axes[2]
    r2b_dir = suite_root / "regime2b_threenets_defender"
    aut_configs = [
        ("Bel\n-Stl (7B)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Prop\n1.5B", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Prop\n7B", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
    ]
    fp_vals, ref_gains, ovr_vals = [], [], []
    for _, fname in aut_configs:
        s = _load_summary(r2b_dir / fname)
        fp = float(s.get("first_pass_valid_rate_percent", 0.0))
        pr = float(s.get("post_reflexion_valid_rate_percent", 0.0))
        ovr = float(s.get("wrapper_override_rate_percent", 0.0))
        fp_vals.append(fp)
        ref_gains.append(max(0.0, pr - fp))
        ovr_vals.append(ovr)

    xc = np.arange(len(aut_configs))
    fp_arr = np.asarray(fp_vals)
    rg_arr = np.asarray(ref_gains)
    ov_arr = np.asarray(ovr_vals)

    ax_c.bar(xc, fp_arr, 0.50, label="First-Pass Valid (%)", color=TEAL_ACCENT, alpha=0.93, zorder=2.5)
    ax_c.bar(xc, rg_arr, 0.50, bottom=fp_arr, label="+1-Step Reflexion (%)", color=BLUE_ACCENT, alpha=0.88, zorder=2.5)
    ax_c.bar(xc, ov_arr, 0.50, bottom=fp_arr + rg_arr, label="Minimax Override (%)", color="#94A3B8", alpha=0.88, zorder=2.5)

    for i in range(len(aut_configs)):
        ax_c.text(xc[i], fp_arr[i] / 2.0, f"{fp_arr[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=7.6, zorder=3.5)
        if rg_arr[i] >= 6.0:
            ax_c.text(xc[i], fp_arr[i] + rg_arr[i] / 2.0, f"+{rg_arr[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=7.4, zorder=3.5)
        if ov_arr[i] >= 8.0:
            ax_c.text(xc[i], fp_arr[i] + rg_arr[i] + ov_arr[i] / 2.0, f"{ov_arr[i]:.1f}%", ha="center", va="center", color=TEXT_DARK, fontweight="bold", fontsize=7.4, zorder=3.5)

    ax_c.set_title("(c) SLM Autonomy Breakdown", fontweight="bold", fontsize=9.2, pad=8, color=TEXT_DARK)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([c[0] for c in aut_configs], fontsize=7.8, color=TEXT_DARK)
    ax_c.set_ylim(0, 134)
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.tick_params(axis="y", labelsize=7.8)
    ax_c.set_ylabel("Share of Executed Steps (%)", fontsize=8.8, color=TEXT_DARK)
    ax_c.grid(axis="y", linestyle="--", color=SUBTLE_BORDER, alpha=0.70, zorder=0.5)
    ax_c.set_axisbelow(True)
    ax_c.legend(
        loc="upper right",
        fontsize=7.0,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=SUBTLE_BORDER,
        framealpha=0.96,
    )

    fig.subplots_adjust(left=0.055, right=0.99, top=0.89, bottom=0.19)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_ablation_and_autonomy.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_fig4_markov_transitions(suite_root: Path, out_dir: Path) -> None:
    """7x7 Action Transition Probability Heatmaps on Three-Networks + Monitor On."""
    regime_dir = suite_root / "regime2b_threenets_defender"
    panels = [
        ("(a) Raw ReAct (7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct_steps.csv"),
        ("(b) BeliefGraph NoStealth (7B-NF4)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_steps.csv"),
        ("(c) BeliefGraph + Minimax Utility (7B-NF4)", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_steps.csv"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.60), sharey=True)
    short_labels = ["Start", "Scan", "FindSrv", "Exploit", "FindData", "Exfil", "Invalid"]

    im = None
    for idx_p, (ax, (title, csv_name)) in enumerate(zip(axes, panels)):
        probs = compute_transition_matrix(regime_dir / csv_name)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        im = ax.imshow(probs, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        ax.set_title(title, fontweight="bold", fontsize=9.0, pad=8, color=TEXT_DARK)
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7.6)
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=7.6)
        ax.set_xlabel("Next Action", fontsize=8.6, color=TEXT_DARK)
        if idx_p > 0:
            ax.tick_params(axis="y", left=False)

        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                val = probs[i, j]
                if val >= 0.01:
                    color = "white" if val >= 0.55 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.2)

    axes[0].set_ylabel("Current State / Action", fontsize=8.6, color=TEXT_DARK)
    fig.subplots_adjust(left=0.075, right=0.895, top=0.89, bottom=0.22, wspace=0.12)
    if im is not None:
        cbar_ax = fig.add_axes([0.912, 0.22, 0.015, 0.67])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=7.6)
        cbar.set_label("Transition Probability $P(a_{t+1} \\mid a_t)$", fontsize=8.5, labelpad=6)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_markov_transitions_comparison.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_fig5_entropy_and_pareto(suite_root: Path, out_dir: Path) -> None:
    """Two-panel figure: (a) Sliding-window Shannon Entropy H_t, (b) Hardware Pareto Frontier."""
    stat_path = suite_root / "statistical_analysis_report.json"
    stat_data = json.loads(stat_path.read_text(encoding="utf-8")) if stat_path.exists() else {}
    r2b_ent = stat_data.get("regimes", {}).get("regime2b_threenets_defender", {}).get("entropy_and_phases", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.85))

    series = [
        ("RawReAct (7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct", SLATE_ACCENT, "--"),
        ("BeliefGraph NoStealth", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth", CRIMSON_ACCENT, "-."),
        ("Proposed 1.5B-BF16", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth", BLUE_ACCENT, "-"),
        ("Proposed 7B-NF4", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth", TEAL_ACCENT, "-"),
    ]
    for label, key, color, ls in series:
        traj = r2b_ent.get(key, {}).get("entropy_trajectory", [])
        if not traj:
            continue
        steps = [pt["step"] for pt in traj if pt["n"] >= 3 and pt["step"] <= 60]
        means = [pt["mean_entropy"] for pt in traj if pt["n"] >= 3 and pt["step"] <= 60]
        cis = [pt["ci95_entropy"] for pt in traj if pt["n"] >= 3 and pt["step"] <= 60]
        if not steps:
            continue
        ax1.plot(steps, means, label=label, color=color, linestyle=ls, linewidth=1.7, zorder=3.0)
        lower = np.maximum(0.0, np.asarray(means) - np.asarray(cis))
        upper = np.asarray(means) + np.asarray(cis)
        ax1.fill_between(steps, lower, upper, color=color, alpha=0.11, zorder=2.0)

    ax1.set_title("(a) Sliding-Window Action Entropy $H_t$", fontweight="bold", fontsize=9.2, color=TEXT_DARK)
    ax1.set_xlabel("Episode Step $t$", fontsize=8.6, color=TEXT_DARK)
    ax1.set_ylabel("Shannon Entropy $H_t$ (bits)", fontsize=8.6, color=TEXT_DARK)
    ax1.set_xlim(1, 60)
    ax1.set_ylim(-0.05, 2.48)
    ax1.tick_params(labelsize=7.8)
    ax1.grid(True, linestyle="--", color=SUBTLE_BORDER, alpha=0.70, zorder=0.5)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, 0.99),
        ncol=2,
        fontsize=6.9,
        columnspacing=0.9,
        handlelength=1.6,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=SUBTLE_BORDER,
        framealpha=0.96,
    )

    r2b_dir = suite_root / "regime2b_threenets_defender"
    s15 = _load_summary(r2b_dir / "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json")
    s7b = _load_summary(r2b_dir / "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json")

    vram_15 = float(s15.get("peak_vram_mb", 3304.53)) / 1024.0
    lat_15 = float(s15.get("avg_latency_ms_per_step", 0.0))
    win_15 = float(s15.get("win_rate_percent", 0.0))

    vram_7b = float(s7b.get("peak_vram_mb", 6070.25)) / 1024.0
    lat_7b = float(s7b.get("avg_latency_ms_per_step", 0.0))
    win_7b = float(s7b.get("win_rate_percent", 0.0))

    # Step-wise Pareto-Optimal Frontier (min VRAM, max Win Rate)
    frontier_x = [vram_15, vram_7b, vram_7b, 16.5]
    frontier_y = [win_15, win_15, win_7b, win_7b]
    ax2.plot(
        frontier_x,
        frontier_y,
        color=TEAL_ACCENT,
        linestyle="-",
        linewidth=1.7,
        alpha=0.85,
        zorder=3.0,
        label="Pareto-Optimal Frontier",
    )
    ax2.fill_between(frontier_x, 0.0, frontier_y, color=TEAL_ACCENT, alpha=0.06, zorder=1.0)

    # Cloud GPT-4 reference line (ESWA 2026: 36.36% -> 36.4%)
    gpt4_win = 36.36
    ax2.axhline(gpt4_win, color=SLATE_ACCENT, linestyle="--", linewidth=1.15, alpha=0.85, zorder=2.2)
    ax2.text(
        16.1,
        gpt4_win + 1.5,
        "GPT-4 (ESWA 2026): 36.4% (Cloud API)",
        ha="right",
        va="bottom",
        fontsize=7.3,
        color=SLATE_ACCENT,
        fontweight="bold",
        zorder=4.0,
    )

    # Scatter markers + short, non-crossing leader lines with generous axis margins
    ax2.scatter([vram_15], [win_15], s=110, color=BLUE_ACCENT, marker="o", zorder=5.0, edgecolors="#0F172A", linewidth=0.9)
    ax2.annotate(
        f"Proposed 1.5B-BF16 (Ours)\n{vram_15:.2f} GB, {win_15:.1f}% ({lat_15:.0f} ms/step)",
        xy=(vram_15, win_15),
        xytext=(0.40, 72.5),
        fontsize=7.0,
        color=BLUE_ACCENT,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=BLUE_ACCENT, lw=0.85, shrinkA=2, shrinkB=5),
        zorder=5.5,
    )

    ax2.scatter([vram_7b], [win_7b], s=110, color=TEAL_ACCENT, marker="s", zorder=5.0, edgecolors="#0F172A", linewidth=0.9)
    ax2.annotate(
        f"Proposed 7B-NF4 (Ours)\n{vram_7b:.2f} GB, {win_7b:.1f}% ({lat_7b:.0f} ms/step)",
        xy=(vram_7b, win_7b),
        xytext=(8.2, 72.5),
        fontsize=7.0,
        color=TEAL_ACCENT,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=TEAL_ACCENT, lw=0.85, shrinkA=2, shrinkB=5),
        zorder=5.5,
    )

    ax2.scatter([14.0], [23.33], s=115, color=CRIMSON_ACCENT, marker="^", zorder=5.0, edgecolors="#0F172A", linewidth=0.9)
    ax2.annotate(
        "Hackphyr 7B-FP16 (ESWA 2026)\n14.00 GB, 23.3% (Dominated)",
        xy=(14.0, 23.33),
        xytext=(7.8, 10.5),
        fontsize=7.3,
        color=CRIMSON_ACCENT,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=CRIMSON_ACCENT, lw=0.85, shrinkA=2, shrinkB=5),
        zorder=5.5,
    )

    ax2.text(
        16.1,
        win_7b + 1.5,
        "Pareto-Optimal Frontier",
        ha="right",
        va="bottom",
        fontsize=7.3,
        color=TEAL_ACCENT,
        fontstyle="italic",
        zorder=4.0,
    )

    ax2.set_title("(b) Hardware-Aware Pareto Frontier", fontweight="bold", fontsize=9.2, color=TEXT_DARK)
    ax2.set_xlabel("Peak GPU VRAM Footprint (GB)", fontsize=8.6, color=TEXT_DARK)
    ax2.set_ylabel("Win Rate (%) — Three-Networks + Mon. On", fontsize=8.4, color=TEXT_DARK)
    ax2.set_xlim(0.0, 16.5)
    ax2.set_ylim(0.0, 88.0)
    ax2.tick_params(labelsize=7.8)
    ax2.grid(True, linestyle="--", color=SUBTLE_BORDER, alpha=0.70, zorder=0.5)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.14, wspace=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_entropy_and_pareto.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 300 DPI publication empirical figures.")
    parser.add_argument("--suite-root", type=str, default="logs/benchmarks/final_paper_150ep")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fig3_ablation_and_autonomy(suite_root, out_dir)
    plot_fig4_markov_transitions(suite_root, out_dir)
    plot_fig5_entropy_and_pareto(suite_root, out_dir)
    print(f"Generated 300 DPI PNG & PDF empirical figures in {out_dir}")


if __name__ == "__main__":
    main()
