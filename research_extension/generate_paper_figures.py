"""
generate_paper_figures.py — Publication-Grade Elsevier ESWA Empirical Figures.

Reads completed metrics and statistical/entropy reports from
`logs/benchmarks/final_paper_150ep/` and generates exact two-column (7.0 in / 178 mm)
300 DPI PNG & PDF figures in `figures/`:
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

# Unified Elsevier ESWA Q1 publication configuration (exact 7.0 in two-column scale)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#94A3B8",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# Cohesive colorblind-safe & grayscale-friendly Q1 editorial palette
TEAL = "#1B6B68"
SLATE_BLUE = "#2B5C8F"
CRIMSON = "#B83A3A"
AMBER = "#D97706"
CHARCOAL = "#334155"
NEUTRAL_LIGHT = "#CBD5E1"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"
SUBTLE_BORDER = "#CBD5E1"
BG_WHITE = "#FFFFFF"

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
    Three-panel empirical figure at exact 7.0 in width:
    (a) Full Topology (Monitor On) Win/Alert across all 7 configs
    (b) Three-Networks (Monitor On) Win/Alert across all 7 configs
    (c) SLM Autonomy Breakdown (First-Pass Valid, +1-Step Reflexion, Minimax Override)
    """
    regimes = [
        ("(a) Full Topology (Monitor On)", suite_root / "regime1b_full_defender"),
        ("(b) Three-Networks (Monitor On)", suite_root / "regime2b_threenets_defender"),
    ]
    configs = [
        ("Random", "Baseline_RandomAttacker_metrics.json"),
        ("Heur\n(NoStl)", "Baseline_PureHeuristic_metrics.json"),
        ("Heur\n(+Stl)", "Ablation_PureHeuristic_WithStealth_metrics.json"),
        ("Raw\nReAct", "Ablation_SLM_7B_NF4_RawReAct_metrics.json"),
        ("BG\n(NoStl)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Prop\n1.5B", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Prop\n7B", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
    ]

    fig = plt.figure(figsize=(6.78, 3.30))
    gs_ab = fig.add_gridspec(1, 2, left=0.068, right=0.702, top=0.88, bottom=0.28, wspace=0.07)
    gs_c = fig.add_gridspec(1, 1, left=0.790, right=0.988, top=0.88, bottom=0.28)
    ax_a = fig.add_subplot(gs_ab[0, 0])
    ax_b = fig.add_subplot(gs_ab[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs_c[0, 0])
    axes_ab = [ax_a, ax_b]

    def _fmt_pct(val: float) -> str:
        if 0.0 < val < 1.0:
            return f"{val:.2f}".rstrip("0").rstrip(".")
        return f"{val:.1f}"

    x = np.arange(len(configs))
    width = 0.35

    for idx_r, (title, rdir) in enumerate(regimes):
        ax = axes_ab[idx_r]
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
            capsize=2.0,
            error_kw=dict(ecolor=CHARCOAL, lw=0.85, capthick=0.85),
            label="Win Rate (%)",
            color=TEAL,
            edgecolor=BG_WHITE,
            linewidth=0.4,
            alpha=0.95,
            zorder=2.5,
        )
        b2 = ax.bar(
            x + width / 2,
            dets,
            width,
            yerr=det_cis,
            capsize=2.0,
            error_kw=dict(ecolor=CHARCOAL, lw=0.85, capthick=0.85),
            label="Alert Rate (%)",
            color=CRIMSON,
            edgecolor=BG_WHITE,
            linewidth=0.4,
            alpha=0.92,
            zorder=2.5,
        )

        # Place bar labels strictly above the upper 95% CI cap (val + ci + 2.5)
        labels_info: List[Dict[str, Any]] = []
        for idx_bar, (bar1, ci1, bar2, ci2) in enumerate(zip(b1, win_cis, b2, det_cis)):
            h1, h2 = bar1.get_height(), bar2.get_height()
            x1 = bar1.get_x() + bar1.get_width() / 2
            x2 = bar2.get_x() + bar2.get_width() / 2
            # Keep tiny bars' labels away from tall partner bars and from the left y-axis
            if h2 - h1 > 20.0:
                x1 -= 0.03 if idx_bar == 0 else 0.06
            elif h1 - h2 > 20.0:
                x2 += 0.06
            labels_info.append({"x": x1, "y": h1 + ci1 + 2.5, "text": _fmt_pct(h1), "color": TEAL})
            labels_info.append({"x": x2, "y": h2 + ci2 + 2.5, "text": _fmt_pct(h2), "color": CRIMSON})

        # General pairwise anti-collision pass (both intra-group and cross-group neighbors)
        for idx_l in range(len(labels_info) - 1):
            cur = labels_info[idx_l]
            nxt = labels_info[idx_l + 1]
            if abs(nxt["x"] - cur["x"]) < 0.72 and abs(nxt["y"] - cur["y"]) < 8.5:
                if nxt["y"] >= cur["y"]:
                    nxt["y"] = cur["y"] + 8.5
                else:
                    cur["y"] = nxt["y"] + 8.5

        for lbl in labels_info:
            ax.text(
                lbl["x"],
                lbl["y"],
                lbl["text"],
                ha="center",
                va="bottom",
                fontsize=6.0,
                color=lbl["color"],
                fontweight="bold",
                zorder=3.5,
            )

        ax.set_title(title, fontweight="bold", fontsize=8.4, pad=7, color=TEXT_DARK)
        ax.set_xlim(-0.62, len(configs) - 1 + 0.58)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in configs], fontsize=5.6, color=TEXT_DARK)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=7.8)
        ax.grid(axis="y", linestyle="--", color="#94A3B8", alpha=0.25, zorder=0.5)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        if idx_r == 0:
            ax.set_ylabel("Percentage (%) with 95% CI", fontsize=8.4, color=TEXT_DARK)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    # Shared horizontal legend centered cleanly beneath panels (a)-(b)
    handles_ab, labels_ab = ax_a.get_legend_handles_labels()
    fig.legend(
        handles_ab,
        labels_ab,
        loc="upper center",
        bbox_to_anchor=(0.396, 0.13),
        ncol=2,
        fontsize=7.5,
        handlelength=1.4,
        columnspacing=1.2,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=SUBTLE_BORDER,
        framealpha=0.98,
    )

    # Panel (c): Stacked Autonomy Breakdown on Three-Networks + Monitor On
    r2b_dir = suite_root / "regime2b_threenets_defender"
    aut_configs = [
        ("BG NoStl\n(7B)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Proposed\n1.5B", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Proposed\n7B", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
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

    ax_c.bar(xc, fp_arr, 0.68, label="First-Pass", color=TEAL, edgecolor=BG_WHITE, linewidth=0.5, alpha=0.95, zorder=2.5)
    ax_c.bar(xc, rg_arr, 0.68, bottom=fp_arr, label="+Reflexion", color=SLATE_BLUE, edgecolor=BG_WHITE, linewidth=0.5, alpha=0.92, zorder=2.5)
    ax_c.bar(xc, ov_arr, 0.68, bottom=fp_arr + rg_arr, label="Override", color=NEUTRAL_LIGHT, edgecolor=CHARCOAL, linewidth=0.5, alpha=0.90, zorder=2.5)

    for i in range(len(aut_configs)):
        ax_c.text(xc[i], fp_arr[i] / 2.0, f"{fp_arr[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=5.8, zorder=3.5)
        if rg_arr[i] >= 6.0:
            ax_c.text(xc[i], fp_arr[i] + rg_arr[i] / 2.0, f"+{rg_arr[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=5.7, zorder=3.5)
        if ov_arr[i] >= 8.0:
            ax_c.text(xc[i], fp_arr[i] + rg_arr[i] + ov_arr[i] / 2.0, f"{ov_arr[i]:.1f}%", ha="center", va="center", color=CHARCOAL, fontweight="bold", fontsize=5.7, zorder=3.5)

    ax_c.set_title("(c) SLM Autonomy Breakdown", fontweight="bold", fontsize=8.3, pad=7, color=TEXT_DARK)
    ax_c.set_xlim(-0.54, len(aut_configs) - 1 + 0.54)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([c[0] for c in aut_configs], fontsize=6.0, color=TEXT_DARK)
    ax_c.set_ylim(0, 118)
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.tick_params(axis="y", labelsize=7.8)
    ax_c.set_ylabel("Share of Executed Steps (%)", fontsize=8.2, color=TEXT_DARK, labelpad=2)
    ax_c.grid(axis="y", linestyle="--", color="#94A3B8", alpha=0.25, zorder=0.5)
    ax_c.grid(axis="x", visible=False)
    ax_c.set_axisbelow(True)

    handles_c, labels_c = ax_c.get_legend_handles_labels()
    fig.legend(
        handles_c,
        labels_c,
        loc="upper center",
        bbox_to_anchor=(0.885, 0.13),
        ncol=3,
        fontsize=6.5,
        handlelength=0.95,
        handletextpad=0.30,
        columnspacing=0.55,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=SUBTLE_BORDER,
        framealpha=0.98,
    )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_ablation_and_autonomy.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_fig4_markov_transitions(suite_root: Path, out_dir: Path) -> None:
    """7x7 Action Transition Probability Heatmaps on Three-Networks + Monitor On (7.0 in width)."""
    regime_dir = suite_root / "regime2b_threenets_defender"
    panels = [
        ("(a) Raw ReAct\n(7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct_steps.csv"),
        ("(b) BeliefGraph NoStealth\n(7B-NF4)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_steps.csv"),
        ("(c) BeliefGraph + Minimax\nUtility (7B-NF4)", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_steps.csv"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.90, 2.80), sharey=True)
    short_labels = ["Start", "Scan", "FindSrv", "Exploit", "FindData", "Exfil", "Invalid"]

    im = None
    for idx_p, (ax, (title, csv_name)) in enumerate(zip(axes, panels)):
        probs = compute_transition_matrix(regime_dir / csv_name)
        ax.grid(False)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color(CHARCOAL)
            ax.spines[sp].set_linewidth(0.85)

        im = ax.imshow(probs, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="equal")
        ax.set_title(title, fontweight="bold", fontsize=8.8, pad=6, color=TEXT_DARK)
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=7.5, color=TEXT_DARK)
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=7.6, color=TEXT_DARK)
        ax.set_xlabel("Next Action", fontsize=8.5, color=TEXT_DARK, labelpad=4)

        # Subtle white grid lines between heatmap cells for crisp reading
        ax.set_xticks(np.arange(-0.5, len(short_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(short_labels), 1), minor=True)
        ax.grid(which="minor", color=BG_WHITE, linestyle="-", linewidth=0.5, alpha=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)

        if idx_p > 0:
            ax.tick_params(axis="y", left=False)

        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                val = probs[i, j]
                if round(val, 2) >= 0.01:
                    is_high = val >= 0.55
                    txt_color = "#FFFFFF" if is_high else "#1E293B"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontweight="bold" if is_high else "normal",
                        fontsize=6.7,
                    )

    axes[0].set_ylabel("Current Action / State", fontsize=8.5, color=TEXT_DARK, labelpad=4)
    fig.subplots_adjust(left=0.095, right=0.885, top=0.84, bottom=0.22, wspace=0.10)

    # Align colorbar precisely to rendered heatmap box height
    fig.canvas.draw()
    pos_right = axes[-1].get_position()
    if im is not None:
        cbar_ax = fig.add_axes([0.902, pos_right.y0, 0.016, pos_right.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.outline.set_edgecolor(CHARCOAL)
        cbar.outline.set_linewidth(0.85)
        cbar.ax.tick_params(labelsize=7.8)
        cbar.set_label("Transition Probability $P(a_{t+1} \\mid a_t)$", fontsize=8.3, labelpad=6, color=TEXT_DARK)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_markov_transitions_comparison.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_fig5_entropy_and_pareto(suite_root: Path, out_dir: Path) -> None:
    """Two-panel figure at exact 7.0 in width: (a) Entropy H_t, (b) Hardware Pareto Frontier."""
    stat_path = suite_root / "statistical_analysis_report.json"
    stat_data = json.loads(stat_path.read_text(encoding="utf-8")) if stat_path.exists() else {}
    r2b_ent = stat_data.get("regimes", {}).get("regime2b_threenets_defender", {}).get("entropy_and_phases", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.25))

    series = [
        ("RawReAct (7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct", CHARCOAL, ":", "D"),
        ("BeliefGraph NoStealth", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth", CRIMSON, "-.", "^"),
        ("Proposed 1.5B-BF16", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth", SLATE_BLUE, "--", "o"),
        ("Proposed 7B-NF4", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth", TEAL, "-", "s"),
    ]
    for label, key, color, ls, marker in series:
        traj = r2b_ent.get(key, {}).get("entropy_trajectory", [])
        if not traj:
            continue
        steps = [pt["step"] for pt in traj if pt["n"] >= 3 and pt["step"] <= 60]
        means = [pt["mean_entropy"] for pt in traj if pt["n"] >= 3 and pt["step"] <= 60]
        cis = [pt["ci95_entropy"] for pt in traj if pt["n"] >= 3 and pt["step"] <= 60]
        if not steps:
            continue
        ax1.plot(
            steps,
            means,
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            markevery=9,
            markersize=3.4,
            linewidth=1.55,
            zorder=3.0,
        )
        lower = np.maximum(0.0, np.asarray(means) - np.asarray(cis))
        upper = np.asarray(means) + np.asarray(cis)
        ax1.fill_between(steps, lower, upper, color=color, alpha=0.11, zorder=2.0)

    ax1.set_title("(a) Sliding-Window Action Entropy $H_t$\n(Three-Networks, Monitor On)", fontweight="bold", fontsize=9.0, pad=6, color=TEXT_DARK)
    ax1.set_xlabel("Episode Step $t$", fontsize=8.5, color=TEXT_DARK)
    ax1.set_ylabel("Shannon Entropy $H_t$ (bits)", fontsize=8.5, color=TEXT_DARK)
    ax1.set_xlim(1, 60)
    ax1.set_ylim(-0.08, 2.58)
    ax1.tick_params(labelsize=8.0)
    ax1.grid(True, linestyle="--", color="#94A3B8", alpha=0.25, zorder=0.5)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=6.6,
        columnspacing=0.65,
        handlelength=1.6,
        handletextpad=0.35,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=SUBTLE_BORDER,
        framealpha=0.92,
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
        color=TEAL,
        linestyle="-",
        linewidth=1.8,
        alpha=0.90,
        zorder=3.0,
    )
    ax2.fill_between(frontier_x, 0.0, frontier_y, color=TEAL, alpha=0.07, zorder=1.0)

    # Cloud GPT-4 reference line (ESWA 2026: 36.36% -> 36.4%)
    gpt4_win = 36.36
    ax2.axhline(gpt4_win, color=CHARCOAL, linestyle="--", linewidth=1.15, alpha=0.85, zorder=2.2)
    ax2.text(
        16.1,
        gpt4_win + 1.4,
        "GPT-4 (ESWA 2026): 36.4% (Cloud API)",
        ha="right",
        va="bottom",
        fontsize=6.9,
        color=CHARCOAL,
        fontweight="bold",
        zorder=4.0,
    )

    ax2.text(
        16.1,
        win_7b - 1.8,
        "Pareto-Optimal Frontier",
        ha="right",
        va="top",
        fontsize=6.9,
        color=TEAL,
        fontstyle="italic",
        zorder=4.0,
    )

    # Scatter markers & non-overlapping framed callout boxes inside (0..16.5, 0..80)
    ax2.scatter([vram_15], [win_15], s=68, color=SLATE_BLUE, marker="o", zorder=5.0, edgecolors=TEXT_DARK, linewidth=0.9)
    ax2.annotate(
        f"Proposed 1.5B-BF16 (Ours)\n{vram_15:.2f} GB, {win_15:.1f}%\n({lat_15:.0f} ms/step)",
        xy=(vram_15, win_15),
        xytext=(0.35, 64.0),
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=SLATE_BLUE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc=BG_WHITE, ec=SLATE_BLUE, lw=0.75, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=SLATE_BLUE, lw=0.9, shrinkA=1, shrinkB=5),
        zorder=5.5,
    )

    ax2.scatter([vram_7b], [win_7b], s=68, color=TEAL, marker="s", zorder=5.0, edgecolors=TEXT_DARK, linewidth=0.9)
    ax2.annotate(
        f"Proposed 7B-NF4 (Ours)\n{vram_7b:.2f} GB, {win_7b:.1f}%\n({lat_7b:.0f} ms/step)",
        xy=(vram_7b, win_7b),
        xytext=(16.15, 64.0),
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=TEAL,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc=BG_WHITE, ec=TEAL, lw=0.75, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.9, shrinkA=1, shrinkB=5),
        zorder=5.5,
    )

    ax2.scatter([14.0], [23.33], s=74, color=CRIMSON, marker="^", zorder=5.0, edgecolors=TEXT_DARK, linewidth=0.9)
    ax2.annotate(
        "Hackphyr 7B-FP16 (ESWA 2026)\n14.00 GB, 23.3% (Dominated)",
        xy=(14.0, 23.33),
        xytext=(10.8, 7.0),
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=CRIMSON,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=BG_WHITE, ec=CRIMSON, lw=0.75, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=0.9, shrinkA=1, shrinkB=5),
        zorder=5.5,
    )

    ax2.set_title("(b) Hardware-Aware Pareto Frontier\n(VRAM vs. Win Rate)", fontweight="bold", fontsize=9.0, pad=6, color=TEXT_DARK)
    ax2.set_xlabel("Peak GPU VRAM Footprint (GB)", fontsize=8.5, color=TEXT_DARK)
    ax2.set_ylabel("Win Rate (%) — Three-Nets + Mon.", fontsize=8.3, color=TEXT_DARK)
    ax2.set_xlim(0.0, 16.5)
    ax2.set_ylim(0.0, 80.0)
    ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax2.tick_params(labelsize=8.0)
    ax2.grid(True, linestyle="--", color="#94A3B8", alpha=0.25, zorder=0.5)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.15, wspace=0.28)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_entropy_and_pareto.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
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
