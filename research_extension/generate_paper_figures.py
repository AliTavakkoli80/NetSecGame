"""
generate_paper_figures.py — Publication-Grade 300 DPI PNG & PDF Figure Generator.

Reads completed metrics and statistical/entropy reports from
`logs/benchmarks/final_paper_150ep/` and generates:
1. figures/fig1_markov_transitions_comparison.{png,pdf}
2. figures/fig2_ablation_win_detect_bars.{png,pdf}
3. figures/fig3_entropy_and_pareto.{png,pdf}
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
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

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
            episodes[int(row["episode_id"])].append((int(row["step_num"]), _normalize_state(row["action_type"])))

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


def plot_fig1_markov_transitions(suite_root: Path, out_dir: Path) -> None:
    """7x7 Action Transition Probability Heatmaps on Three-Networks + Monitor On."""
    regime_dir = suite_root / "regime2b_threenets_defender"
    panels = [
        ("Raw ReAct (7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct_steps.csv"),
        ("BeliefGraph NoStealth (7B-NF4)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_steps.csv"),
        ("BeliefGraph + Minimax Utility (7B-NF4)", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_steps.csv"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    short_labels = ["Start", "Scan", "FindSrv", "Exploit", "FindData", "Exfil", "Invalid"]

    im = None
    for ax, (title, csv_name) in zip(axes, panels):
        probs = compute_transition_matrix(regime_dir / csv_name)
        im = ax.imshow(probs, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels)
        ax.set_xlabel("Next Action")

        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                val = probs[i, j]
                if val >= 0.01:
                    color = "white" if val >= 0.55 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

    axes[0].set_ylabel("Current State / Action")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
        cbar.set_label("Transition Probability $P(a_{t+1} \\mid a_t)$")

    fig.suptitle("Action Transition Dynamics (Three-Networks Topology, Monitor Active, n=150)", fontweight="bold", y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_markov_transitions_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("summary", {})


def plot_fig2_ablation_bars(suite_root: Path, out_dir: Path) -> None:
    """Grouped bar charts with 95% CI error bars for Win % & Alert % with Monitor On."""
    regimes = [
        ("Full Topology (Monitor On)", suite_root / "regime1b_full_defender"),
        ("Three-Networks Topology (Monitor On)", suite_root / "regime2b_threenets_defender"),
    ]
    configs = [
        ("Random\n(n=2000)", "Baseline_RandomAttacker_metrics.json"),
        ("Heuristic\n(n=150)", "Baseline_PureHeuristic_metrics.json"),
        ("RawReAct\n7B-NF4", "Ablation_SLM_7B_NF4_RawReAct_metrics.json"),
        ("BeliefGraph\nNoStealth", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Proposed\n1.5B-BF16", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Proposed\n7B-NF4", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    x = np.arange(len(configs))
    width = 0.36

    for ax, (title, rdir) in zip(axes, regimes):
        wins, win_cis, dets, det_cis = [], [], [], []
        for _, fname in configs:
            s = _load_summary(rdir / fname)
            wins.append(float(s.get("win_rate_percent", 0.0)))
            win_cis.append(float(s.get("win_rate_ci95", 0.0)))
            dets.append(float(s.get("detection_rate_percent", 0.0)))
            det_cis.append(float(s.get("detection_rate_ci95", 0.0)))

        b1 = ax.bar(x - width / 2, wins, width, yerr=win_cis, capsize=3, label="Win Rate (%)", color="#2b6cb0", alpha=0.92)
        b2 = ax.bar(x + width / 2, dets, width, yerr=det_cis, capsize=3, label="Monitor Alert Rate (%)", color="#c53030", alpha=0.88)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in configs])
        ax.set_ylim(0, 112)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Percentage (%) with 95% CI")
    fig.suptitle("Stealth & Mission Completion Ablation Across Randomized Topologies", fontweight="bold", y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_ablation_win_detect_bars.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_entropy_and_pareto(suite_root: Path, out_dir: Path) -> None:
    """Two-panel figure: (a) Sliding-window Shannon Entropy H_t, (b) Hardware Pareto Frontier."""
    stat_path = suite_root / "statistical_analysis_report.json"
    stat_data = json.loads(stat_path.read_text(encoding="utf-8")) if stat_path.exists() else {}
    r2b_ent = stat_data.get("regimes", {}).get("regime2b_threenets_defender", {}).get("entropy_and_phases", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Panel (a): Shannon Entropy Trajectory H_t
    series = [
        ("RawReAct (7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct", "#718096", "--"),
        ("BeliefGraph NoStealth (7B-NF4)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth", "#d69e2e", "-."),
        ("Proposed 1.5B-BF16 (WithStealth)", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth", "#38a169", "-"),
        ("Proposed 7B-NF4 (WithStealth)", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth", "#2b6cb0", "-"),
    ]
    for label, key, color, ls in series:
        traj = r2b_ent.get(key, {}).get("entropy_trajectory", [])
        if not traj:
            continue
        steps = [pt["step"] for pt in traj if pt["n"] >= 3]
        means = [pt["mean_entropy"] for pt in traj if pt["n"] >= 3]
        cis = [pt["ci95_entropy"] for pt in traj if pt["n"] >= 3]
        if not steps:
            continue
        ax1.plot(steps, means, label=label, color=color, linestyle=ls, linewidth=1.8)
        lower = np.maximum(0.0, np.asarray(means) - np.asarray(cis))
        upper = np.asarray(means) + np.asarray(cis)
        ax1.fill_between(steps, lower, upper, color=color, alpha=0.12)

    ax1.set_title("(a) Sliding-Window Action Entropy $H_t$ (Three-Networks, Monitor On)", fontweight="bold")
    ax1.set_xlabel("Episode Step $t$")
    ax1.set_ylabel("Shannon Entropy $H_t$ (bits)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")

    # Panel (b): Hardware-Aware Pareto Frontier (Peak VRAM GB vs Win Rate %, bubble size/label = Latency ms)
    r2b_dir = suite_root / "regime2b_threenets_defender"
    s15 = _load_summary(r2b_dir / "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json")
    s7b = _load_summary(r2b_dir / "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json")

    vram_15 = float(s15.get("peak_vram_mb", 3100.0)) / 1024.0
    lat_15 = float(s15.get("avg_latency_ms_per_step", 0.0))
    win_15 = float(s15.get("win_rate_percent", 0.0))

    vram_7b = float(s7b.get("peak_vram_mb", 5800.0)) / 1024.0
    lat_7b = float(s7b.get("avg_latency_ms_per_step", 0.0))
    win_7b = float(s7b.get("win_rate_percent", 0.0))

    # Published reference point from Rigaki et al. (ESWA 2026) Table 5: Hackphyr-7B FP16 (14 GB VRAM, 42.3% Three-Nets Defender win rate)
    points = [
        ("Proposed 1.5B-BF16\n(Ours)", vram_15, win_15, f"{lat_15:.0f} ms/step", "#38a169", "o"),
        ("Proposed 7B-NF4\n(Ours)", vram_7b, win_7b, f"{lat_7b:.0f} ms/step", "#2b6cb0", "s"),
        ("Hackphyr 7B-FP16\n(ESWA 2026 Ref)", 14.0, 42.3, "~14.0 GB FP16", "#c53030", "^"),
    ]
    for lbl, vx, wy, note, col, marker in points:
        ax2.scatter([vx], [wy], s=180, color=col, marker=marker, zorder=5, edgecolors="black", linewidth=0.8)
        ax2.annotate(
            f"{lbl}\n({vx:.1f} GB, {wy:.1f}%)\n[{note}]",
            xy=(vx, wy),
            xytext=(8, -12 if vx < 10 else 8),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.85),
        )

    ax2.set_title("(b) Hardware-Aware Pareto Frontier (VRAM vs. Win Rate)", fontweight="bold")
    ax2.set_xlabel("Peak GPU VRAM Footprint (GB)")
    ax2.set_ylabel("Win Rate (%) — Three-Networks + Monitor On")
    ax2.set_xlim(0, 16.5)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_entropy_and_pareto.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 300 DPI publication figures.")
    parser.add_argument("--suite-root", type=str, default="logs/benchmarks/final_paper_150ep")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_markov_transitions(suite_root, out_dir)
    plot_fig2_ablation_bars(suite_root, out_dir)
    plot_fig3_entropy_and_pareto(suite_root, out_dir)
    print(f"Generated 300 DPI PNG & PDF figures in {out_dir}")


if __name__ == "__main__":
    main()
