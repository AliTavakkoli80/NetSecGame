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
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
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
        ("Random\n(n=2000)", "Baseline_RandomAttacker_metrics.json"),
        ("Heur.\n(NoStl)", "Baseline_PureHeuristic_metrics.json"),
        ("Heur.\n(+Stl)", "Ablation_PureHeuristic_WithStealth_metrics.json"),
        ("RawReAct\n(7B)", "Ablation_SLM_7B_NF4_RawReAct_metrics.json"),
        ("Belief\n(NoStl)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Prop.\n1.5B", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Prop.\n7B-NF4", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
    ]

    fig = plt.figure(figsize=(16.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.95], wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    x = np.arange(len(configs))
    width = 0.38

    for idx_r, (title, rdir) in enumerate(regimes):
        ax = axes[idx_r]
        wins, win_cis, dets, det_cis = [], [], [], []
        for _, fname in configs:
            s = _load_summary(rdir / fname)
            wins.append(float(s.get("win_rate_percent", 0.0)))
            win_cis.append(float(s.get("win_rate_ci95", 0.0)))
            dets.append(float(s.get("detection_rate_percent", 0.0)))
            det_cis.append(float(s.get("detection_rate_ci95", 0.0)))

        b1 = ax.bar(x - width / 2, wins, width, yerr=win_cis, capsize=2.5, label="Win Rate (%)", color="#0072B2", alpha=0.92)
        b2 = ax.bar(x + width / 2, dets, width, yerr=det_cis, capsize=2.5, label="Alert Rate (%)", color="#D55E00", alpha=0.88)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.0)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.0)

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in configs], fontsize=7.5)
        ax.set_ylim(0, 114)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if idx_r == 0:
            ax.set_ylabel("Percentage (%) with 95% CI")
        ax.legend(loc="upper right", fontsize=7.8)

    # Panel (c): Stacked Autonomy Breakdown on Three-Networks + Monitor On
    ax_c = axes[2]
    r2b_dir = suite_root / "regime2b_threenets_defender"
    aut_configs = [
        ("BeliefGraph\nNoStealth (7B)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth_metrics.json"),
        ("Proposed\n1.5B-BF16", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json"),
        ("Proposed\n7B-NF4", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json"),
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

    ax_c.bar(xc, fp_arr, 0.52, label="First-Pass Valid (%)", color="#009E73", alpha=0.92)
    ax_c.bar(xc, rg_arr, 0.52, bottom=fp_arr, label="+1-Step Reflexion (%)", color="#56B4E9", alpha=0.92)
    ax_c.bar(xc, ov_arr, 0.52, bottom=fp_arr + rg_arr, label="Minimax Override (%)", color="#94A3B8", alpha=0.85)

    for i in range(len(aut_configs)):
        ax_c.text(xc[i], fp_arr[i] / 2.0, f"{fp_arr[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.0)
        if rg_arr[i] >= 6.0:
            ax_c.text(xc[i], fp_arr[i] + rg_arr[i] / 2.0, f"+{rg_arr[i]:.1f}%", ha="center", va="center", color="#0F172A", fontweight="bold", fontsize=7.8)
        if ov_arr[i] >= 8.0:
            ax_c.text(xc[i], fp_arr[i] + rg_arr[i] + ov_arr[i] / 2.0, f"{ov_arr[i]:.1f}%", ha="center", va="center", color="#0F172A", fontweight="bold", fontsize=7.8)

    ax_c.set_title("(c) SLM Decision Autonomy Breakdown", fontweight="bold", pad=8)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([c[0] for c in aut_configs], fontsize=8.0)
    ax_c.set_ylim(0, 114)
    ax_c.set_ylabel("Share of Executed Steps (%)")
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Empirical Ablation Across 7 Configurations & SLM Autonomy Breakdown (n=150 Paired Episodes)",
        fontweight="bold",
        y=1.02,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_ablation_and_autonomy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_markov_transitions(suite_root: Path, out_dir: Path) -> None:
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

    fig.suptitle(
        "Action Transition Dynamics (Three-Networks Topology, Monitor Active, n=150)",
        fontweight="bold",
        y=1.02,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_markov_transitions_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5_entropy_and_pareto(suite_root: Path, out_dir: Path) -> None:
    """Two-panel figure: (a) Sliding-window Shannon Entropy H_t, (b) Hardware Pareto Frontier."""
    stat_path = suite_root / "statistical_analysis_report.json"
    stat_data = json.loads(stat_path.read_text(encoding="utf-8")) if stat_path.exists() else {}
    r2b_ent = stat_data.get("regimes", {}).get("regime2b_threenets_defender", {}).get("entropy_and_phases", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    series = [
        ("RawReAct (7B-NF4)", "Ablation_SLM_7B_NF4_RawReAct", "#718096", "--"),
        ("BeliefGraph NoStealth (7B-NF4)", "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth", "#E69F00", "-."),
        ("Proposed 1.5B-BF16 (WithStealth)", "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth", "#009E73", "-"),
        ("Proposed 7B-NF4 (WithStealth)", "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth", "#0072B2", "-"),
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
        ax1.plot(steps, means, label=label, color=color, linestyle=ls, linewidth=1.8)
        lower = np.maximum(0.0, np.asarray(means) - np.asarray(cis))
        upper = np.asarray(means) + np.asarray(cis)
        ax1.fill_between(steps, lower, upper, color=color, alpha=0.12)

    ax1.set_title("(a) Sliding-Window Action Entropy $H_t$ (Three-Networks, Monitor On)", fontweight="bold")
    ax1.set_xlabel("Episode Step $t$")
    ax1.set_ylabel("Shannon Entropy $H_t$ (bits)")
    ax1.set_xlim(1, 60)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")

    r2b_dir = suite_root / "regime2b_threenets_defender"
    s15 = _load_summary(r2b_dir / "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth_metrics.json")
    s7b = _load_summary(r2b_dir / "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth_metrics.json")

    vram_15 = float(s15.get("peak_vram_mb", 3304.53)) / 1024.0
    lat_15 = float(s15.get("avg_latency_ms_per_step", 0.0))
    win_15 = float(s15.get("win_rate_percent", 0.0))

    vram_7b = float(s7b.get("peak_vram_mb", 6070.25)) / 1024.0
    lat_7b = float(s7b.get("avg_latency_ms_per_step", 0.0))
    win_7b = float(s7b.get("win_rate_percent", 0.0))

    points = [
        ("Proposed 1.5B-BF16\n(Ours)", vram_15, win_15, f"{lat_15:.0f} ms/step", "#009E73", "o", (10, -18)),
        ("Proposed 7B-NF4\n(Ours)", vram_7b, win_7b, f"{lat_7b:.0f} ms/step", "#0072B2", "s", (10, -14)),
        ("Hackphyr 7B-FP16\n(ESWA 2026 Ref)", 14.0, 23.3, "~14.0 GB FP16", "#D55E00", "^", (-115, 14)),
    ]
    for lbl, vx, wy, note, col, marker, xy_off in points:
        ax2.scatter([vx], [wy], s=180, color=col, marker=marker, zorder=5, edgecolors="black", linewidth=0.8)
        ax2.annotate(
            f"{lbl}\n({vx:.2f} GB, {wy:.1f}%)\n[{note}]",
            xy=(vx, wy),
            xytext=xy_off,
            textcoords="offset points",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.9),
        )

    ax2.set_title("(b) Hardware-Aware Pareto Frontier (VRAM vs. Win Rate)", fontweight="bold")
    ax2.set_xlabel("Peak GPU VRAM Footprint (GB)")
    ax2.set_ylabel("Win Rate (%) — Three-Networks + Monitor On")
    ax2.set_xlim(0, 16.5)
    ax2.set_ylim(0, 90)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_entropy_and_pareto.{ext}", dpi=300, bbox_inches="tight")
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
