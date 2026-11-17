"""
generate_architecture_diagrams.py — Vector Architecture & Mechanism Figures.

Generates publication-grade conceptual diagrams in `figures/` (300 DPI PNG + PDF):
1. fig1_system_architecture.{png,pdf}: End-to-end zero-shot cognitive pipeline.
2. fig2_belief_and_hazard_mechanism.{png,pdf}:
   (a) Epistemic host lifecycle state machine & MITRE ATT&CK transitions.
   (b) Multi-scale minimax hazard utility U(a) over w in {3, 5, 8}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "figure.dpi": 300,
})


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    fc: str,
    ec: str,
    title_color: str = "#111827",
    lw: float = 1.4,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#374151",
        zorder=3,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple,
    end: tuple,
    label: str = "",
    color: str = "#374151",
    rad: float = 0.0,
    lw: float = 1.5,
    label_offset: tuple = (0.0, 0.025),
) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6,head_width=3.5",
        color=color,
        linewidth=lw,
        zorder=4,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2.0 + label_offset[0]
        my = (start[1] + end[1]) / 2.0 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
            zorder=5,
        )


def plot_fig1_system_architecture(out_dir: Path) -> None:
    """Horizontal/modular end-to-end cognitive architecture diagram."""
    fig, ax = plt.subplots(figsize=(14.0, 5.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Outer cognitive wrapper container
    wrapper = FancyBboxPatch(
        (0.20, 0.22),
        0.61,
        0.68,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#F8FAFC",
        edgecolor="#94A3B8",
        linestyle="--",
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(wrapper)
    ax.text(
        0.505,
        0.865,
        "Training-Free Zero-Shot Cognitive & Stealth Wrapper (Zero Hardcoding)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1E293B",
    )

    # Left: NetSecGame Environment
    _draw_box(
        ax,
        0.02,
        0.42,
        0.145,
        0.26,
        "NetSecGame\nC-POMDP",
        "Randomized IPs\n& Goals (T=100)\nGlobalDefender (W*)",
        fc="#EFF6FF",
        ec="#0072B2",
        title_color="#005587",
    )

    # Module A: Topological Belief Graph
    _draw_box(
        ax,
        0.225,
        0.60,
        0.255,
        0.20,
        "Module A: Topological\nBelief Graph $\\mathcal{G}_t$",
        "Epistemic Nodes (4 States)\nExhausted Edge Pruning $\\mathcal{E}_{\\mathrm{exec}}$",
        fc="#ECFDF5",
        ec="#009E73",
        title_color="#065F46",
    )

    # Module B: Multi-Scale Minimax Hazard Utility
    _draw_box(
        ax,
        0.525,
        0.60,
        0.265,
        0.20,
        "Module B: Multi-Scale\nMinimax Utility $U(a)$",
        "$\\alpha\\cdot\\mathrm{ProgressGain} - \\beta\\max_{w\\in\\{3,5,8\\}} R_w$\nBurst & Exfil Spacing Guard",
        fc="#FFFBEB",
        ec="#E69F00",
        title_color="#92400E",
    )

    # Module C: Low-Latency Local SLM Engine
    _draw_box(
        ax,
        0.225,
        0.28,
        0.255,
        0.20,
        "Module C: Local SLM\nInference Engine",
        "Static Prefix KV-Cache\nJSON-Brace Early Stopping",
        fc="#F5F3FF",
        ec="#7C3AED",
        title_color="#4C1D95",
    )

    # Module D: 1-Step Counterfactual Reflexion & Fallback
    _draw_box(
        ax,
        0.525,
        0.28,
        0.265,
        0.20,
        "Module D: 1-Step Reflexion\n& Minimax Fallback",
        "Pass 1 -> Cue Pass 2 ->\n$\\arg\\max_a U(a)$ (Logged Override)",
        fc="#FFF1F2",
        ec="#D55E00",
        title_color="#9A3412",
    )

    # Right: Executed Action & Telemetry
    _draw_box(
        ax,
        0.84,
        0.42,
        0.14,
        0.26,
        "Action $a_t$ &\nTelemetry",
        "1stPass / PostRef %\nEntropy $H_t$ & VRAM",
        fc="#F0F9FF",
        ec="#0284C7",
        title_color="#075985",
    )

    # Arrows
    _draw_arrow(ax, (0.168, 0.59), (0.220, 0.69), label="$o_t, r_t$", color="#0072B2", label_offset=(-0.005, 0.042))
    _draw_arrow(ax, (0.483, 0.70), (0.520, 0.70), label="$\\mathcal{A}_t^{\\mathrm{valid}}$", color="#009E73", label_offset=(0.0, 0.042))
    _draw_arrow(ax, (0.655, 0.595), (0.365, 0.485), label="Ranked Frontier", color="#B45309", rad=0.10, label_offset=(-0.025, 0.032))
    _draw_arrow(ax, (0.483, 0.38), (0.520, 0.38), label="JSON $r_1$", color="#6D28D9", label_offset=(0.0, 0.038))
    _draw_arrow(ax, (0.62, 0.275), (0.38, 0.275), label="1-Step Cue (if invalid/repeat)", color="#D55E00", rad=-0.20, label_offset=(0.0, -0.052))
    _draw_arrow(ax, (0.793, 0.40), (0.836, 0.50), label="$a_t$", color="#0F172A", label_offset=(0.0, 0.038))
    # Orthogonal feedback path cleanly routed below the cognitive wrapper (y = 0.09)
    ax.plot([0.91, 0.91, 0.09], [0.415, 0.09, 0.09], color="#475569", linewidth=1.6, zorder=4)
    _draw_arrow(ax, (0.09, 0.09), (0.09, 0.415), color="#475569", lw=1.6)
    ax.text(
        0.50,
        0.09,
        "Environment Step Transition $\\mathcal{T}(s_{t+1} \\mid s_t, a_t)$",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95),
        zorder=5,
    )

    ax.set_title(
        "Domain-Agnostic Stealth-Aware Topological Belief Graph & Multi-Scale Minimax SLM Pipeline",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_system_architecture.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_belief_and_hazard_mechanism(out_dir: Path) -> None:
    """Two-panel mechanism diagram: (a) Epistemic FSM, (b) Multi-Scale Hazard U(a)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2))

    # --- Panel (a): Epistemic Host Lifecycle State Machine ---
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")

    _draw_box(ax1, 0.04, 0.56, 0.34, 0.24, "UNEXPLORED", "Discovered IP\nNo Service Probe Yet", "#FEF3C7", "#D97706", "#92400E")
    _draw_box(ax1, 0.60, 0.56, 0.35, 0.24, "MAPPED", "Remote Services Known\nExploit Candidate Ready", "#DBEAFE", "#2563EB", "#1E40AF")
    _draw_box(ax1, 0.60, 0.14, 0.35, 0.24, "COMPROMISED", "Controlled Foothold\nPivot & Data Search", "#D1FAE5", "#059669", "#065F46")
    _draw_box(ax1, 0.04, 0.14, 0.34, 0.24, "EXHAUSTED", "All Valid Edges Pruned\nExcluded from Frontier", "#F1F5F9", "#64748B", "#334155")

    _draw_arrow(ax1, (0.385, 0.68), (0.595, 0.68), label="FindServices\n(Services > 0)", color="#1D4ED8", label_offset=(0.0, 0.09))
    _draw_arrow(ax1, (0.775, 0.555), (0.775, 0.385), label="ExploitService\n(Shell Gained)", color="#047857", label_offset=(0.135, 0.0))
    _draw_arrow(ax1, (0.595, 0.26), (0.385, 0.26), label="FindData + Exfil\n(No New Edges)", color="#475569", label_offset=(0.0, -0.095))
    _draw_arrow(ax1, (0.21, 0.555), (0.21, 0.385), label="No Remote Srv", color="#B45309", label_offset=(-0.115, 0.0))

    # Top annotation for ScanNetwork discovery
    ax1.text(
        0.50,
        0.92,
        "ScanNetwork(subnet) discovers new hosts -> UNEXPLORED",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec="#CBD5E1"),
    )
    ax1.set_title("(a) Epistemic Host Lifecycle in Topological Belief Graph $\\mathcal{G}_t$", fontweight="bold", pad=8)

    # --- Panel (b): Multi-Scale Hazard Utility Comparison ---
    actions = ["FindServices\n(Consec. Burst)", "ScanNetwork\n(Repeat Window)", "FindData\n(Fresh Host)", "ExploitService\n(Interleaved)", "ExfiltrateData\n(Spaced >= w)"]
    gains = np.array([0.42, 0.22, 0.80, 0.62, 1.08])
    hazards = np.array([0.55, 0.67, 0.20, 0.20, 0.20])
    utilities = gains - 1.2 * hazards

    y = np.arange(len(actions))
    h_bar = 0.26
    ax2.barh(y + h_bar, gains, height=h_bar, color="#0072B2", alpha=0.88, label="ProgressGain $\\alpha \\cdot G(a)$")
    ax2.barh(y, -1.2 * hazards, height=h_bar, color="#D55E00", alpha=0.85, label="Minimax Penalty $-\\beta \\max_{w\\in\\{3,5,8\\}} R_w$")
    ax2.barh(y - h_bar, utilities, height=h_bar, color="#009E73", alpha=0.92, label="Net Utility $U(a)$")

    ax2.axvline(0.0, color="#1F2937", linewidth=1.0, linestyle="-")
    for idx, u_val in enumerate(utilities):
        offset = 0.03 if u_val >= 0 else -0.03
        ha = "left" if u_val >= 0 else "right"
        ax2.text(u_val + offset, idx - h_bar, f"{u_val:+.2f}", va="center", ha=ha, fontsize=8.0, fontweight="bold", color="#065F46" if u_val >= 0 else "#991B1B")

    ax2.set_yticks(y)
    ax2.set_yticklabels(actions, fontsize=8.5)
    ax2.set_xlim(-1.05, 1.32)
    ax2.set_xlabel("Decision Utility / Component Score ($\\alpha = 1.0, \\beta = 1.2$)")
    ax2.set_title("(b) Multi-Scale Minimax Hazard Utility $U(a)$ Across $w \\in \\{3,5,8\\}$", fontweight="bold", pad=8)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=8.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_belief_and_hazard_mechanism.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conceptual architecture & mechanism figures.")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1_system_architecture(out_dir)
    plot_fig2_belief_and_hazard_mechanism(out_dir)
    print(f"Generated architectural figures in {out_dir}")


if __name__ == "__main__":
    main()
