"""
generate_architecture_diagrams.py — Authentic Elsevier / IEEE Print Vector Diagrams.

Generates compact, print-proportion conceptual diagrams in `figures/` (300 DPI PNG + PDF):
1. fig1_system_architecture.{png,pdf}: 3-column academic block diagram with embedded micro-schematics.
2. fig2_belief_and_hazard_mechanism.{png,pdf}:
   (a) Formal epistemic host automaton in Topological Belief Graph G_t.
   (b) Multi-scale minimax hazard utility U(a) decomposition across w in {3, 5, 8}.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# Restrained Elsevier / IEEE TIFS print palette
INK_DARK = "#1E293B"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#475569"
BORDER_SUBTLE = "#94A3B8"
GRID_LIGHT = "#CBD5E1"
BG_WHITE = "#FFFFFF"
BG_TINT = "#F8FAFC"
BG_HEADER = "#F1F5F9"

BLUE_BELIEF = "#1D4ED8"
BLUE_TINT = "#EFF6FF"
TEAL_UTILITY = "#0F766E"
TEAL_TINT = "#F0FDFA"
CRIMSON_HAZARD = "#B91C1C"
CRIMSON_TINT = "#FEF2F2"
SLATE_GAIN = "#475569"


def _draw_academic_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    header_text: str,
    header_h: float = 0.056,
    header_bg: str = BG_HEADER,
    body_bg: str = BG_WHITE,
    border_color: str = INK_DARK,
    header_color: str = TEXT_PRIMARY,
    lw: float = 1.0,
    rule_lw: float = 0.7,
    header_fontsize: float = 8.0,
    rounding: float = 0.008,
    zorder: float = 2.0,
) -> None:
    """Draw a minimalist print block with a thin border and single horizontal header rule."""
    hy = y + h - header_h
    body = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=body_bg,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_patch(body)

    if header_bg != body_bg:
        hdr = FancyBboxPatch(
            (x, hy),
            w,
            header_h,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            facecolor=header_bg,
            edgecolor="none",
            zorder=zorder + 0.05,
        )
        ax.add_patch(hdr)
        # Flatten the lower header seam so the divider rule meets cleanly.
        ax.add_patch(
            plt.Rectangle(
                (x, hy),
                w,
                header_h * 0.42,
                facecolor=header_bg,
                edgecolor="none",
                zorder=zorder + 0.05,
            )
        )

    ax.plot(
        [x, x + w],
        [hy, hy],
        color=border_color,
        linewidth=rule_lw,
        zorder=zorder + 0.2,
    )

    border = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor="none",
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder + 0.3,
    )
    ax.add_patch(border)

    ax.text(
        x + w * 0.5,
        hy + header_h * 0.5,
        header_text,
        ha="center",
        va="center",
        fontsize=header_fontsize,
        fontweight="bold",
        color=header_color,
        zorder=zorder + 0.4,
    )


def _draw_node_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    fc: str = BG_WHITE,
    ec: str = INK_DARK,
    tc: str = TEXT_PRIMARY,
    lw: float = 0.85,
    ls: str = "-",
    fontsize: float = 6.6,
    bold: bool = False,
    rounding: float = 0.006,
    zorder: float = 3.0,
) -> None:
    """Draw a sharp/micro-radius schematic box with centered serif text."""
    patch = FancyBboxPatch(
        (cx - w * 0.5, cy - h * 0.5),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=tc,
        zorder=zorder + 0.1,
    )


def _draw_ortho_arrow(
    ax: plt.Axes,
    points: Iterable[Tuple[float, float]],
    label: str = "",
    label_pos: Tuple[float, float] | None = None,
    color: str = INK_DARK,
    lw: float = 1.0,
    ls: str = "-",
    label_fontsize: float = 7.2,
    label_color: str | None = None,
    zorder: float = 4.0,
    arrow: bool = True,
) -> None:
    """Draw a clean orthogonal polyline ending with a crisp vector arrowhead."""
    pts = list(points)
    if len(pts) < 2:
        return
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="butt", zorder=zorder)

    p_prev, p_end = pts[-2], pts[-1]
    if arrow:
        arr = FancyArrowPatch(
            p_prev,
            p_end,
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            linestyle=ls,
            mutation_scale=11.0,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
        ax.add_patch(arr)
    else:
        ax.plot([p_prev[0], p_end[0]], [p_prev[1], p_end[1]], color=color, linewidth=lw, linestyle=ls, zorder=zorder)

    if label and label_pos is not None:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color=label_color or color,
            bbox=dict(boxstyle="square,pad=0.14", fc=BG_WHITE, ec="none"),
            zorder=zorder + 1.0,
        )


def plot_fig1_system_architecture(out_dir: Path) -> None:
    """Compact 3-column academic block diagram matching Elsevier/IEEE print typography."""
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # --- Center Dashed System Boundary ---
    boundary = FancyBboxPatch(
        (0.214, 0.132),
        0.574,
        0.842,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        facecolor=BG_TINT,
        edgecolor=TEXT_MUTED,
        linestyle="--",
        linewidth=0.9,
        zorder=1.0,
    )
    ax.add_patch(boundary)
    ax.text(
        0.501,
        0.946,
        "Training-Free Cognitive & Stealth Architecture — Zero Hardcoding",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=TEXT_PRIMARY,
        zorder=1.5,
    )

    # --- Left Block: NetSecGame C-POMDP ---
    _draw_academic_block(
        ax,
        0.006,
        0.155,
        0.172,
        0.745,
        "NetSecGame C-POMDP",
        header_h=0.068,
        header_bg=BG_HEADER,
        body_bg=BG_WHITE,
        header_fontsize=7.9,
    )
    env_items = [
        "• Randomized Topologies",
        "• Horizon $T = 100$",
        "• GlobalDefender $W^*$",
        "• $\\mathcal{A} = \\mathcal{A}_{\\mathrm{type}} \\times \\Theta$",
    ]
    for i, line in enumerate(env_items):
        ax.text(
            0.014,
            0.785 - i * 0.062,
            line,
            ha="left",
            va="center",
            fontsize=6.8,
            color=TEXT_PRIMARY,
            zorder=3.0,
        )

    # Embedded 3-node graph inside Left Block
    _draw_node_box(ax, 0.092, 0.345, 0.156, 0.335, "", fc=BG_TINT, ec=GRID_LIGHT, lw=0.7, rounding=0.006, zorder=2.5)
    ax.text(0.092, 0.482, "Episode Topology", ha="center", va="center", fontsize=6.6, fontweight="bold", color=TEXT_MUTED, zorder=3.0)
    _draw_node_box(ax, 0.038, 0.365, 0.039, 0.088, "Client", fc=BLUE_TINT, ec=BLUE_BELIEF, tc=BLUE_BELIEF, lw=0.8, fontsize=5.6, bold=True)
    _draw_node_box(ax, 0.092, 0.365, 0.039, 0.088, "Router", fc=BG_WHITE, ec=INK_DARK, tc=TEXT_PRIMARY, lw=0.8, fontsize=5.6, bold=True)
    _draw_node_box(ax, 0.146, 0.365, 0.039, 0.088, "Server", fc=TEAL_TINT, ec=TEAL_UTILITY, tc=TEAL_UTILITY, lw=0.8, fontsize=5.6, bold=True)
    _draw_ortho_arrow(ax, [(0.0575, 0.365), (0.0725, 0.365)], color=INK_DARK, lw=0.85, zorder=3.5)
    _draw_ortho_arrow(ax, [(0.1115, 0.365), (0.1265, 0.365)], color=INK_DARK, lw=0.85, zorder=3.5)
    ax.text(0.092, 0.242, "Per-episode random IPs,\nservices & goal data", ha="center", va="center", fontsize=6.1, color=TEXT_MUTED, zorder=3.0)

    # --- Module (A): Topological Belief Graph G_t ---
    _draw_academic_block(
        ax,
        0.226,
        0.562,
        0.252,
        0.352,
        "(A) Topological Belief Graph $\\mathcal{G}_t$",
        header_h=0.056,
        header_fontsize=7.8,
    )
    m1_lines = [
        ("Epistemic Tags (4):", True, TEXT_PRIMARY),
        ("COMP, MAP, UNEXP, EXH", False, TEXT_MUTED),
        ("Pruned Edges $\\mathcal{E}_{\\mathrm{exec}}$:", True, TEXT_PRIMARY),
        ("Filter repeated tuples", False, TEXT_MUTED),
        ("Valid Frontier $\\mathcal{A}_t^{\\mathrm{valid}}$", True, TEAL_UTILITY),
    ]
    for i, (txt, is_b, col) in enumerate(m1_lines):
        ax.text(
            0.232,
            0.822 - i * 0.054,
            txt,
            ha="left",
            va="center",
            fontsize=6.1,
            fontweight="bold" if is_b else "normal",
            color=col,
            zorder=3.0,
        )

    # Mini 4-node directed graph schematic inside (A)
    _draw_node_box(ax, 0.414, 0.708, 0.116, 0.262, "", fc=BG_TINT, ec=GRID_LIGHT, lw=0.7, rounding=0.006, zorder=2.5)
    ax.text(0.414, 0.818, "Graph $\\mathcal{G}_t$", ha="center", va="center", fontsize=6.3, fontweight="bold", color=TEXT_MUTED, zorder=3.0)
    _draw_node_box(ax, 0.386, 0.744, 0.044, 0.072, "$h_0$\nCOMP", fc=TEAL_TINT, ec=TEAL_UTILITY, tc=TEAL_UTILITY, lw=0.85, fontsize=5.6, bold=True)
    _draw_node_box(ax, 0.442, 0.744, 0.044, 0.072, "$h_1$\nMAP", fc=BLUE_TINT, ec=BLUE_BELIEF, tc=BLUE_BELIEF, lw=0.85, fontsize=5.6, bold=True)
    _draw_node_box(ax, 0.442, 0.628, 0.044, 0.072, "$h_2$\nUNEXP", fc=BG_WHITE, ec=INK_DARK, tc=TEXT_PRIMARY, lw=0.85, fontsize=5.6, bold=True)
    _draw_node_box(ax, 0.386, 0.628, 0.044, 0.072, "$h_3$\nEXH", fc=BG_HEADER, ec=BORDER_SUBTLE, tc=TEXT_MUTED, lw=0.8, ls="--", fontsize=5.6, bold=True)
    _draw_ortho_arrow(ax, [(0.408, 0.744), (0.420, 0.744)], color=INK_DARK, lw=0.85, zorder=3.6)
    _draw_ortho_arrow(ax, [(0.442, 0.708), (0.442, 0.664)], color=INK_DARK, lw=0.85, zorder=3.6)
    _draw_ortho_arrow(ax, [(0.386, 0.708), (0.386, 0.664)], color=BORDER_SUBTLE, lw=0.85, ls="--", zorder=3.6)

    # --- Module (B): Multi-Scale Minimax Utility U(a) ---
    _draw_academic_block(
        ax,
        0.524,
        0.562,
        0.252,
        0.352,
        "(B) Multi-Scale Minimax Utility $U(a)$",
        header_h=0.056,
        header_fontsize=7.8,
    )
    _draw_node_box(
        ax,
        0.650,
        0.808,
        0.234,
        0.068,
        "$U(a) = \\alpha\\,\\mathrm{ProgressGain} - \\beta\\max_{w\\in\\{3,5,8\\}} R_w$",
        fc=BG_TINT,
        ec=GRID_LIGHT,
        tc=TEXT_PRIMARY,
        lw=0.7,
        fontsize=6.9,
        rounding=0.005,
    )
    _draw_node_box(ax, 0.650, 0.668, 0.234, 0.186, "", fc=BG_WHITE, ec=GRID_LIGHT, lw=0.7, rounding=0.006, zorder=2.5)
    slot_labels = ["...", "$a_{t-3}$", "$a_{t-2}$", "$a_{t-1}$", "$a_t$"]
    slot_xs = [0.560, 0.605, 0.650, 0.695, 0.740]
    for idx_s, (sx, slbl) in enumerate(zip(slot_xs, slot_labels)):
        is_cand = idx_s == 4
        _draw_node_box(
            ax,
            sx,
            0.728,
            0.036,
            0.044,
            slbl,
            fc=TEAL_TINT if is_cand else BG_TINT,
            ec=TEAL_UTILITY if is_cand else BORDER_SUBTLE,
            tc=TEAL_UTILITY if is_cand else TEXT_PRIMARY,
            lw=0.95 if is_cand else 0.7,
            fontsize=6.2,
            bold=is_cand,
            rounding=0.004,
        )

    brackets = [
        (0.632, 0.758, 0.680, TEAL_UTILITY, "$w=3$"),
        (0.587, 0.758, 0.646, BLUE_BELIEF, "$w=5$"),
        (0.542, 0.758, 0.612, CRIMSON_HAZARD, "$w=8$ (minimax $\\max R_w$)"),
    ]
    for bx0, bx1, by, bcol, blbl in brackets:
        ax.plot([bx0, bx1], [by, by], color=bcol, lw=1.05, zorder=3.3)
        ax.plot([bx0, bx0], [by, by + 0.011], color=bcol, lw=1.05, zorder=3.3)
        ax.plot([bx1, bx1], [by, by + 0.011], color=bcol, lw=1.05, zorder=3.3)
        if bx0 > 0.56:
            ax.text(bx0 - 0.007, by, blbl, ha="right", va="center", fontsize=6.1, fontweight="bold", color=bcol, zorder=3.4)
        else:
            ax.text(0.650, by - 0.011, blbl, ha="center", va="top", fontsize=6.0, fontweight="bold", color=bcol, zorder=3.4)

    # --- Module (C): Low-Latency Local SLM Engine ---
    _draw_academic_block(
        ax,
        0.226,
        0.155,
        0.252,
        0.340,
        "(C) Low-Latency Local SLM Engine",
        header_h=0.056,
        header_fontsize=7.8,
    )
    _draw_node_box(ax, 0.289, 0.382, 0.105, 0.068, "Static Prefix\n[KV-Cache]", fc=BG_HEADER, ec=INK_DARK, tc=TEXT_PRIMARY, lw=0.8, fontsize=6.2, bold=True)
    ax.text(0.352, 0.382, "+", ha="center", va="center", fontsize=8.5, fontweight="bold", color=TEXT_PRIMARY, zorder=3.5)
    _draw_node_box(ax, 0.415, 0.382, 0.105, 0.068, "State Tokens\n[$\\mathcal{G}_t$ Frontier]", fc=BLUE_TINT, ec=BLUE_BELIEF, tc=BLUE_BELIEF, lw=0.8, fontsize=6.2, bold=True)

    _draw_ortho_arrow(ax, [(0.352, 0.362), (0.352, 0.328), (0.291, 0.328), (0.291, 0.304)], color=INK_DARK, lw=0.85, zorder=3.5)
    _draw_node_box(ax, 0.291, 0.264, 0.114, 0.074, "Qwen2.5-Coder\n(1.5B-BF16 / 7B-NF4)", fc=BG_WHITE, ec=INK_DARK, tc=TEXT_PRIMARY, lw=0.9, fontsize=5.7, bold=True)
    _draw_ortho_arrow(ax, [(0.348, 0.264), (0.364, 0.264)], color=INK_DARK, lw=0.85, zorder=3.5)
    _draw_node_box(ax, 0.416, 0.264, 0.102, 0.074, "JSON {...}\nStopper", fc=TEAL_TINT, ec=TEAL_UTILITY, tc=TEAL_UTILITY, lw=0.85, fontsize=6.1, bold=True)
    ax.text(0.352, 0.184, "Greedy (T=0.0) · Prefix Reuse · <6 GB VRAM", ha="center", va="center", fontsize=6.2, color=TEXT_MUTED, zorder=3.0)

    # --- Module (D): 1-Step Reflexion & Fallback ---
    _draw_academic_block(
        ax,
        0.524,
        0.155,
        0.252,
        0.340,
        "(D) 1-Step Reflexion & Fallback",
        header_h=0.056,
        header_fontsize=7.8,
    )
    _draw_node_box(ax, 0.650, 0.388, 0.224, 0.052, "Pass 1 JSON Valid & Non-Redundant?", fc=BG_TINT, ec=INK_DARK, tc=TEXT_PRIMARY, lw=0.85, fontsize=6.4, bold=True)
    _draw_node_box(ax, 0.583, 0.286, 0.092, 0.060, "Execute $a_t$\n(Autonomous)", fc=TEAL_TINT, ec=TEAL_UTILITY, tc=TEAL_UTILITY, lw=0.9, fontsize=6.0, bold=True)
    _draw_node_box(ax, 0.713, 0.286, 0.112, 0.060, "Counterfactual Cue\n$\\rightarrow$ Pass 2 Retry", fc=BG_WHITE, ec=INK_DARK, tc=TEXT_PRIMARY, lw=0.85, fontsize=5.7, bold=True)
    _draw_node_box(ax, 0.650, 0.192, 0.224, 0.048, "Pass 2 Fail: $\\arg\\max_a U(a)$ Fallback", fc=CRIMSON_TINT, ec=CRIMSON_HAZARD, tc=CRIMSON_HAZARD, lw=0.85, fontsize=6.3, bold=True)

    _draw_ortho_arrow(ax, [(0.583, 0.362), (0.583, 0.317)], color=TEAL_UTILITY, lw=0.85, zorder=3.5)
    ax.text(0.563, 0.340, "Yes", ha="center", va="center", fontsize=6.0, fontweight="bold", color=TEAL_UTILITY, zorder=3.6)
    _draw_ortho_arrow(ax, [(0.713, 0.362), (0.713, 0.317)], color=CRIMSON_HAZARD, lw=0.85, zorder=3.5)
    ax.text(0.733, 0.340, "No", ha="center", va="center", fontsize=6.0, fontweight="bold", color=CRIMSON_HAZARD, zorder=3.6)
    _draw_ortho_arrow(ax, [(0.657, 0.286), (0.630, 0.286)], color=TEAL_UTILITY, lw=0.85, zorder=3.5)
    ax.text(0.6435, 0.302, "OK", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=TEAL_UTILITY, zorder=3.6)
    _draw_ortho_arrow(ax, [(0.713, 0.256), (0.713, 0.217)], color=CRIMSON_HAZARD, lw=0.85, zorder=3.5)
    ax.text(0.734, 0.237, "Fail", ha="center", va="center", fontsize=6.0, fontweight="bold", color=CRIMSON_HAZARD, zorder=3.6)

    # --- Right Block: Action a_t & Telemetry ---
    _draw_academic_block(
        ax,
        0.824,
        0.155,
        0.168,
        0.745,
        "Action $a_t$ & Telemetry",
        header_h=0.068,
        header_bg=BG_HEADER,
        body_bg=BG_WHITE,
        header_fontsize=8.0,
    )
    _draw_node_box(
        ax,
        0.908,
        0.710,
        0.144,
        0.118,
        "Executed Action $a_t$\n(type, params)",
        fc=TEAL_TINT,
        ec=TEAL_UTILITY,
        tc=TEAL_UTILITY,
        lw=0.9,
        fontsize=6.9,
        bold=True,
    )
    _draw_node_box(
        ax,
        0.908,
        0.520,
        0.144,
        0.165,
        "SLM Autonomy\nFirstPass / PostRef %\nOverride %",
        fc=BG_TINT,
        ec=GRID_LIGHT,
        tc=TEXT_PRIMARY,
        lw=0.75,
        fontsize=6.6,
    )
    _draw_node_box(
        ax,
        0.908,
        0.298,
        0.144,
        0.180,
        "Stealth & Hardware\nWin / Alert Rate %\nEntropy $H_t$ & VRAM",
        fc=BG_TINT,
        ec=GRID_LIGHT,
        tc=TEXT_PRIMARY,
        lw=0.75,
        fontsize=6.6,
    )

    # --- Orthogonal Inter-Module Arrows ---
    _draw_ortho_arrow(
        ax,
        [(0.178, 0.735), (0.226, 0.735)],
        label="$o_t, r_t$",
        label_pos=(0.200, 0.774),
        color=INK_DARK,
        label_color=BLUE_BELIEF,
    )
    _draw_ortho_arrow(
        ax,
        [(0.478, 0.735), (0.524, 0.735)],
        label="$\\mathcal{A}_t^{\\mathrm{valid}}$",
        label_pos=(0.500, 0.774),
        color=INK_DARK,
        label_color=TEAL_UTILITY,
    )
    _draw_ortho_arrow(
        ax,
        [(0.650, 0.562), (0.650, 0.528), (0.352, 0.528), (0.352, 0.495)],
        label="Ranked $U(a)$",
        label_pos=(0.501, 0.528),
        color=INK_DARK,
    )
    _draw_ortho_arrow(
        ax,
        [(0.478, 0.376), (0.524, 0.376)],
        label="$r_1$",
        label_pos=(0.499, 0.410),
        color=INK_DARK,
        label_color=BLUE_BELIEF,
    )
    _draw_ortho_arrow(
        ax,
        [(0.524, 0.272), (0.478, 0.272)],
        label="Cue",
        label_pos=(0.502, 0.238),
        color=CRIMSON_HAZARD,
        ls="--",
    )
    _draw_ortho_arrow(
        ax,
        [(0.776, 0.320), (0.824, 0.320)],
        label="$a_t$",
        label_pos=(0.799, 0.354),
        color=INK_DARK,
        label_color=TEAL_UTILITY,
    )
    _draw_ortho_arrow(
        ax,
        [(0.908, 0.155), (0.908, 0.056), (0.092, 0.056), (0.092, 0.155)],
        label="$\\mathcal{T}(s_{t+1} \\mid s_t, a_t)$",
        label_pos=(0.500, 0.056),
        color=INK_DARK,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_system_architecture.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_fig2_belief_and_hazard_mechanism(out_dir: Path) -> None:
    """Two-panel print mechanism figure: (a) Epistemic Host Automaton, (b) Minimax Utility U(a)."""
    fig = plt.figure(figsize=(10.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.48, 0.52], wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --- Panel (a): Epistemic Host Automaton in G_t ---
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")

    # Initial state marker
    ax1.plot([0.048], [0.892], marker="o", markersize=6.8, color=INK_DARK, zorder=4.0)
    _draw_ortho_arrow(
        ax1,
        [(0.056, 0.892), (0.335, 0.892), (0.335, 0.772)],
        label="ScanNetwork\n[new host]",
        label_pos=(0.195, 0.892),
        color=INK_DARK,
        lw=0.9,
        label_fontsize=6.8,
    )

    states = [
        (0.018, 0.522, "UNEXPLORED", BG_HEADER, BG_WHITE, INK_DARK, TEXT_PRIMARY, "$h \\in \\mathcal{K}_t^{\\mathrm{host}} \\setminus \\mathcal{K}_t^{\\mathrm{ctrl}}$", "No remote probe yet"),
        (0.622, 0.522, "MAPPED", BLUE_TINT, BG_WHITE, BLUE_BELIEF, BLUE_BELIEF, "$|\\mathcal{S}_t^{\\mathrm{rem}}(h)| > 0$", "Services enumerated"),
        (0.622, 0.088, "COMPROMISED", TEAL_TINT, BG_WHITE, TEAL_UTILITY, TEAL_UTILITY, "$h \\in \\mathcal{K}_t^{\\mathrm{ctrl}}$", "Active pivot & data"),
        (0.018, 0.088, "EXHAUSTED", BG_HEADER, BG_TINT, INK_DARK, TEXT_MUTED, "$\\mathcal{A}_t^{\\mathrm{valid}}(h) = \\emptyset$", "Pruned from frontier"),
    ]
    for sx, sy, title, hbg, bbg, ec, hc, formula, desc in states:
        _draw_academic_block(
            ax1,
            sx,
            sy,
            0.360,
            0.250,
            title,
            header_h=0.062,
            header_bg=hbg,
            body_bg=bbg,
            border_color=ec,
            header_color=hc,
            lw=0.9,
            rule_lw=0.6,
            header_fontsize=7.9,
            rounding=0.008,
        )
        cx = sx + 0.180
        ax1.text(cx, sy + 0.120, formula, ha="center", va="center", fontsize=7.6, color=TEXT_PRIMARY, zorder=3.0)
        ax1.text(cx, sy + 0.050, desc, ha="center", va="center", fontsize=6.9, color=TEXT_MUTED, zorder=3.0)

    transitions = [
        ([(0.378, 0.647), (0.622, 0.647)], "FindServices\n[services > 0]", (0.500, 0.647), BLUE_BELIEF),
        ([(0.198, 0.522), (0.198, 0.338)], "FindServices\n[services = 0]", (0.198, 0.430), CRIMSON_HAZARD),
        ([(0.802, 0.522), (0.802, 0.338)], "ExploitService\n[exploit_ok]", (0.802, 0.430), TEAL_UTILITY),
        ([(0.622, 0.213), (0.378, 0.213)], "FindData + Exfil\n[frontier $= \\emptyset$]", (0.500, 0.213), TEXT_MUTED),
    ]
    for pts, lbl, lpos, col in transitions:
        _draw_ortho_arrow(
            ax1,
            pts,
            label=lbl,
            label_pos=lpos,
            color=INK_DARK,
            label_color=col,
            lw=0.9,
            label_fontsize=6.7,
        )

    ax1.set_title("(a) Epistemic Host Automaton in $\\mathcal{G}_t$", fontweight="bold", pad=8, color=TEXT_PRIMARY)

    # --- Panel (b): Multi-Scale Minimax Utility U(a) (w in {3, 5, 8}) ---
    actions = [
        "ScanNetwork\n(Repeat Window)",
        "FindServices\n(Consec. Burst)",
        "ExploitService\n(Interleaved)",
        "FindData\n(Fresh Host)",
        "ExfiltrateData\n(Spaced $\\geq w$)",
    ]
    gains = np.array([0.22, 0.42, 0.62, 0.80, 1.08])
    hazards = np.array([0.67, 0.55, 0.20, 0.20, 0.20])
    penalties = -1.2 * hazards
    utilities = gains + penalties

    y = np.arange(len(actions))
    h_bar = 0.25

    ax2.barh(y + h_bar, gains, height=h_bar, color=SLATE_GAIN, alpha=0.92, label="ProgressGain $\\alpha\\cdot G(a)$", zorder=2.5)
    ax2.barh(y, penalties, height=h_bar, color=CRIMSON_HAZARD, alpha=0.88, label="Minimax Penalty $-\\beta\\max_{w\\in\\{3,5,8\\}} R_w$", zorder=2.5)
    ax2.barh(y - h_bar, utilities, height=h_bar, color=TEAL_UTILITY, alpha=0.95, label="Net Utility $U(a)$", zorder=2.5)

    ax2.axvline(0.0, color=INK_DARK, linewidth=0.9, zorder=3.0)
    for idx, u_val in enumerate(utilities):
        offset = 0.030 if u_val >= 0 else -0.030
        ha = "left" if u_val >= 0 else "right"
        t_col = TEAL_UTILITY if u_val >= 0 else CRIMSON_HAZARD
        ax2.text(
            u_val + offset,
            idx - h_bar,
            f"{u_val:+.2f}",
            va="center",
            ha=ha,
            fontsize=7.6,
            fontweight="bold",
            color=t_col,
            zorder=3.5,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(actions, fontsize=7.8, color=TEXT_PRIMARY)
    ax2.set_xlim(-1.08, 1.48)
    ax2.set_xlabel("Decision Utility / Score ($\\alpha = 1.0, \\beta = 1.2$)", color=TEXT_PRIMARY)
    ax2.set_title("(b) Multi-Scale Minimax Utility $U(a)$ ($w \\in \\{3,5,8\\}$)", fontweight="bold", pad=8, color=TEXT_PRIMARY)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(INK_DARK)
    ax2.spines["left"].set_linewidth(0.8)
    ax2.spines["bottom"].set_color(INK_DARK)
    ax2.spines["bottom"].set_linewidth(0.8)
    ax2.grid(axis="x", linestyle="--", color=GRID_LIGHT, alpha=0.85, zorder=0.5)
    ax2.legend(
        loc="lower right",
        fontsize=6.7,
        handlelength=1.4,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=GRID_LIGHT,
        framealpha=0.96,
    )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_belief_and_hazard_mechanism.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Elsevier/IEEE print-style architecture diagrams.")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1_system_architecture(out_dir)
    plot_fig2_belief_and_hazard_mechanism(out_dir)
    print(f"Generated architectural figures in {out_dir}")


if __name__ == "__main__":
    main()
