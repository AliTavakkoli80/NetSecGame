"""
generate_architecture_diagrams.py — Publication-Grade Elsevier ESWA Vector Diagrams.

Generates two clean, high-clarity conceptual figures in `figures/` (300 DPI PNG, PDF, SVG)
at exact two-column print scale (7.0 in / 178 mm):
1. fig1_system_architecture.{png,pdf,svg}:
   Clockwise closed-loop system architecture showing the NetSecGame C-POMDP environment
   coupled with the four zero-hardcoding modules of ModernSLMAttacker:
   (A) Topological Belief Graph G_t, (B) Multi-Scale Minimax Hazard Utility U(a),
   (C) KV-Cached Local SLM Engine with JSON early stopping, and (D) Validation,
   1-Step Counterfactual Reflexion & Deadlock-Safe Fallback.
2. fig2_belief_and_hazard_mechanism.{png,pdf,svg}:
   (a) Epistemic host lifecycle state machine inside the Topological Belief Graph G_t.
   (b) Multi-Scale Minimax Hazard Utility U(a) decomposition across w in {3, 5, 8},
       computed dynamically via SemanticMemoryStore().compute_minimax_hazard_utility().
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research_extension.memory_module import SemanticMemoryStore

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
TEAL_TINT = "#EDF6F6"
SLATE_BLUE = "#2B5C8F"
SLATE_BLUE_TINT = "#EEF4FA"
CRIMSON = "#B83A3A"
CRIMSON_TINT = "#FDF2F2"
AMBER = "#D97706"
AMBER_TINT = "#FEF7EC"
AMBER_DARK = "#9A4D04"
CHARCOAL = "#334155"
CHARCOAL_TINT = "#F1F5F9"

TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#475569"
BORDER_SUBTLE = "#CBD5E1"
BG_WHITE = "#FFFFFF"
BG_CONTAINER = "#F8FAFC"


def _draw_rounded_card(
    ax: plt.Axes,
    x0: float,
    y0: float,
    w: float,
    h: float,
    title: str,
    header_bg: str,
    edge_color: str,
    title_color: str,
    header_h: float = 0.28,
    rounding: float = 0.06,
    lw: float = 1.2,
    title_fontsize: float = 7.6,
    zorder: float = 2.0,
) -> float:
    """
    Draw a publication-grade rounded module card with a tinted header band.
    Returns the y-coordinate of the header divider line (top of the card body).
    """
    hy = y0 + h - header_h

    base = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=BG_WHITE,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_patch(base)

    hdr_top = FancyBboxPatch(
        (x0, hy),
        w,
        header_h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=header_bg,
        edgecolor="none",
        zorder=zorder + 0.05,
    )
    ax.add_patch(hdr_top)
    hdr_flat = Rectangle(
        (x0, hy),
        w,
        header_h * 0.48,
        facecolor=header_bg,
        edgecolor="none",
        zorder=zorder + 0.06,
    )
    ax.add_patch(hdr_flat)

    ax.plot(
        [x0, x0 + w],
        [hy, hy],
        color=edge_color,
        linewidth=0.75,
        alpha=0.85,
        zorder=zorder + 0.15,
    )

    border = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor="none",
        edgecolor=edge_color,
        linewidth=lw,
        zorder=zorder + 0.25,
    )
    ax.add_patch(border)

    ax.text(
        x0 + w * 0.5,
        hy + header_h * 0.5,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 1.5,
    )
    return hy


def _draw_subpill(
    ax: plt.Axes,
    x0: float,
    y0: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    lw: float = 0.75,
    ls: str = "-",
    rounding: float = 0.05,
    zorder: float = 2.4,
) -> None:
    """Draw a subtle inner rounded callout box."""
    patch = FancyBboxPatch(
        (x0, y0),
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


def _draw_polyline_arrow(
    ax: plt.Axes,
    points: Iterable[Tuple[float, float]],
    color: str = CHARCOAL,
    lw: float = 1.15,
    ls: str = "-",
    mutation_scale: float = 10.0,
    zorder: float = 3.2,
    head: bool = True,
) -> None:
    """Draw an orthogonal or straight polyline with a crisp arrowhead on the last segment."""
    pts = list(points)
    if len(pts) < 2:
        return
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=zorder)

    if head:
        arrow = FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            linestyle=ls,
            mutation_scale=mutation_scale,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder + 0.05,
        )
        ax.add_patch(arrow)
    else:
        ax.plot(
            [pts[-2][0], pts[-1][0]],
            [pts[-2][1], pts[-1][1]],
            color=color,
            linewidth=lw,
            linestyle=ls,
            solid_capstyle="round",
            zorder=zorder,
        )


def _draw_label_pill(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str = TEXT_PRIMARY,
    fontsize: float = 7.2,
    bold: bool = False,
    fc: str = BG_WHITE,
    ec: str = BORDER_SUBTLE,
    pad: float = 0.18,
    ha: str = "center",
    va: str = "center",
    zorder: float = 4.5,
) -> None:
    """Draw a cleanly padded white-backed text pill so no arrow line ever crosses text."""
    bbox = dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=0.65) if fc else None
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=color,
        bbox=bbox,
        zorder=zorder,
    )


def plot_fig1_system_architecture(out_dir: Path) -> None:
    """
    Figure 1: Clockwise closed-loop system architecture of ModernSLMAttacker
    interacting with the NetSecGame C-POMDP environment (exact 7.0 in width).
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.35))
    ax.set_xlim(0.0, 7.0)
    ax.set_ylim(0.0, 4.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # 1. LEFT COLUMN: NetSecGame C-POMDP Environment
    # =========================================================================
    env_x0, env_y0, env_w, env_h = 0.03, 0.14, 1.54, 4.07
    env_cx = env_x0 + env_w * 0.5
    _draw_rounded_card(
        ax,
        env_x0,
        env_y0,
        env_w,
        env_h,
        title="NetSecGame C-POMDP",
        header_bg=CHARCOAL_TINT,
        edge_color=CHARCOAL,
        title_color=TEXT_PRIMARY,
        header_h=0.33,
        rounding=0.07,
        lw=1.2,
        title_fontsize=7.6,
        zorder=2.0,
    )

    sub_x0, sub_w = env_x0 + 0.09, env_w - 0.18

    # Sub-block 1: Dynamic Topology
    _draw_subpill(ax, sub_x0, 2.86, sub_w, 0.88, BG_CONTAINER, BORDER_SUBTLE, rounding=0.05, zorder=2.4)
    ax.text(env_cx, 3.55, "Dynamic Topologies", ha="center", va="center", fontsize=7.3, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 3.29, "Full & Three-Networks", ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, zorder=4.0)
    ax.text(env_cx, 3.03, "Random IPs ($T_{\\max}=100$)", ha="center", va="center", fontsize=6.9, color=TEXT_MUTED, zorder=4.0)

    # Sub-block 2: Partial Observation
    _draw_subpill(ax, sub_x0, 1.62, sub_w, 1.08, BG_CONTAINER, BORDER_SUBTLE, rounding=0.05, zorder=2.4)
    ax.text(env_cx, 2.49, "Partial Observation $o_t$", ha="center", va="center", fontsize=7.3, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 2.18, "$(\\mathcal{K}_t^{\\mathrm{net}}, \\mathcal{K}_t^{\\mathrm{host}}, \\mathcal{K}_t^{\\mathrm{ctrl}},$", ha="center", va="center", fontsize=7.1, color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 1.86, "$\\mathcal{K}_t^{\\mathrm{srv}}, \\mathcal{K}_t^{\\mathrm{data}}, \\mathcal{K}_t^{\\mathrm{blk}}, g)$", ha="center", va="center", fontsize=7.1, color=TEXT_PRIMARY, zorder=4.0)

    # Sub-block 3: Active GlobalDefender Monitor
    _draw_subpill(ax, sub_x0, 0.28, sub_w, 1.18, CRIMSON_TINT, CRIMSON, lw=0.95, rounding=0.05, zorder=2.4)
    ax.text(env_cx, 1.26, "GlobalDefender", ha="center", va="center", fontsize=7.4, fontweight="bold", color=CRIMSON, zorder=4.0)
    ax.text(env_cx, 1.00, "Hidden Window $W^*$", ha="center", va="center", fontsize=7.1, color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 0.72, "Ratio $\\tau_{\\mathrm{ratio}}$ & $\\tau_{\\mathrm{consec}}$", ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, zorder=4.0)
    ax.text(env_cx, 0.45, "Alert Penalty ($r = -10$)", ha="center", va="center", fontsize=6.9, color=CRIMSON, zorder=4.0)

    # =========================================================================
    # 2. RIGHT ENCLOSURE: ModernSLMAttacker Planner (no embedded top banner)
    # =========================================================================
    plan_x0, plan_y0, plan_w, plan_h = 2.02, 0.08, 4.95, 4.19
    enclosure = FancyBboxPatch(
        (plan_x0, plan_y0),
        plan_w,
        plan_h,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=BG_CONTAINER,
        edgecolor="#94A3B8",
        linewidth=1.0,
        linestyle="--",
        zorder=1.0,
    )
    ax.add_patch(enclosure)

    # Exact aligned coordinate grid for Modules (A), (B), (C), (D)
    col_left_x0 = 2.11
    col_w = 2.13
    col_right_x0 = 4.76  # horizontal corridor = 0.52 in (4.24 .. 4.76)
    row_top_y0, row_top_h = 2.52, 1.66  # 2.52 .. 4.18
    row_bot_y0, row_bot_h = 0.17, 1.66  # 0.17 .. 1.83; middle corridor = 0.69 in (1.83 .. 2.52)

    left_cx = col_left_x0 + col_w * 0.5
    right_cx = col_right_x0 + col_w * 0.5
    top_mid_y = row_top_y0 + row_top_h * 0.5
    bot_mid_y = row_bot_y0 + row_bot_h * 0.5

    # -------------------------------------------------------------------------
    # Module (A): Topological Belief Graph G_t (Top-Left)
    # -------------------------------------------------------------------------
    _draw_rounded_card(
        ax,
        col_left_x0,
        row_top_y0,
        col_w,
        row_top_h,
        title="(A) Topological Belief Graph $\\mathcal{G}_t$",
        header_bg=SLATE_BLUE_TINT,
        edge_color=SLATE_BLUE,
        title_color=SLATE_BLUE,
        header_h=0.30,
        rounding=0.07,
        lw=1.2,
        title_fontsize=7.4,
        zorder=2.0,
    )
    ax.text(left_cx, 3.74, "4 Epistemic Host States:", ha="center", va="center", fontsize=7.3, color=TEXT_PRIMARY, zorder=4.0)

    badge_w, badge_h = 0.92, 0.24
    b_gap_x, b_gap_y = 0.08, 0.06
    bx_left = left_cx - badge_w - b_gap_x * 0.5
    bx_right = left_cx + b_gap_x * 0.5
    by_top = 3.35
    by_bot = by_top - badge_h - b_gap_y

    ep_grid = [
        (bx_left, by_top, "UNEXPLORED", CHARCOAL_TINT, CHARCOAL),
        (bx_right, by_top, "MAPPED", SLATE_BLUE_TINT, SLATE_BLUE),
        (bx_left, by_bot, "EXHAUSTED", CHARCOAL_TINT, TEXT_MUTED),
        (bx_right, by_bot, "COMPROMISED", TEAL_TINT, TEAL),
    ]
    for bx, by, lbl, bfc, btc in ep_grid:
        _draw_subpill(ax, bx, by, badge_w, badge_h, bfc, BORDER_SUBTLE, lw=0.65, rounding=0.04, zorder=2.5)
        ax.text(bx + badge_w * 0.5, by + badge_h * 0.5, lbl, ha="center", va="center", fontsize=6.7, fontweight="bold", color=btc, zorder=4.0)

    ax.text(left_cx, 2.86, "Prunes executed $\\mathcal{E}_{\\mathrm{exec}}$ & blocks $\\mathcal{K}_t^{\\mathrm{blk}}$", ha="center", va="center", fontsize=6.9, color=TEXT_MUTED, zorder=4.0)
    ax.text(left_cx, 2.65, "Emits valid frontier $\\mathcal{A}_t^{\\mathrm{valid}}$", ha="center", va="center", fontsize=7.2, fontweight="bold", color=SLATE_BLUE, zorder=4.0)

    # -------------------------------------------------------------------------
    # Module (B): Multi-Scale Minimax Hazard U(a) (Top-Right)
    # -------------------------------------------------------------------------
    _draw_rounded_card(
        ax,
        col_right_x0,
        row_top_y0,
        col_w,
        row_top_h,
        title="(B) Multi-Scale Minimax Hazard $U(a)$",
        header_bg=TEAL_TINT,
        edge_color=TEAL,
        title_color=TEAL,
        header_h=0.30,
        rounding=0.07,
        lw=1.2,
        title_fontsize=7.4,
        zorder=2.0,
    )
    _draw_subpill(ax, col_right_x0 + 0.09, 3.24, col_w - 0.18, 0.54, BG_CONTAINER, BORDER_SUBTLE, lw=0.75, rounding=0.05, zorder=2.5)
    ax.text(
        right_cx,
        3.62,
        "$U(a) = \\alpha\\,\\mathrm{ProgressGain}(a \\mid \\mathcal{G}_t)$",
        ha="center",
        va="center",
        fontsize=7.2,
        color=TEXT_PRIMARY,
        zorder=4.0,
    )
    ax.text(
        right_cx,
        3.38,
        "$-\\;\\beta \\max_{w \\in \\{3,5,8\\}} R_w(a \\mid \\mathcal{H}_t)$",
        ha="center",
        va="center",
        fontsize=7.2,
        color=CRIMSON,
        zorder=4.0,
    )
    ax.text(right_cx, 3.04, "Suppresses burst & exfiltration clusters", ha="center", va="center", fontsize=6.8, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 2.84, "(no known monitor window $W^*$)", ha="center", va="center", fontsize=6.8, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 2.65, "Soft mask & descending $U(a)$ rank", ha="center", va="center", fontsize=7.1, fontweight="bold", color=TEAL, zorder=4.0)

    # -------------------------------------------------------------------------
    # Module (C): Low-Latency Local SLM Engine (Bottom-Right)
    # -------------------------------------------------------------------------
    _draw_rounded_card(
        ax,
        col_right_x0,
        row_bot_y0,
        col_w,
        row_bot_h,
        title="(C) Low-Latency Local SLM Engine",
        header_bg=AMBER_TINT,
        edge_color=AMBER,
        title_color=AMBER_DARK,
        header_h=0.30,
        rounding=0.07,
        lw=1.2,
        title_fontsize=7.4,
        zorder=2.0,
    )
    ax.text(right_cx, 1.36, "Qwen2.5-Coder (7B-NF4 / 1.5B-BF16)", ha="center", va="center", fontsize=6.9, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(right_cx, 1.13, "Peak VRAM: 5.92 GB / 3.22 GB", ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 0.87, "Static System-Prefix KV-Cache", ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 0.62, "JSON-Brace Early Stopping ($T=0$)", ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 0.34, "Autonomous choice on frontier", ha="center", va="center", fontsize=7.1, fontweight="bold", color=AMBER_DARK, zorder=4.0)

    # -------------------------------------------------------------------------
    # Module (D): Validation, 1-Step Reflexion & Fallback (Bottom-Left)
    # -------------------------------------------------------------------------
    _draw_rounded_card(
        ax,
        col_left_x0,
        row_bot_y0,
        col_w,
        row_bot_h,
        title="(D) Validation, Reflexion & Fallback",
        header_bg=CHARCOAL_TINT,
        edge_color=CHARCOAL,
        title_color=TEXT_PRIMARY,
        header_h=0.30,
        rounding=0.07,
        lw=1.2,
        title_fontsize=7.4,
        zorder=2.0,
    )
    col_rhs = col_left_x0 + 0.96
    ax.text(col_left_x0 + 0.08, 1.29, "1. Pass-1 Valid", ha="left", va="center", fontsize=6.9, fontweight="bold", color=TEAL, zorder=4.0)
    ax.text(col_rhs, 1.29, "$\\rightarrow$ Execute $a_1$", ha="left", va="center", fontsize=6.9, color=TEXT_PRIMARY, zorder=4.0)

    ax.text(col_left_x0 + 0.08, 0.88, "2. Invalid/Rep.", ha="left", va="center", fontsize=6.9, fontweight="bold", color=CRIMSON, zorder=4.0)
    ax.text(col_rhs, 0.88, "$\\rightarrow$ 1-Step Cue to (C)", ha="left", va="center", fontsize=6.9, color=TEXT_PRIMARY, zorder=4.0)

    ax.text(col_left_x0 + 0.08, 0.47, "3. Pass-2 Fail", ha="left", va="center", fontsize=6.9, fontweight="bold", color=TEXT_MUTED, zorder=4.0)
    ax.text(col_rhs, 0.47, "$\\rightarrow \\arg\\max_a U(a)$", ha="left", va="center", fontsize=6.9, color=TEXT_PRIMARY, zorder=4.0)

    # =========================================================================
    # 3. CLOCKWISE CLOSED-LOOP ARROWS (ALL LABELS WITH WHITE PILLS)
    # =========================================================================
    # Env -> (A): Partial Observation o_t, r_t
    _draw_polyline_arrow(
        ax,
        [(env_x0 + env_w, top_mid_y), (col_left_x0, top_mid_y)],
        color=CHARCOAL,
        lw=1.2,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (env_x0 + env_w + col_left_x0) * 0.5 - 0.02,
        top_mid_y + 0.25,
        "$o_t, r_t$",
        color=TEXT_PRIMARY,
        fontsize=7.6,
        bold=True,
        fc=BG_WHITE,
        ec=BORDER_SUBTLE,
        pad=0.14,
    )

    # (A) -> (B): Valid Frontier A_t^valid
    _draw_polyline_arrow(
        ax,
        [(col_left_x0 + col_w, top_mid_y), (col_right_x0, top_mid_y)],
        color=SLATE_BLUE,
        lw=1.2,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (col_left_x0 + col_w + col_right_x0) * 0.5,
        top_mid_y + 0.26,
        "$\\mathcal{A}_t^{\\mathrm{valid}}$",
        color=SLATE_BLUE,
        fontsize=7.9,
        bold=True,
        fc=BG_WHITE,
        ec=SLATE_BLUE_TINT,
        pad=0.14,
    )

    # (B) -> (C): Ranked Frontier + Stealth Advisory (right vertical channel)
    x_bc = 6.32
    _draw_polyline_arrow(
        ax,
        [(x_bc, row_top_y0), (x_bc, row_bot_y0 + row_bot_h)],
        color=TEAL,
        lw=1.2,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        x_bc,
        (row_top_y0 + row_bot_y0 + row_bot_h) * 0.5,
        "Ranked $U(a)$\n& Advisory",
        color=TEAL,
        fontsize=7.0,
        bold=True,
        fc=BG_WHITE,
        ec=TEAL_TINT,
        pad=0.15,
        ha="center",
    )

    # (D) -> (C): 1-Step Counterfactual Reflexion (upper dashed crimson track)
    y_dc = 1.32
    _draw_polyline_arrow(
        ax,
        [(col_left_x0 + col_w, y_dc), (col_right_x0, y_dc)],
        color=CRIMSON,
        lw=1.15,
        ls="--",
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (col_left_x0 + col_w + col_right_x0) * 0.5,
        y_dc + 0.25,
        "1-Step\nReflexion",
        color=CRIMSON,
        fontsize=6.5,
        bold=True,
        fc=BG_WHITE,
        ec=CRIMSON_TINT,
        pad=0.10,
    )

    # (C) -> (D): JSON candidate r_1 / r_2 (lower horizontal track)
    y_cd = 0.68
    _draw_polyline_arrow(
        ax,
        [(col_right_x0, y_cd), (col_left_x0 + col_w, y_cd)],
        color=AMBER_DARK,
        lw=1.2,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (col_left_x0 + col_w + col_right_x0) * 0.5,
        y_cd - 0.25,
        "JSON\n$r_1 / r_2$",
        color=AMBER_DARK,
        fontsize=6.7,
        bold=True,
        fc=BG_WHITE,
        ec=AMBER_TINT,
        pad=0.11,
    )

    # (D) -> Env: Executed Action a_t
    _draw_polyline_arrow(
        ax,
        [(col_left_x0, bot_mid_y), (env_x0 + env_w, bot_mid_y)],
        color=CHARCOAL,
        lw=1.2,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        1.78,
        bot_mid_y - 0.29,
        "Action $a_t$",
        color=TEXT_PRIMARY,
        fontsize=7.0,
        bold=True,
        fc=BG_WHITE,
        ec=BORDER_SUBTLE,
        pad=0.12,
    )

    # =========================================================================
    # 4. MEMORY UPDATE FEEDBACK BUS (a_t -> E_exec in A & H_t in B)
    # =========================================================================
    x_tap = 1.88
    y_bus = 2.17
    x_to_a = 3.00
    x_to_b = 5.44

    ax.add_patch(Circle((x_tap, bot_mid_y), 0.036, facecolor=CHARCOAL, edgecolor="none", zorder=4.2))

    _draw_polyline_arrow(
        ax,
        [(x_tap, bot_mid_y), (x_tap, y_bus), (x_to_b, y_bus), (x_to_b, row_top_y0)],
        color=TEXT_MUTED,
        lw=1.05,
        ls="--",
        mutation_scale=9.5,
        zorder=3.1,
    )
    ax.add_patch(Circle((x_to_a, y_bus), 0.032, facecolor=TEXT_MUTED, edgecolor="none", zorder=3.6))
    _draw_polyline_arrow(
        ax,
        [(x_to_a, y_bus), (x_to_a, row_top_y0)],
        color=TEXT_MUTED,
        lw=1.05,
        ls="--",
        mutation_scale=9.5,
        zorder=3.1,
    )
    _draw_label_pill(
        ax,
        (x_to_a + x_to_b) * 0.5,
        y_bus,
        "Episodic & Edge Update $(a_t \\rightarrow \\mathcal{E}_{\\mathrm{exec}},\\, \\mathcal{H}_t)$",
        color=TEXT_MUTED,
        fontsize=6.9,
        bold=False,
        fc=BG_WHITE,
        ec=BORDER_SUBTLE,
        pad=0.15,
        zorder=4.5,
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"fig1_system_architecture.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def plot_fig2_belief_and_hazard_mechanism(out_dir: Path) -> None:
    """
    Figure 2: Two-panel mechanism diagram at exact 7.0 in width:
    (a) Epistemic host lifecycle state machine inside Topological Belief Graph G_t.
    (b) Multi-Scale Minimax Hazard Utility U(a) decomposition across w in {3, 5, 8}.
    """
    fig = plt.figure(figsize=(7.0, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.51, 0.49], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # =========================================================================
    # PANEL (a): Epistemic Host Lifecycle State Machine in G_t
    # =========================================================================
    ax1.set_xlim(0.0, 4.10)
    ax1.set_ylim(0.0, 3.45)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Symmetrical 2x2 grid with generous gaps so pills never touch card borders
    sw, sh = 1.44, 0.80
    cx_left, cx_right = 0.75, 3.35   # horizontal gap = 2.63 - 1.47 = 1.16 units
    cy_bot, cy_top = 0.46, 2.34      # vertical gap = 1.88 - 0.80 = 1.08 units

    # Initial entry marker -> UNEXPLORED
    init_x, init_y = 0.14, 3.08
    ax1.add_patch(Circle((init_x, init_y), 0.055, facecolor=CHARCOAL, edgecolor=CHARCOAL, zorder=4.0))
    _draw_polyline_arrow(
        ax1,
        [(init_x + 0.055, init_y), (cx_left, init_y), (cx_left, cy_top + sh * 0.5)],
        color=CHARCOAL,
        lw=1.05,
        mutation_scale=9.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        0.96,
        init_y,
        "ScanNetwork [new host]",
        color=TEXT_PRIMARY,
        fontsize=6.8,
        fc=BG_WHITE,
        ec=BORDER_SUBTLE,
        pad=0.15,
        ha="left",
        zorder=4.5,
    )

    states = [
        (
            cx_left,
            cy_top,
            "UNEXPLORED",
            CHARCOAL_TINT,
            CHARCOAL,
            TEXT_PRIMARY,
            "$h \\in \\mathcal{K}_t^{\\mathrm{host}} \\setminus \\mathcal{K}_t^{\\mathrm{ctrl}}$",
            "Not yet probed",
        ),
        (
            cx_right,
            cy_top,
            "MAPPED",
            SLATE_BLUE_TINT,
            SLATE_BLUE,
            SLATE_BLUE,
            "$|\\mathcal{S}_t(h)| > 0,\\; h \\notin \\mathcal{K}_t^{\\mathrm{ctrl}}$",
            "Remote services found",
        ),
        (
            cx_right,
            cy_bot,
            "COMPROMISED",
            TEAL_TINT,
            TEAL,
            TEAL,
            "$h \\in \\mathcal{K}_t^{\\mathrm{ctrl}}$",
            "Active pivot & searched",
        ),
        (
            cx_left,
            cy_bot,
            "EXHAUSTED",
            CHARCOAL_TINT,
            TEXT_MUTED,
            TEXT_MUTED,
            "$\\mathcal{A}_t^{\\mathrm{valid}}(h) = \\emptyset$",
            "Pruned from frontier",
        ),
    ]
    for cx, cy, title, hbg, ec, hc, formula, desc in states:
        x0 = cx - sw * 0.5
        y0 = cy - sh * 0.5
        _draw_rounded_card(
            ax1,
            x0,
            y0,
            sw,
            sh,
            title=title,
            header_bg=hbg,
            edge_color=ec,
            title_color=hc,
            header_h=0.23,
            rounding=0.06,
            lw=1.2,
            title_fontsize=7.2,
            zorder=2.0,
        )
        ax1.text(cx, y0 + 0.39, formula, ha="center", va="center", fontsize=6.9, color=TEXT_PRIMARY, zorder=4.0)
        ax1.text(cx, y0 + 0.14, desc, ha="center", va="center", fontsize=5.9, color=TEXT_MUTED, zorder=4.0)

    # 1. UNEXPLORED -> MAPPED (top horizontal, pill centered in wide 1.16 gap)
    _draw_polyline_arrow(
        ax1,
        [(cx_left + sw * 0.5, cy_top), (cx_right - sw * 0.5, cy_top)],
        color=SLATE_BLUE,
        lw=1.10,
        mutation_scale=9.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        (cx_left + cx_right) * 0.5,
        cy_top,
        "FindServices\n[services $> 0$]",
        color=SLATE_BLUE,
        fontsize=6.3,
        bold=True,
        fc=BG_WHITE,
        ec=SLATE_BLUE_TINT,
        pad=0.12,
    )

    # 2. UNEXPLORED -> EXHAUSTED (left vertical)
    _draw_polyline_arrow(
        ax1,
        [(cx_left, cy_top - sh * 0.5), (cx_left, cy_bot + sh * 0.5)],
        color=CRIMSON,
        lw=1.10,
        mutation_scale=9.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        cx_left,
        (cy_top + cy_bot) * 0.5,
        "FindServices\n[services $= 0$]",
        color=CRIMSON,
        fontsize=6.3,
        fc=BG_WHITE,
        ec=CRIMSON_TINT,
        pad=0.12,
    )

    # 3. MAPPED -> COMPROMISED (right vertical)
    _draw_polyline_arrow(
        ax1,
        [(cx_right, cy_top - sh * 0.5), (cx_right, cy_bot + sh * 0.5)],
        color=TEAL,
        lw=1.10,
        mutation_scale=9.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        cx_right,
        (cy_top + cy_bot) * 0.5,
        "ExploitService\n[exploit ok]",
        color=TEAL,
        fontsize=6.3,
        bold=True,
        fc=BG_WHITE,
        ec=TEAL_TINT,
        pad=0.12,
    )

    # 4. MAPPED -> EXHAUSTED (diagonal dashed transition)
    diag_arrow = FancyArrowPatch(
        (cx_right - sw * 0.5, cy_top - sh * 0.5),
        (cx_left + sw * 0.5, cy_bot + sh * 0.5),
        arrowstyle="-|>",
        color=CRIMSON,
        linewidth=1.05,
        linestyle="--",
        mutation_scale=9.5,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=3.15,
    )
    ax1.add_patch(diag_arrow)
    _draw_label_pill(
        ax1,
        (cx_left + cx_right) * 0.5,
        (cy_top + cy_bot) * 0.5,
        "ExploitService\n[all failed]",
        color=CRIMSON,
        fontsize=6.3,
        fc=BG_WHITE,
        ec=BORDER_SUBTLE,
        pad=0.12,
    )

    ax1.set_title("(a) Epistemic Host Lifecycle in $\\mathcal{G}_t$", fontweight="bold", fontsize=9.5, pad=8, color=TEXT_PRIMARY)

    # =========================================================================
    # PANEL (b): Multi-Scale Minimax Hazard Utility U(a) (w in {3, 5, 8})
    # =========================================================================
    beta_val = 1.2
    specs = [
        (
            "ScanNetwork\n(Repeat $w=3$)",
            ["ScanNetwork", "FindServices"],
            "ScanNetwork",
            None,
            None,
        ),
        (
            "FindServices\n(Repeat $w=5$)",
            ["FindServices", "FindServices", "ScanNetwork", "ExploitService"],
            "FindServices",
            {"target_host": "unprobed_host"},
            None,
        ),
        (
            "ExploitService\n(Interleaved)",
            ["ScanNetwork", "FindServices"],
            "ExploitService",
            None,
            None,
        ),
        (
            "FindData\n(Fresh Host)",
            ["ScanNetwork", "ExploitService"],
            "FindData",
            None,
            None,
        ),
        (
            "ExfiltrateData\n(Spaced $\\geq w$)",
            ["ScanNetwork", "FindServices"],
            "ExfiltrateData",
            {"target_host": "goal_host"},
            "goal_host",
        ),
    ]
    actions, gains_list, penalties_list, utilities_list = [], [], [], []
    for label, hist, act, params, goal_ip in specs:
        mem = SemanticMemoryStore(enable_stealth_pacing=True)
        mem.action_history = list(hist)
        if goal_ip:
            mem.goal_target_ip = goal_ip
        u_val, g_val, h_val = mem.compute_minimax_hazard_utility(act, params, alpha=1.0, beta=beta_val)
        actions.append(label)
        gains_list.append(g_val)
        penalties_list.append(round(-beta_val * h_val, 4))
        utilities_list.append(u_val)

    gains = np.asarray(gains_list, dtype=float)
    penalties = np.asarray(penalties_list, dtype=float)
    utilities = np.asarray(utilities_list, dtype=float)

    y = np.arange(len(actions))
    h_bar = 0.24

    ax2.axvspan(-1.15, 0.0, facecolor=CRIMSON_TINT, alpha=0.45, zorder=0.2)
    ax2.axvspan(0.0, 1.35, facecolor=TEAL_TINT, alpha=0.45, zorder=0.2)
    ax2.grid(axis="x", linestyle="--", color="#94A3B8", alpha=0.25, zorder=0.5)
    ax2.grid(axis="y", visible=False)

    ax2.barh(
        y + h_bar,
        gains,
        height=h_bar,
        color=CHARCOAL,
        alpha=0.92,
        label="$\\alpha\\,\\mathrm{ProgressGain}(a\\mid\\mathcal{G}_t)$",
        zorder=2.5,
    )
    ax2.barh(
        y,
        penalties,
        height=h_bar,
        color=CRIMSON,
        alpha=0.88,
        label="$-\\beta\\max_{w\\in\\{3,5,8\\}} R_w$",
        zorder=2.5,
    )
    ax2.barh(
        y - h_bar,
        utilities,
        height=h_bar,
        color=TEAL,
        alpha=0.96,
        label="Net Utility $U(a)$",
        zorder=2.5,
    )

    ax2.axvline(0.0, color=CHARCOAL, linewidth=0.95, zorder=3.0)

    ax2.text(-0.56, 4.60, "Suppressed ($U < 0$)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=CRIMSON, zorder=3.5)
    ax2.text(0.68, 4.60, "Preferred ($U > 0$)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=TEAL, zorder=3.5)

    ax2.axhline(1.5, color=BORDER_SUBTLE, linewidth=0.85, linestyle=":", zorder=1.2)

    for idx, (g_val, p_val, u_val) in enumerate(zip(gains, penalties, utilities)):
        ax2.text(
            g_val + 0.025,
            idx + h_bar,
            f"+{g_val:.2f}",
            va="center",
            ha="left",
            fontsize=6.8,
            color=CHARCOAL,
            zorder=3.6,
        )
        ax2.text(
            p_val - 0.025,
            idx,
            f"{p_val:.2f}",
            va="center",
            ha="right",
            fontsize=6.8,
            color=CRIMSON,
            zorder=3.6,
        )
        offset = 0.028 if u_val >= 0 else -0.028
        ha = "left" if u_val >= 0 else "right"
        t_col = TEAL if u_val >= 0 else CRIMSON
        ax2.text(
            u_val + offset,
            idx - h_bar,
            f"{u_val:+.2f}",
            va="center",
            ha=ha,
            fontsize=7.2,
            fontweight="bold",
            color=t_col,
            zorder=3.6,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(actions, fontsize=7.5, color=TEXT_PRIMARY)
    ax2.set_ylim(-0.58, 4.86)
    ax2.set_xlim(-1.15, 1.35)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.tick_params(axis="x", labelsize=7.8)
    ax2.set_xlabel("Utility / Score ($\\alpha = 1.0,\\, \\beta = 1.2$)", fontsize=8.3, color=TEXT_PRIMARY)
    ax2.set_title("(b) Multi-Scale Minimax Hazard Utility $U(a)$", fontweight="bold", fontsize=9.5, pad=8, color=TEXT_PRIMARY)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(CHARCOAL)
    ax2.spines["left"].set_linewidth(0.9)
    ax2.spines["bottom"].set_color(CHARCOAL)
    ax2.spines["bottom"].set_linewidth(0.9)

    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.18),
        ncol=3,
        fontsize=6.8,
        handlelength=1.05,
        columnspacing=0.65,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=BORDER_SUBTLE,
        framealpha=0.98,
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.89, bottom=0.22)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"fig2_belief_and_hazard_mechanism.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Elsevier ESWA architecture & mechanism diagrams.")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1_system_architecture(out_dir)
    plot_fig2_belief_and_hazard_mechanism(out_dir)
    print(f"Generated architectural figures in {out_dir}")


if __name__ == "__main__":
    main()
