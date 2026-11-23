"""
generate_architecture_diagrams.py — Publication-Grade Elsevier ESWA Vector Diagrams.

Generates two clean, high-clarity conceptual figures in `figures/` (300 DPI PNG, PDF, SVG):
1. fig1_system_architecture.{png,pdf,svg}:
   Clockwise closed-loop system architecture showing the NetSecGame C-POMDP environment
   coupled with the four zero-hardcoding modules of ModernSLMAttacker:
   (A) Topological Belief Graph G_t, (B) Multi-Scale Minimax Hazard Utility U(a),
   (C) KV-Cached Local SLM Engine with JSON early stopping, and (D) Validation,
   1-Step Counterfactual Reflexion & Deadlock-Safe Fallback.
2. fig2_belief_and_hazard_mechanism.{png,pdf,svg}:
   (a) Epistemic host lifecycle state machine inside the Topological Belief Graph G_t.
   (b) Multi-Scale Minimax Hazard Utility U(a) decomposition across w in {3, 5, 8},
       contrasting preferred depth-first interleaving against burst and exfiltration suppression.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research_extension.memory_module import SemanticMemoryStore

# Elsevier ESWA publication typography: DejaVu Sans + STIX math + TrueType PDF embedding
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# Cohesive academic palette aligned with Figs. 3-5
INK_DARK = "#1E293B"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#475569"
BORDER_SUBTLE = "#CBD5E1"
GRID_LIGHT = "#CBD5E1"
BG_WHITE = "#FFFFFF"
BG_CONTAINER = "#F8FAFC"
BG_SLATE_TINT = "#F1F5F9"

BLUE_BELIEF = "#1D4ED8"
BLUE_TINT = "#EFF6FF"
TEAL_UTILITY = "#0F766E"
TEAL_TINT = "#F0FDFA"
INDIGO_SLM = "#3730A3"
INDIGO_TINT = "#EEF2FF"
CRIMSON_HAZARD = "#B91C1C"
CRIMSON_TINT = "#FEF2F2"
SLATE_GAIN = "#475569"
AMBER_MUTED = "#92400E"


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
    header_h: float = 0.34,
    rounding: float = 0.08,
    lw: float = 1.05,
    title_fontsize: float = 8.4,
    zorder: float = 2.0,
) -> float:
    """
    Draw a publication-grade rounded module card with a tinted header band.
    Returns the y-coordinate of the header divider line (top of the card body).
    """
    hy = y0 + h - header_h

    # Base white card body
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

    # Header tinted fill (rounded top + flat bottom)
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

    # Divider line beneath header
    ax.plot(
        [x0, x0 + w],
        [hy, hy],
        color=edge_color,
        linewidth=0.70,
        alpha=0.85,
        zorder=zorder + 0.15,
    )

    # Crisp outer border
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

    # Header title
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
    rounding: float = 0.06,
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
    color: str = INK_DARK,
    lw: float = 1.15,
    ls: str = "-",
    mutation_scale: float = 11.5,
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
    fontsize: float = 7.4,
    bold: bool = False,
    fc: str = BG_WHITE,
    ec: str = "none",
    pad: float = 0.16,
    ha: str = "center",
    va: str = "center",
    zorder: float = 4.2,
) -> None:
    """Draw a cleanly padded text label (optionally pill-backed) at (x, y)."""
    bbox = dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=0.6) if fc else None
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
    interacting with the NetSecGame C-POMDP environment.
    """
    fig, ax = plt.subplots(figsize=(10.5, 4.30))
    ax.set_xlim(0.0, 11.4)
    ax.set_ylim(0.12, 4.68)
    ax.axis("off")

    # =========================================================================
    # 1. LEFT COLUMN: NetSecGame C-POMDP Environment
    # =========================================================================
    env_x0, env_y0, env_w, env_h = 0.15, 0.35, 2.22, 4.10
    env_cx = env_x0 + env_w * 0.5
    _draw_rounded_card(
        ax,
        env_x0,
        env_y0,
        env_w,
        env_h,
        title="NetSecGame C-POMDP",
        header_bg=BG_SLATE_TINT,
        edge_color=INK_DARK,
        title_color=TEXT_PRIMARY,
        header_h=0.38,
        lw=1.15,
        title_fontsize=8.6,
        zorder=2.0,
    )

    # Internal Environment Sub-blocks (top to bottom)
    sub_x0, sub_w = env_x0 + 0.13, env_w - 0.26

    # Sub-block 1: Dynamic Topology
    _draw_subpill(ax, sub_x0, 3.06, sub_w, 0.88, BG_CONTAINER, BORDER_SUBTLE, zorder=2.4)
    ax.text(env_cx, 3.74, "Dynamic Topologies", ha="center", va="center", fontsize=7.8, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 3.47, "Full & Three-Networks", ha="center", va="center", fontsize=7.2, color=TEXT_MUTED, zorder=4.0)
    ax.text(env_cx, 3.23, "Randomized IPs ($T_{\\max}=100$)", ha="center", va="center", fontsize=7.1, color=TEXT_MUTED, zorder=4.0)

    # Sub-block 2: Partial Observation
    _draw_subpill(ax, sub_x0, 1.86, sub_w, 1.04, BG_CONTAINER, BORDER_SUBTLE, zorder=2.4)
    ax.text(env_cx, 2.69, "Partial Observation $o_t$", ha="center", va="center", fontsize=7.8, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 2.38, "$(\\mathcal{K}_t^{\\mathrm{net}}, \\mathcal{K}_t^{\\mathrm{host}}, \\mathcal{K}_t^{\\mathrm{ctrl}},$", ha="center", va="center", fontsize=7.4, color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 2.08, "$\\mathcal{K}_t^{\\mathrm{srv}}, \\mathcal{K}_t^{\\mathrm{data}}, \\mathcal{K}_t^{\\mathrm{blk}}, g)$", ha="center", va="center", fontsize=7.4, color=TEXT_PRIMARY, zorder=4.0)

    # Sub-block 3: Active GlobalDefender Monitor
    _draw_subpill(ax, sub_x0, 0.50, sub_w, 1.18, CRIMSON_TINT, CRIMSON_HAZARD, lw=0.85, zorder=2.4)
    ax.text(env_cx, 1.45, "GlobalDefender Monitor", ha="center", va="center", fontsize=7.7, fontweight="bold", color=CRIMSON_HAZARD, zorder=4.0)
    ax.text(env_cx, 1.16, "Hidden Window $W^*$", ha="center", va="center", fontsize=7.3, color=TEXT_PRIMARY, zorder=4.0)
    ax.text(env_cx, 0.90, "Ratio $\\tau_{\\mathrm{ratio}}$ & Burst $\\tau_{\\mathrm{consec}}$", ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, zorder=4.0)
    ax.text(env_cx, 0.66, "Alert Penalty ($r = -10$)", ha="center", va="center", fontsize=6.9, color=CRIMSON_HAZARD, zorder=4.0)

    # =========================================================================
    # 2. RIGHT ENCLOSURE: ModernSLMAttacker Planner (no embedded top banner)
    # =========================================================================
    plan_x0, plan_y0, plan_w, plan_h = 3.05, 0.22, 8.20, 4.32
    enclosure = FancyBboxPatch(
        (plan_x0, plan_y0),
        plan_w,
        plan_h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=BG_CONTAINER,
        edgecolor="#94A3B8",
        linewidth=1.0,
        linestyle="--",
        zorder=1.0,
    )
    ax.add_patch(enclosure)

    # Grid geometry for the 4 modules (A, B on top; D, C on bottom -> clockwise loop)
    col_left_x0 = 3.24
    col_w = 3.44
    col_right_x0 = 7.62  # horizontal gap = 0.94 (6.68 .. 7.62)
    row_top_y0, row_top_h = 2.78, 1.62
    row_bot_y0, row_bot_h = 0.36, 1.62  # middle corridor = 0.80 (1.98 .. 2.78)

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
        header_bg=BLUE_TINT,
        edge_color=BLUE_BELIEF,
        title_color=BLUE_BELIEF,
        header_h=0.35,
        zorder=2.0,
    )
    ax.text(left_cx, 3.83, "Compiles $o_t$ into 4 Epistemic Host States:", ha="center", va="center", fontsize=7.5, color=TEXT_PRIMARY, zorder=4.0)
    # 4 compact epistemic badges inside Module A
    ep_labels = [
        ("UNEXPLORED", BG_SLATE_TINT, TEXT_MUTED),
        ("MAPPED", BLUE_TINT, BLUE_BELIEF),
        ("COMPROMISED", TEAL_TINT, TEAL_UTILITY),
        ("EXHAUSTED", BG_SLATE_TINT, TEXT_MUTED),
    ]
    badge_w = 0.77
    gap_b = 0.055
    start_bx = col_left_x0 + 0.10
    for idx, (lbl, bfc, btc) in enumerate(ep_labels):
        bx = start_bx + idx * (badge_w + gap_b)
        _draw_subpill(ax, bx, 3.44, badge_w, 0.24, bfc, BORDER_SUBTLE, lw=0.6, rounding=0.04, zorder=2.5)
        ax.text(bx + badge_w * 0.5, 3.56, lbl, ha="center", va="center", fontsize=5.5, fontweight="bold", color=btc, zorder=4.0)

    ax.text(left_cx, 3.22, "Prunes executed edges $\\mathcal{E}_{\\mathrm{exec}}$ & blocks $\\mathcal{K}_t^{\\mathrm{blk}}$", ha="center", va="center", fontsize=7.3, color=TEXT_MUTED, zorder=4.0)
    ax.text(left_cx, 2.97, "Emits valid unexplored frontier $\\mathcal{A}_t^{\\mathrm{valid}}$", ha="center", va="center", fontsize=7.5, fontweight="bold", color=BLUE_BELIEF, zorder=4.0)

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
        edge_color=TEAL_UTILITY,
        title_color=TEAL_UTILITY,
        header_h=0.35,
        zorder=2.0,
    )
    # Framed core equation
    _draw_subpill(ax, col_right_x0 + 0.10, 3.50, col_w - 0.20, 0.45, BG_CONTAINER, BORDER_SUBTLE, lw=0.75, rounding=0.05, zorder=2.5)
    ax.text(
        right_cx,
        3.72,
        "$U(a) = \\alpha\\,\\mathrm{ProgressGain}(a \\mid \\mathcal{G}_t) - \\beta \\max_{w \\in \\{3,5,8\\}} R_w(a \\mid \\mathcal{H}_t)$",
        ha="center",
        va="center",
        fontsize=6.95,
        color=TEXT_PRIMARY,
        zorder=4.0,
    )
    ax.text(right_cx, 3.25, "Suppresses burst streaks & exfil clusters (no $W^*$)", ha="center", va="center", fontsize=7.3, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 2.98, "Deadlock-safe soft mask & descending $U(a)$ rank", ha="center", va="center", fontsize=7.4, fontweight="bold", color=TEAL_UTILITY, zorder=4.0)

    # -------------------------------------------------------------------------
    # Module (C): Local SLM Engine (Bottom-Right)
    # -------------------------------------------------------------------------
    _draw_rounded_card(
        ax,
        col_right_x0,
        row_bot_y0,
        col_w,
        row_bot_h,
        title="(C) Low-Latency Local SLM Engine",
        header_bg=INDIGO_TINT,
        edge_color=INDIGO_SLM,
        title_color=INDIGO_SLM,
        header_h=0.35,
        zorder=2.0,
    )
    ax.text(right_cx, 1.41, "Qwen2.5-Coder: 7B-NF4 (5.92 GB) / 1.5B (3.22 GB)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(right_cx, 1.11, "Static System-Prefix KV-Cache (prefills state only)", ha="center", va="center", fontsize=7.2, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 0.83, "Balanced JSON-Brace Early Stopping (greedy $T=0$)", ha="center", va="center", fontsize=7.2, color=TEXT_MUTED, zorder=4.0)
    ax.text(right_cx, 0.56, "Autonomous selection from unmasked frontier", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INDIGO_SLM, zorder=4.0)

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
        header_bg=BG_SLATE_TINT,
        edge_color=INK_DARK,
        title_color=TEXT_PRIMARY,
        header_h=0.35,
        zorder=2.0,
    )
    # 3 crisp, column-aligned outcome rows inside Module D
    col_rhs = col_left_x0 + 1.40
    ax.text(col_left_x0 + 0.14, 1.40, "1. Pass-1 Valid", ha="left", va="center", fontsize=7.2, fontweight="bold", color=TEAL_UTILITY, zorder=4.0)
    ax.text(col_rhs, 1.40, "$\\rightarrow$ Execute $a_1$ (FirstPass)", ha="left", va="center", fontsize=7.1, color=TEXT_PRIMARY, zorder=4.0)

    ax.text(col_left_x0 + 0.14, 1.04, "2. Invalid / Repeat", ha="left", va="center", fontsize=7.2, fontweight="bold", color=CRIMSON_HAZARD, zorder=4.0)
    ax.text(col_rhs, 1.04, "$\\rightarrow$ 1-Step Cue to (C)", ha="left", va="center", fontsize=7.1, color=TEXT_PRIMARY, zorder=4.0)

    ax.text(col_left_x0 + 0.14, 0.68, "3. Pass-2 Fail", ha="left", va="center", fontsize=7.2, fontweight="bold", color=TEXT_MUTED, zorder=4.0)
    ax.text(col_rhs, 0.68, "$\\rightarrow$ $\\arg\\max_{a} U(a)$ (Override)", ha="left", va="center", fontsize=7.1, color=TEXT_PRIMARY, zorder=4.0)

    # =========================================================================
    # 3. CLOCKWISE CLOSED-LOOP ARROWS (ZERO CROSSINGS)
    # =========================================================================
    # Env -> (A): Partial Observation o_t
    _draw_polyline_arrow(
        ax,
        [(env_x0 + env_w, top_mid_y), (col_left_x0, top_mid_y)],
        color=INK_DARK,
        lw=1.25,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (env_x0 + env_w + plan_x0) * 0.5,
        top_mid_y + 0.22,
        "$o_t, r_t$",
        color=TEXT_PRIMARY,
        fontsize=8.2,
        bold=True,
        fc="",
    )

    # (A) -> (B): Valid Frontier A_t^valid
    _draw_polyline_arrow(
        ax,
        [(col_left_x0 + col_w, top_mid_y), (col_right_x0, top_mid_y)],
        color=BLUE_BELIEF,
        lw=1.25,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (col_left_x0 + col_w + col_right_x0) * 0.5,
        top_mid_y + 0.24,
        "$\\mathcal{A}_t^{\\mathrm{valid}}$",
        color=BLUE_BELIEF,
        fontsize=8.8,
        bold=True,
        fc="",
    )

    # (B) -> (C): Ranked Frontier + Stealth Advisory (right vertical channel at x = 9.78)
    x_bc = 9.78
    _draw_polyline_arrow(
        ax,
        [(x_bc, row_top_y0), (x_bc, row_bot_y0 + row_bot_h)],
        color=TEAL_UTILITY,
        lw=1.25,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        x_bc + 0.14,
        (row_top_y0 + row_bot_y0 + row_bot_h) * 0.5,
        "Ranked $U(a)$\n& Advisory",
        color=TEAL_UTILITY,
        fontsize=7.3,
        bold=True,
        fc=BG_CONTAINER,
        pad=0.10,
        ha="left",
    )

    # (C) -> (D): JSON candidate r_1, r_2 (lower horizontal track)
    y_cd = 0.88
    _draw_polyline_arrow(
        ax,
        [(col_right_x0, y_cd), (col_left_x0 + col_w, y_cd)],
        color=INDIGO_SLM,
        lw=1.20,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (col_left_x0 + col_w + col_right_x0) * 0.5,
        y_cd - 0.21,
        "JSON $r_1 / r_2$",
        color=INDIGO_SLM,
        fontsize=7.5,
        bold=True,
        fc="",
    )

    # (D) -> (C): 1-Step Counterfactual Reflexion (upper dashed crimson track)
    y_dc = 1.44
    _draw_polyline_arrow(
        ax,
        [(col_left_x0 + col_w, y_dc), (col_right_x0, y_dc)],
        color=CRIMSON_HAZARD,
        lw=1.15,
        ls="--",
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (col_left_x0 + col_w + col_right_x0) * 0.5,
        y_dc + 0.24,
        "1-Step\nReflexion",
        color=CRIMSON_HAZARD,
        fontsize=7.3,
        bold=True,
        fc="",
    )

    # (D) -> Env: Executed Action a_t (bottom-left return arrow)
    _draw_polyline_arrow(
        ax,
        [(col_left_x0, bot_mid_y), (env_x0 + env_w, bot_mid_y)],
        color=INK_DARK,
        lw=1.25,
        zorder=3.4,
    )
    _draw_label_pill(
        ax,
        (env_x0 + env_w + plan_x0) * 0.5,
        bot_mid_y - 0.25,
        "Action $a_t$",
        color=TEXT_PRIMARY,
        fontsize=7.8,
        bold=True,
        fc="",
    )

    # =========================================================================
    # 4. MEMORY UPDATE FEEDBACK BUS (a_t -> E_exec in A & H_t in B)
    # =========================================================================
    # Tap cleanly from the return arrow at x = 2.72 -> up to corridor y = 2.38 -> into (A) and (B)
    x_tap = 2.72
    y_bus = 2.38
    x_to_a = left_cx
    x_to_b = 8.42  # well left of x_bc = 9.78

    # Junction dot on return arrow
    ax.add_patch(Circle((x_tap, bot_mid_y), 0.042, facecolor=INK_DARK, edgecolor="none", zorder=4.0))

    # Main dashed bus spine from tap to (B) entry
    _draw_polyline_arrow(
        ax,
        [(x_tap, bot_mid_y), (x_tap, y_bus), (x_to_b, y_bus), (x_to_b, row_top_y0)],
        color=TEXT_MUTED,
        lw=1.0,
        ls="--",
        mutation_scale=10.0,
        zorder=3.1,
    )
    # Branch upward into (A)
    ax.add_patch(Circle((x_to_a, y_bus), 0.035, facecolor=TEXT_MUTED, edgecolor="none", zorder=3.5))
    _draw_polyline_arrow(
        ax,
        [(x_to_a, y_bus), (x_to_a, row_top_y0)],
        color=TEXT_MUTED,
        lw=1.0,
        ls="--",
        mutation_scale=10.0,
        zorder=3.1,
    )
    # Clean pill label on the memory update bus
    _draw_label_pill(
        ax,
        6.68,
        y_bus,
        "Episodic & Edge Update $(a_t \\rightarrow \\mathcal{E}_{\\mathrm{exec}},\\, \\mathcal{H}_t)$",
        color=TEXT_MUTED,
        fontsize=7.2,
        bold=False,
        fc=BG_CONTAINER,
        ec=BORDER_SUBTLE,
        pad=0.18,
        zorder=4.2,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"fig1_system_architecture.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)


def plot_fig2_belief_and_hazard_mechanism(out_dir: Path) -> None:
    """
    Figure 2: Two-panel mechanism diagram:
    (a) Epistemic host lifecycle state machine inside Topological Belief Graph G_t.
    (b) Multi-Scale Minimax Hazard Utility U(a) decomposition across w in {3, 5, 8}.
    """
    fig = plt.figure(figsize=(10.5, 4.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.49, 0.51], wspace=0.33)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # =========================================================================
    # PANEL (a): Epistemic Host Lifecycle State Machine in G_t
    # =========================================================================
    ax1.set_xlim(0.0, 6.0)
    ax1.set_ylim(0.0, 4.35)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Initial entry marker -> UNEXPLORED
    init_x, init_y = 0.28, 3.92
    ax1.add_patch(Circle((init_x, init_y), 0.08, facecolor=INK_DARK, edgecolor=INK_DARK, zorder=4.0))
    _draw_polyline_arrow(
        ax1,
        [(init_x + 0.08, init_y), (1.15, init_y), (1.15, 3.44)],
        color=INK_DARK,
        lw=1.05,
        mutation_scale=10.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        1.32,
        3.92,
        "ScanNetwork [new host discovered]",
        color=TEXT_PRIMARY,
        fontsize=7.2,
        fc=BG_SLATE_TINT,
        ec=BORDER_SUBTLE,
        pad=0.16,
        ha="left",
        zorder=4.2,
    )

    # Four epistemic host state cards
    sw, sh = 2.22, 1.08
    states = [
        (
            1.15,
            2.90,
            "UNEXPLORED",
            BG_SLATE_TINT,
            INK_DARK,
            TEXT_PRIMARY,
            "$h \\in \\mathcal{K}_t^{\\mathrm{host}} \\setminus \\mathcal{K}_t^{\\mathrm{ctrl}}$",
            "Not yet probed",
        ),
        (
            4.85,
            2.90,
            "MAPPED",
            BLUE_TINT,
            BLUE_BELIEF,
            BLUE_BELIEF,
            "$|\\mathcal{S}_t(h)| > 0,\\; h \\notin \\mathcal{K}_t^{\\mathrm{ctrl}}$",
            "Remote services found",
        ),
        (
            4.85,
            0.70,
            "COMPROMISED",
            TEAL_TINT,
            TEAL_UTILITY,
            TEAL_UTILITY,
            "$h \\in \\mathcal{K}_t^{\\mathrm{ctrl}}$",
            "Active pivot & data_searched",
        ),
        (
            1.15,
            0.70,
            "EXHAUSTED",
            BG_SLATE_TINT,
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
            header_h=0.29,
            rounding=0.07,
            lw=1.0,
            title_fontsize=7.8,
            zorder=2.0,
        )
        ax1.text(cx, y0 + 0.50, formula, ha="center", va="center", fontsize=7.8, color=TEXT_PRIMARY, zorder=4.0)
        ax1.text(cx, y0 + 0.20, desc, ha="center", va="center", fontsize=6.4, color=TEXT_MUTED, zorder=4.0)

    # 1. UNEXPLORED -> MAPPED (top horizontal)
    _draw_polyline_arrow(
        ax1,
        [(1.15 + sw * 0.5, 2.90), (4.85 - sw * 0.5, 2.90)],
        color=BLUE_BELIEF,
        lw=1.10,
        mutation_scale=10.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        3.00,
        3.20,
        "FindServices\n[services $> 0$]",
        color=BLUE_BELIEF,
        fontsize=6.9,
        bold=True,
        fc=BG_WHITE,
        pad=0.10,
    )

    # 2. UNEXPLORED -> EXHAUSTED (left vertical)
    _draw_polyline_arrow(
        ax1,
        [(1.15, 2.90 - sh * 0.5), (1.15, 0.70 + sh * 0.5)],
        color=CRIMSON_HAZARD,
        lw=1.05,
        mutation_scale=10.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        1.15,
        1.80,
        "FindServices\n[services $= 0$]",
        color=CRIMSON_HAZARD,
        fontsize=6.8,
        fc=BG_WHITE,
        ec=CRIMSON_TINT,
        pad=0.12,
    )

    # 3. MAPPED -> COMPROMISED (right vertical)
    _draw_polyline_arrow(
        ax1,
        [(4.85, 2.90 - sh * 0.5), (4.85, 0.70 + sh * 0.5)],
        color=TEAL_UTILITY,
        lw=1.10,
        mutation_scale=10.5,
        zorder=3.2,
    )
    _draw_label_pill(
        ax1,
        4.85,
        1.80,
        "ExploitService\n[exploit ok]",
        color=TEAL_UTILITY,
        fontsize=6.9,
        bold=True,
        fc=BG_WHITE,
        ec=TEAL_TINT,
        pad=0.12,
    )

    # 4. MAPPED -> EXHAUSTED (clean diagonal dashed transition when all exploits fail)
    diag_arrow = FancyArrowPatch(
        (3.74, 2.36),
        (2.26, 1.24),
        arrowstyle="-|>",
        color=CRIMSON_HAZARD,
        linewidth=1.0,
        linestyle="--",
        mutation_scale=10.0,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=3.15,
    )
    ax1.add_patch(diag_arrow)
    _draw_label_pill(
        ax1,
        3.00,
        1.80,
        "ExploitService\n[all failed]",
        color=CRIMSON_HAZARD,
        fontsize=6.7,
        fc=BG_WHITE,
        ec=BORDER_SUBTLE,
        pad=0.12,
    )

    ax1.set_title("(a) Epistemic Host Lifecycle in $\\mathcal{G}_t$", fontweight="bold", pad=10, color=TEXT_PRIMARY)

    # =========================================================================
    # PANEL (b): Multi-Scale Minimax Hazard Utility U(a) (w in {3, 5, 8})
    # =========================================================================
    # Dynamically evaluate all 5 action bars via SemanticMemoryStore (zero hardcoding)
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

    # Subtle decision-region shading behind bars (explicit low zorder)
    ax2.axvspan(-1.32, 0.0, facecolor=CRIMSON_TINT, alpha=0.45, zorder=0.2)
    ax2.axvspan(0.0, 1.50, facecolor=TEAL_TINT, alpha=0.45, zorder=0.2)
    ax2.grid(axis="x", linestyle="--", color=GRID_LIGHT, alpha=0.85, zorder=0.5)

    # Grouped horizontal bars with explicit zorder
    ax2.barh(
        y + h_bar,
        gains,
        height=h_bar,
        color=SLATE_GAIN,
        alpha=0.92,
        label="$\\alpha \\cdot \\mathrm{ProgressGain}(a \\mid \\mathcal{G}_t)$",
        zorder=2.5,
    )
    ax2.barh(
        y,
        penalties,
        height=h_bar,
        color=CRIMSON_HAZARD,
        alpha=0.88,
        label="$-\\beta \\max_{w\\in\\{3,5,8\\}} R_w(a \\mid \\mathcal{H}_t)$",
        zorder=2.5,
    )
    ax2.barh(
        y - h_bar,
        utilities,
        height=h_bar,
        color=TEAL_UTILITY,
        alpha=0.96,
        label="Net Utility $U(a)$",
        zorder=2.5,
    )

    # Zero threshold axis
    ax2.axvline(0.0, color=INK_DARK, linewidth=0.95, zorder=3.0)

    # Decision-region headers at top of plot
    ax2.text(-0.62, 4.58, "Suppressed ($U < 0$)", ha="center", va="center", fontsize=7.2, fontweight="bold", color=CRIMSON_HAZARD, zorder=3.5)
    ax2.text(0.74, 4.58, "Preferred Interleaving ($U > 0$)", ha="center", va="center", fontsize=7.2, fontweight="bold", color=TEAL_UTILITY, zorder=3.5)

    # Subtle horizontal separator between suppressed (0..1) and preferred (2..4) groups
    ax2.axhline(1.5, color=BORDER_SUBTLE, linewidth=0.85, linestyle=":", zorder=1.2)

    # Crisp tip labels
    for idx, (g_val, p_val, u_val) in enumerate(zip(gains, penalties, utilities)):
        ax2.text(
            g_val + 0.03,
            idx + h_bar,
            f"+{g_val:.2f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=SLATE_GAIN,
            zorder=3.6,
        )
        ax2.text(
            p_val - 0.03,
            idx,
            f"{p_val:.2f}",
            va="center",
            ha="right",
            fontsize=6.5,
            color=CRIMSON_HAZARD,
            zorder=3.6,
        )
        offset = 0.035 if u_val >= 0 else -0.035
        ha = "left" if u_val >= 0 else "right"
        t_col = TEAL_UTILITY if u_val >= 0 else CRIMSON_HAZARD
        ax2.text(
            u_val + offset,
            idx - h_bar,
            f"{u_val:+.2f}",
            va="center",
            ha=ha,
            fontsize=7.5,
            fontweight="bold",
            color=t_col,
            zorder=3.6,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(actions, fontsize=7.8, color=TEXT_PRIMARY)
    ax2.set_ylim(-0.62, 4.84)
    ax2.set_xlim(-1.32, 1.50)
    ax2.set_xlabel("Decision Utility / Component Score ($\\alpha = 1.0,\\, \\beta = 1.2$)", color=TEXT_PRIMARY)
    ax2.set_title("(b) Multi-Scale Minimax Hazard Utility $U(a)$", fontweight="bold", pad=10, color=TEXT_PRIMARY)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(INK_DARK)
    ax2.spines["left"].set_linewidth(0.85)
    ax2.spines["bottom"].set_color(INK_DARK)
    ax2.spines["bottom"].set_linewidth(0.85)

    # Clean horizontal legend below x-axis (zero overlap with bars or labels)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.20),
        ncol=3,
        fontsize=7.3,
        handlelength=1.25,
        columnspacing=1.1,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=BORDER_SUBTLE,
        framealpha=0.98,
    )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.89, bottom=0.21)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"fig2_belief_and_hazard_mechanism.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
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
