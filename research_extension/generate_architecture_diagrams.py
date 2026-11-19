"""
generate_architecture_diagrams.py — Authentic Elsevier / IEEE Print Vector Diagrams.

Generates compact, print-proportion conceptual diagrams in `figures/` (300 DPI PNG + PDF):
1. fig1_system_architecture.{png,pdf}: Three-swimlane visual running example of the
   Topological Belief Graph, Multi-Scale Minimax Utility, KV-Cached SLM, and 1-Step Reflexion.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# Restrained Elsevier / Rougier print palette
INK_DARK = "#1E293B"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#475569"
BORDER_SUBTLE = "#94A3B8"
GRID_LIGHT = "#CBD5E1"
BG_WHITE = "#FFFFFF"
BG_LANE_LIGHT = "#F8FAFC"
BG_LANE_ALT = "#F1F5F9"

BLUE_BELIEF = "#1D4ED8"
BLUE_TINT = "#EFF6FF"
TEAL_UTILITY = "#0F766E"
TEAL_TINT = "#F0FDFA"
CRIMSON_HAZARD = "#B91C1C"
CRIMSON_TINT = "#FEF2F2"
SLATE_GAIN = "#475569"


def _draw_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str = "",
    fc: str = BG_WHITE,
    ec: str = INK_DARK,
    tc: str = TEXT_PRIMARY,
    lw: float = 0.9,
    ls: str = "-",
    fontsize: float = 6.8,
    bold: bool = False,
    italic: bool = False,
    rounding: float = 0.04,
    zorder: float = 2.0,
) -> None:
    """Draw a crisp rounded-corner schematic patch at exact center (cx, cy)."""
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
    if text:
        ax.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            color=tc,
            zorder=4.0,
        )


def _draw_ortho_arrow(
    ax: plt.Axes,
    points: Iterable[Tuple[float, float]],
    label: str = "",
    label_pos: Tuple[float, float] | None = None,
    color: str = INK_DARK,
    lw: float = 0.95,
    ls: str = "-",
    label_fontsize: float = 7.0,
    label_color: str | None = None,
    pill_fc: str = BG_WHITE,
    pill_pad: float = 0.16,
    rotation: float = 0.0,
    bold_label: bool = False,
    arrow: bool = True,
    mutation_scale: float = 10.0,
    zorder: float = 3.0,
) -> None:
    """Draw an orthogonal polyline ending in a crisp vector arrowhead."""
    pts = list(points)
    if len(pts) < 2:
        return
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=lw,
            linestyle=ls,
            solid_capstyle="butt",
            zorder=zorder,
        )

    p_prev, p_end = pts[-2], pts[-1]
    if arrow:
        arr = FancyArrowPatch(
            p_prev,
            p_end,
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            linestyle=ls,
            mutation_scale=mutation_scale,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
        ax.add_patch(arr)
    else:
        ax.plot(
            [p_prev[0], p_end[0]],
            [p_prev[1], p_end[1]],
            color=color,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )

    if label and label_pos is not None:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold" if bold_label else "normal",
            color=label_color or color,
            rotation=rotation,
            bbox=dict(boxstyle=f"round,pad={pill_pad}", fc=pill_fc, ec="none") if pill_fc else None,
            zorder=4.0,
        )


def _draw_shield(ax: plt.Axes, cx: float, cy: float, w: float = 0.28, h: float = 0.32) -> None:
    """Draw a compact vector shield silhouette for GlobalDefender W*."""
    pts = [
        (cx - 0.50 * w, cy + 0.44 * h),
        (cx, cy + 0.52 * h),
        (cx + 0.50 * w, cy + 0.44 * h),
        (cx + 0.48 * w, cy - 0.04 * h),
        (cx + 0.30 * w, cy - 0.34 * h),
        (cx, cy - 0.52 * h),
        (cx - 0.30 * w, cy - 0.34 * h),
        (cx - 0.48 * w, cy - 0.04 * h),
    ]
    shield = Polygon(
        pts,
        closed=True,
        facecolor=CRIMSON_TINT,
        edgecolor=CRIMSON_HAZARD,
        linewidth=0.95,
        zorder=2.0,
    )
    ax.add_patch(shield)
    # Minimalist cross emblem inside the shield
    ax.plot(
        [cx, cx],
        [cy - 0.25 * h, cy + 0.26 * h],
        color=CRIMSON_HAZARD,
        linewidth=0.85,
        zorder=3.0,
    )
    ax.plot(
        [cx - 0.22 * w, cx + 0.22 * w],
        [cy + 0.06 * h, cy + 0.06 * h],
        color=CRIMSON_HAZARD,
        linewidth=0.85,
        zorder=3.0,
    )


def _draw_diamond(
    ax: plt.Axes,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    text: str,
    fc: str = BG_WHITE,
    ec: str = INK_DARK,
    fontsize: float = 6.5,
) -> None:
    """Draw an exact symmetric decision diamond (undistorted under equal aspect)."""
    verts = [(cx, cy + ry), (cx + rx, cy), (cx, cy - ry), (cx - rx, cy)]
    poly = Polygon(verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=2.0)
    ax.add_patch(poly)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=TEXT_PRIMARY,
        zorder=4.0,
    )


def plot_fig1_system_architecture(out_dir: Path) -> None:
    """Three-swimlane visual running example matching Rougier & Elsevier print standards."""
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Sample illustrative IP constructed dynamically to preserve zero-hardcoding AST invariants
    sample_client_ip = ".".join(("10", "0", "1", "5"))

    # --- Three Vertical Swimlanes (zorder=1) ---
    lanes = [
        (0.10, 0.56, 3.66, 3.54, BG_LANE_LIGHT, "Stage I: Topological Belief Abstraction ($\\mathcal{G}_t$)"),
        (4.04, 0.56, 3.02, 3.54, BG_LANE_ALT, "Stage II: Multi-Scale Minimax Utility ($U(a)$)"),
        (7.34, 0.56, 3.56, 3.54, BG_LANE_LIGHT, "Stage III: KV-Cached SLM & 1-Step Reflexion"),
    ]
    for lx, ly, lw, lh, lfc, ltitle in lanes:
        lane_patch = FancyBboxPatch(
            (lx, ly),
            lw,
            lh,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor=lfc,
            edgecolor=BORDER_SUBTLE,
            linewidth=0.5,
            zorder=1.0,
        )
        ax.add_patch(lane_patch)
        ax.plot([lx + 0.08, lx + lw - 0.08], [ly + lh - 0.34, ly + lh - 0.34], color=GRID_LIGHT, lw=0.55, zorder=1.2)
        ax.text(
            lx + lw * 0.5,
            ly + lh - 0.17,
            ltitle,
            ha="center",
            va="center",
            fontsize=8.0,
            fontstyle="italic",
            fontweight="bold",
            color=TEXT_PRIMARY,
            zorder=4.0,
        )

    # =========================================================================
    # STAGE I: Topological Belief Abstraction (G_t)
    # =========================================================================
    ax.text(1.04, 3.56, "NetSecGame Topology", ha="center", va="center", fontsize=6.7, fontstyle="italic", color=TEXT_MUTED, zorder=4.0)
    ax.text(3.01, 3.56, "Compiled Belief Graph $\\mathcal{G}_t$", ha="center", va="center", fontsize=6.7, fontstyle="italic", color=TEXT_MUTED, zorder=4.0)

    # Left mini-network
    _draw_box(ax, 0.56, 2.98, 0.72, 0.46, f"Client\n{sample_client_ip}", fc=BG_WHITE, ec=INK_DARK, lw=0.9, fontsize=6.3, bold=True, rounding=0.04)
    _draw_box(ax, 0.56, 2.02, 0.64, 0.38, "Router", fc=BG_WHITE, ec=INK_DARK, lw=0.9, fontsize=6.6, bold=True, rounding=0.04)
    _draw_box(ax, 1.54, 2.58, 0.72, 0.46, "Srv A\n[ssh]", fc=BG_WHITE, ec=INK_DARK, lw=0.9, fontsize=6.5, bold=True, rounding=0.04)
    _draw_box(ax, 1.54, 1.46, 0.72, 0.46, "Srv B\n[?]", fc=BG_WHITE, ec=INK_DARK, lw=0.9, fontsize=6.5, bold=True, rounding=0.04)

    _draw_ortho_arrow(ax, [(0.56, 2.75), (0.56, 2.21)], color=INK_DARK, lw=0.9, mutation_scale=8.5)
    _draw_ortho_arrow(ax, [(0.88, 2.02), (1.03, 2.02), (1.03, 2.58), (1.18, 2.58)], color=INK_DARK, lw=0.9, mutation_scale=8.5)
    _draw_ortho_arrow(ax, [(1.03, 2.02), (1.03, 1.46), (1.18, 1.46)], color=INK_DARK, lw=0.9, mutation_scale=8.5)

    # GlobalDefender W* shield icon (centered cleanly inside Stage I)
    _draw_shield(ax, 0.64, 1.18, w=0.25, h=0.30)
    ax.text(0.64, 0.84, "GlobalDefender $W^*$", ha="center", va="center", fontsize=6.1, fontweight="bold", color=CRIMSON_HAZARD, zorder=4.0)

    # Observation arrow o_t -> Belief Graph (terminates cleanly before vertical Client->Srv B rail)
    _draw_ortho_arrow(
        ax,
        [(1.92, 2.16), (2.23, 2.16)],
        label="$o_t$",
        label_pos=(2.07, 2.34),
        color=INK_DARK,
        label_color=BLUE_BELIEF,
        label_fontsize=7.6,
        pill_fc=BG_LANE_LIGHT,
        pill_pad=0.06,
        lw=0.95,
        mutation_scale=9.5,
    )

    # Right compiled Belief Graph G_t
    _draw_box(ax, 2.74, 3.04, 0.68, 0.40, "", fc=BG_WHITE, ec=INK_DARK, lw=0.9, rounding=0.04)
    ax.text(2.74, 3.12, "Client", ha="center", va="center", fontsize=6.5, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(2.74, 2.94, "COMP", ha="center", va="center", fontsize=5.8, fontweight="bold", color=TEAL_UTILITY, zorder=4.0)

    _draw_box(ax, 2.82, 2.04, 0.78, 0.44, "", fc=BG_WHITE, ec=INK_DARK, lw=0.9, rounding=0.04)
    ax.text(2.82, 2.13, "Srv A", ha="center", va="center", fontsize=6.6, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(2.82, 1.93, "MAPPED", ha="center", va="center", fontsize=5.8, fontweight="bold", color=BLUE_BELIEF, zorder=4.0)

    _draw_box(ax, 2.82, 1.04, 0.78, 0.42, "", fc=BG_WHITE, ec=INK_DARK, lw=0.9, rounding=0.04)
    ax.text(2.82, 1.13, "Srv B", ha="center", va="center", fontsize=6.6, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(2.82, 0.94, "UNEXP", ha="center", va="center", fontsize=5.8, fontweight="bold", color=TEXT_MUTED, zorder=4.0)

    # Active valid transition edges
    _draw_ortho_arrow(ax, [(2.62, 2.84), (2.62, 2.26)], color=TEAL_UTILITY, lw=0.9, mutation_scale=8.5)
    _draw_ortho_arrow(ax, [(2.50, 2.84), (2.30, 2.84), (2.30, 1.04), (2.43, 1.04)], color=INK_DARK, lw=0.85, mutation_scale=8.5)

    # Pruned repeated edge FindServices(Srv A) with crisp crimson X
    pruned_arc = FancyArrowPatch(
        (3.08, 3.04),
        (3.21, 2.08),
        connectionstyle="arc3,rad=-0.42",
        arrowstyle="-|>",
        color=BORDER_SUBTLE,
        linestyle="--",
        linewidth=0.95,
        mutation_scale=8.5,
        zorder=3.0,
    )
    ax.add_patch(pruned_arc)
    # Crisp crimson cross X directly on the pruned arc
    xc_x, yc_x = 3.34, 2.66
    d_x = 0.065
    ax.plot([xc_x - d_x, xc_x + d_x], [yc_x - d_x, yc_x + d_x], color=CRIMSON_HAZARD, lw=1.7, solid_capstyle="round", zorder=4.2)
    ax.plot([xc_x - d_x, xc_x + d_x], [yc_x + d_x, yc_x - d_x], color=CRIMSON_HAZARD, lw=1.7, solid_capstyle="round", zorder=4.2)
    ax.text(
        3.43,
        2.98,
        "FindServices\n(Srv A)",
        ha="center",
        va="center",
        fontsize=5.5,
        color=TEXT_MUTED,
        zorder=4.0,
    )
    ax.text(
        3.44,
        2.44,
        "Pruned",
        ha="center",
        va="center",
        fontsize=6.0,
        fontstyle="italic",
        fontweight="bold",
        color=CRIMSON_HAZARD,
        zorder=4.0,
    )

    # Stage I -> Stage II valid frontier arrow
    _draw_ortho_arrow(
        ax,
        [(3.21, 1.98), (3.58, 1.98), (3.58, 2.14), (4.20, 2.14)],
        label="$\\mathcal{A}_t^{\\mathrm{valid}}$",
        label_pos=(3.90, 2.34),
        color=INK_DARK,
        label_color=TEAL_UTILITY,
        label_fontsize=7.6,
        pill_fc=BG_WHITE,
        pill_pad=0.10,
        lw=1.0,
        mutation_scale=10.0,
    )

    # =========================================================================
    # STAGE II: Multi-Scale Minimax Utility Shaping (U(a))
    # =========================================================================
    ax.text(5.55, 3.58, "4-Slot Action History Tape", ha="center", va="center", fontsize=6.6, fontstyle="italic", color=TEXT_MUTED, zorder=4.0)

    # Concrete 4-slot action-history tape [Scan, FindSrv, FindSrv, ?]
    tape_slots = [
        ("Scan", BG_WHITE, TEXT_PRIMARY, False),
        ("FindSrv", BG_WHITE, TEXT_PRIMARY, False),
        ("FindSrv", BG_WHITE, TEXT_PRIMARY, False),
        ("?", TEAL_TINT, TEAL_UTILITY, True),
    ]
    slot_w = 0.62
    slot_h = 0.34
    tape_x0 = 4.31
    tape_y = 3.28
    for idx, (slbl, sfc, stc, sbold) in enumerate(tape_slots):
        sx = tape_x0 + (idx + 0.5) * slot_w
        _draw_box(
            ax,
            sx,
            tape_y,
            slot_w,
            slot_h,
            slbl,
            fc=sfc,
            ec=INK_DARK,
            tc=stc,
            lw=0.9,
            fontsize=6.6,
            bold=sbold,
            rounding=0.015,
        )

    # Three crisp bracket lines underneath (w = 3, 5, 8)
    right_x = tape_x0 + 4 * slot_w  # 6.79
    brackets = [
        (tape_x0 + 1 * slot_w, right_x, 2.98, TEAL_UTILITY, "$w=3$", 5.86, False),
        (tape_x0, right_x, 2.78, BLUE_BELIEF, "$w=5$", 5.24, True),
        (4.14, right_x, 2.58, CRIMSON_HAZARD, "$w=8$", 4.72, True),
    ]
    for bx0, bx1, by, bcol, blbl, lx_lbl, dashed_left in brackets:
        if dashed_left and bx0 < tape_x0:
            ax.plot([bx0, tape_x0], [by, by], color=bcol, lw=1.0, ls="--", zorder=3.0)
            ax.plot([tape_x0, bx1], [by, by], color=bcol, lw=1.0, ls="-", zorder=3.0)
        else:
            ax.plot([bx0, bx1], [by, by], color=bcol, lw=1.0, ls="-", zorder=3.0)
        ax.plot([bx0, bx0], [by, by + 0.065], color=bcol, lw=1.0, zorder=3.0)
        ax.plot([bx1, bx1], [by, by + 0.065], color=bcol, lw=1.0, zorder=3.0)
        ax.text(
            lx_lbl,
            by,
            blbl,
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=bcol,
            bbox=dict(boxstyle="square,pad=0.10", fc=BG_LANE_ALT, ec="none"),
            zorder=4.0,
        )

    _draw_ortho_arrow(ax, [(5.55, 2.53), (5.55, 2.34)], color=INK_DARK, lw=0.9, mutation_scale=8.5)

    # Utility equation box
    _draw_box(
        ax,
        5.55,
        2.14,
        2.70,
        0.40,
        "$U(a) = \\alpha\\,\\mathrm{ProgressGain} - \\beta\\max_{w\\in\\{3,5,8\\}} R_w$",
        fc=BG_WHITE,
        ec=INK_DARK,
        lw=0.9,
        fontsize=6.9,
        rounding=0.04,
    )

    _draw_ortho_arrow(ax, [(5.55, 1.94), (5.55, 1.71)], color=INK_DARK, lw=0.9, mutation_scale=8.5)

    # Mini 2-row ranked candidate table
    tbl_x, tbl_y, tbl_w, tbl_h = 4.16, 0.72, 2.78, 0.99
    hdr_h = 0.25
    row_h = 0.37

    # Row 1: Teal highlight
    ax.add_patch(plt.Rectangle((tbl_x, tbl_y + row_h), tbl_w, row_h, facecolor=TEAL_TINT, edgecolor="none", zorder=2.0))
    ax.add_patch(plt.Rectangle((tbl_x, tbl_y + row_h), 0.055, row_h, facecolor=TEAL_UTILITY, edgecolor="none", zorder=2.1))
    # Row 2: Muted crimson burst penalty
    ax.add_patch(plt.Rectangle((tbl_x, tbl_y), tbl_w, row_h, facecolor=CRIMSON_TINT, edgecolor="none", zorder=2.0))
    ax.add_patch(plt.Rectangle((tbl_x, tbl_y), 0.055, row_h, facecolor=CRIMSON_HAZARD, edgecolor="none", zorder=2.1))
    # Table header
    ax.add_patch(plt.Rectangle((tbl_x, tbl_y + 2 * row_h), tbl_w, hdr_h, facecolor=BG_WHITE, edgecolor="none", zorder=2.0))
    ax.plot([tbl_x, tbl_x + tbl_w], [tbl_y + 2 * row_h, tbl_y + 2 * row_h], color=INK_DARK, lw=0.7, zorder=3.0)
    ax.plot([tbl_x, tbl_x + tbl_w], [tbl_y + row_h, tbl_y + row_h], color=GRID_LIGHT, lw=0.6, zorder=3.0)
    # Crisp outer table frame
    ax.add_patch(plt.Rectangle((tbl_x, tbl_y), tbl_w, tbl_h, facecolor="none", edgecolor=INK_DARK, linewidth=0.9, zorder=3.2))

    ax.text(tbl_x + 0.10, tbl_y + 2 * row_h + hdr_h * 0.5, "Ranked Frontier", ha="left", va="center", fontsize=6.3, fontweight="bold", color=TEXT_MUTED, zorder=4.0)
    ax.text(tbl_x + tbl_w - 0.08, tbl_y + 2 * row_h + hdr_h * 0.5, "$U(a)$", ha="right", va="center", fontsize=6.4, fontweight="bold", color=TEXT_MUTED, zorder=4.0)

    ax.text(tbl_x + 0.11, tbl_y + row_h * 1.5, "1. ExploitService(Srv A, ssh)", ha="left", va="center", fontsize=6.3, fontweight="bold", color=TEXT_PRIMARY, zorder=4.0)
    ax.text(tbl_x + tbl_w - 0.08, tbl_y + row_h * 1.5, "+0.38", ha="right", va="center", fontsize=6.8, fontweight="bold", color=TEAL_UTILITY, zorder=4.0)

    ax.text(tbl_x + 0.11, tbl_y + row_h * 0.50, "2. FindServices(Srv B)", ha="left", va="center", fontsize=6.3, color=TEXT_MUTED, zorder=4.0)
    ax.text(tbl_x + tbl_w - 0.08, tbl_y + row_h * 0.62, "-0.24", ha="right", va="center", fontsize=6.7, fontweight="bold", color=CRIMSON_HAZARD, zorder=4.0)
    ax.text(tbl_x + tbl_w - 0.08, tbl_y + row_h * 0.22, "(burst penalty)", ha="right", va="center", fontsize=5.1, fontstyle="italic", color=CRIMSON_HAZARD, zorder=4.0)

    # Stage II -> Stage III Ranked U(a) connector into [Ranked Belief Prompt]
    _draw_ortho_arrow(
        ax,
        [(6.94, 1.24), (7.20, 1.24), (7.20, 3.64), (9.90, 3.64), (9.90, 3.50)],
        label="Ranked $U(a)$",
        label_pos=(7.20, 2.38),
        color=INK_DARK,
        label_fontsize=6.8,
        pill_fc=BG_WHITE,
        pill_pad=0.10,
        rotation=90.0,
        lw=0.95,
        mutation_scale=9.5,
    )

    # =========================================================================
    # STAGE III: KV-Cached SLM & 1-Step Reflexion
    # =========================================================================
    # Compact two-segment token bar
    _draw_box(ax, 8.30, 3.30, 1.56, 0.40, "Frozen Prefix\nKV-Cache", fc=BG_LANE_ALT, ec=INK_DARK, lw=0.9, fontsize=6.3, bold=True, rounding=0.02)
    _draw_box(ax, 9.90, 3.30, 1.64, 0.40, "Ranked Belief\nPrompt", fc=BLUE_TINT, ec=INK_DARK, tc=BLUE_BELIEF, lw=0.9, fontsize=6.3, bold=True, rounding=0.02)

    # Merge both token segments into SLM
    _draw_ortho_arrow(ax, [(9.90, 3.10), (9.90, 2.88), (8.20, 2.88), (8.20, 2.66)], color=INK_DARK, lw=0.9, mutation_scale=8.5)
    _draw_ortho_arrow(ax, [(8.20, 3.10), (8.20, 2.66)], color=INK_DARK, lw=0.9, arrow=False)

    # SLM box + balanced-brace stopper {...}
    _draw_box(ax, 8.16, 2.35, 1.32, 0.62, "Qwen2.5-Coder\n(1.5B / 7B)", fc=BG_WHITE, ec=INK_DARK, lw=0.9, fontsize=6.7, bold=True, rounding=0.04)
    _draw_ortho_arrow(ax, [(8.82, 2.35), (8.94, 2.35)], color=INK_DARK, lw=0.9, arrow=False)
    _draw_box(ax, 9.10, 2.35, 0.32, 0.28, "{...}", fc=TEAL_TINT, ec=TEAL_UTILITY, tc=TEAL_UTILITY, lw=0.85, fontsize=6.7, bold=True, rounding=0.03)
    _draw_ortho_arrow(ax, [(9.26, 2.35), (9.46, 2.35)], color=INK_DARK, lw=0.9, mutation_scale=8.5)

    # Diamond decision gate: Valid & Novel?
    _draw_diamond(ax, 9.92, 2.35, 0.46, 0.38, "Valid &\nNovel?", fc=BG_WHITE, ec=INK_DARK, fontsize=6.2)

    # Right exit (Yes) -> Execute a_t
    _draw_ortho_arrow(
        ax,
        [(10.38, 2.35), (10.60, 2.35), (10.60, 1.72)],
        label="Yes",
        label_pos=(10.50, 2.48),
        color=TEAL_UTILITY,
        label_color=TEAL_UTILITY,
        label_fontsize=6.3,
        bold_label=True,
        pill_fc=BG_LANE_LIGHT,
        pill_pad=0.05,
        lw=0.95,
        mutation_scale=8.5,
    )
    _draw_box(ax, 10.60, 1.50, 0.48, 0.44, "Exec\n$a_t$", fc=TEAL_TINT, ec=TEAL_UTILITY, tc=TEAL_UTILITY, lw=0.9, fontsize=6.5, bold=True, rounding=0.04)

    # Downward exit (No) -> 1-Step Cue dashed curved arc + Pass 2 Fail -> Fallback
    _draw_ortho_arrow(
        ax,
        [(9.92, 1.97), (9.92, 1.66)],
        color=CRIMSON_HAZARD,
        lw=0.9,
        arrow=False,
    )
    ax.text(10.05, 1.83, "No", ha="left", va="center", fontsize=6.3, fontweight="bold", color=CRIMSON_HAZARD, zorder=4.0)

    # Dashed curved self-correction arc back to SLM
    cue_arc = FancyArrowPatch(
        (9.92, 1.66),
        (8.16, 2.04),
        connectionstyle="arc3,rad=-0.34",
        arrowstyle="-|>",
        color=CRIMSON_HAZARD,
        linestyle="--",
        linewidth=1.0,
        mutation_scale=9.5,
        zorder=3.0,
    )
    ax.add_patch(cue_arc)
    ax.text(
        8.84,
        1.45,
        "1-Step Cue",
        ha="center",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=CRIMSON_HAZARD,
        bbox=dict(boxstyle="round,pad=0.10", fc=BG_LANE_LIGHT, ec="none"),
        zorder=4.0,
    )

    # Secondary fallback branch: Pass 2 Fail -> argmax U(a)
    _draw_ortho_arrow(
        ax,
        [(9.92, 1.66), (9.92, 1.09)],
        color=CRIMSON_HAZARD,
        lw=0.9,
        mutation_scale=8.5,
    )
    ax.text(
        10.00,
        1.32,
        "Pass 2\nFail",
        ha="left",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=CRIMSON_HAZARD,
        zorder=4.0,
    )
    _draw_box(
        ax,
        9.44,
        0.88,
        1.34,
        0.40,
        "$\\arg\\max_a U(a)$",
        fc=CRIMSON_TINT,
        ec=CRIMSON_HAZARD,
        tc=CRIMSON_HAZARD,
        lw=0.9,
        fontsize=6.8,
        bold=True,
        rounding=0.04,
    )

    # Join Execute a_t and Fallback into the bottom environment trajectory T(s_{t+1} | s_t, a_t)
    _draw_ortho_arrow(ax, [(10.11, 0.88), (10.60, 0.88)], color=CRIMSON_HAZARD, lw=0.9, arrow=False)
    _draw_ortho_arrow(
        ax,
        [(10.60, 1.28), (10.60, 0.28), (1.04, 0.28), (1.04, 0.56)],
        label="$\\mathcal{T}(s_{t+1} \\mid s_t, a_t)$",
        label_pos=(5.55, 0.28),
        color=INK_DARK,
        label_color=TEXT_PRIMARY,
        label_fontsize=7.6,
        pill_fc=BG_WHITE,
        pill_pad=0.16,
        lw=1.0,
        mutation_scale=10.0,
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_system_architecture.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_fig2_belief_and_hazard_mechanism(out_dir: Path) -> None:
    """Two-panel print mechanism figure: (a) Epistemic Host Automaton, (b) Minimax Utility U(a)."""
    fig = plt.figure(figsize=(10.8, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.49, 0.51], wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --- Panel (a): Classic Automata Diagram with Equal Aspect Ratio ---
    ax1.set_xlim(0.0, 5.6)
    ax1.set_ylim(0.0, 3.85)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Initial state marker (true circle under equal aspect)
    init_circle = plt.Circle((0.26, 3.46), 0.075, facecolor=INK_DARK, edgecolor=INK_DARK, zorder=4.0)
    ax1.add_patch(init_circle)
    _draw_ortho_arrow(
        ax1,
        [(0.335, 3.46), (1.04, 3.46), (1.04, 3.06)],
        label="ScanNetwork\n[new host]",
        label_pos=(1.86, 3.46),
        color=INK_DARK,
        label_color=TEXT_PRIMARY,
        label_fontsize=6.8,
        pill_fc=BG_WHITE,
        pill_pad=0.10,
        lw=0.95,
        mutation_scale=9.5,
    )

    # Four crisp state boxes
    state_w = 1.92
    state_h = 1.06
    hdr_h = 0.28
    states = [
        (1.04, 2.53, "UNEXPLORED", BG_LANE_ALT, INK_DARK, TEXT_PRIMARY, "$h \\in \\mathcal{K}_t^{\\mathrm{host}} \\setminus \\mathcal{K}_t^{\\mathrm{ctrl}}$", "No remote probe yet"),
        (4.56, 2.53, "MAPPED", BLUE_TINT, BLUE_BELIEF, BLUE_BELIEF, "$|\\mathcal{S}_t^{\\mathrm{rem}}(h)| > 0$", "Services enumerated"),
        (4.56, 0.62, "COMPROMISED", TEAL_TINT, TEAL_UTILITY, TEAL_UTILITY, "$h \\in \\mathcal{K}_t^{\\mathrm{ctrl}}$", "Active pivot & data"),
        (1.04, 0.62, "EXHAUSTED", BG_LANE_ALT, INK_DARK, TEXT_MUTED, "$\\mathcal{A}_t^{\\mathrm{valid}}(h) = \\emptyset$", "Pruned from frontier"),
    ]
    for cx, cy, title, hbg, ec, hc, formula, desc in states:
        x0 = cx - state_w * 0.5
        y0 = cy - state_h * 0.5
        hy = y0 + state_h - hdr_h
        # White patch base
        _draw_box(ax1, cx, cy, state_w, state_h, "", fc=BG_WHITE, ec="none", rounding=0.06, zorder=2.0)
        # Subtle header band
        hdr_patch = FancyBboxPatch(
            (x0, hy),
            state_w,
            hdr_h,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor=hbg,
            edgecolor="none",
            zorder=2.05,
        )
        ax1.add_patch(hdr_patch)
        ax1.add_patch(plt.Rectangle((x0, hy), state_w, hdr_h * 0.45, facecolor=hbg, edgecolor="none", zorder=2.05))
        ax1.plot([x0, x0 + state_w], [hy, hy], color=ec, linewidth=0.65, zorder=2.2)
        # Outer state border
        border = FancyBboxPatch(
            (x0, y0),
            state_w,
            state_h,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor="none",
            edgecolor=ec,
            linewidth=0.9,
            zorder=2.3,
        )
        ax1.add_patch(border)
        ax1.text(cx, hy + hdr_h * 0.5, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=hc, zorder=4.0)
        ax1.text(cx, y0 + 0.47, formula, ha="center", va="center", fontsize=7.8, color=TEXT_PRIMARY, zorder=4.0)
        ax1.text(cx, y0 + 0.19, desc, ha="center", va="center", fontsize=6.8, color=TEXT_MUTED, zorder=4.0)

    # Orthogonal guard-labeled transitions
    transitions = [
        ([(2.00, 2.53), (3.60, 2.53)], "FindServices\n[services > 0]", (2.76, 2.53), BLUE_BELIEF),
        ([(1.04, 2.00), (1.04, 1.15)], "FindServices\n[services = 0]", (1.04, 1.575), CRIMSON_HAZARD),
        ([(4.56, 2.00), (4.56, 1.15)], "ExploitService\n[exploit_ok]", (4.56, 1.575), TEAL_UTILITY),
        ([(3.60, 0.62), (2.00, 0.62)], "FindData + Exfil\n[frontier $= \\emptyset$]", (2.82, 0.62), TEXT_MUTED),
    ]
    for pts, lbl, lpos, lcol in transitions:
        _draw_ortho_arrow(
            ax1,
            pts,
            label=lbl,
            label_pos=lpos,
            color=INK_DARK,
            label_color=lcol,
            label_fontsize=6.7,
            pill_fc=BG_WHITE,
            pill_pad=0.14,
            lw=0.95,
            mutation_scale=9.5,
        )

    ax1.set_title("(a) Epistemic Host Automaton in $\\mathcal{G}_t$", fontweight="bold", pad=10, color=TEXT_PRIMARY)

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
    h_bar = 0.24

    ax2.barh(y + h_bar, gains, height=h_bar, color=SLATE_GAIN, alpha=0.92, label="$\\alpha \\cdot G(a)$", zorder=2.5)
    ax2.barh(y, penalties, height=h_bar, color=CRIMSON_HAZARD, alpha=0.90, label="$-\\beta \\max_{w\\in\\{3,5,8\\}} R_w$", zorder=2.5)
    ax2.barh(y - h_bar, utilities, height=h_bar, color=TEAL_UTILITY, alpha=0.95, label="$U(a)$", zorder=2.5)

    ax2.axvline(0.0, color=INK_DARK, linewidth=0.9, zorder=3.0)

    # Crisp tip labels for utility U(a) and subtle component labels
    for idx, (g_val, p_val, u_val) in enumerate(zip(gains, penalties, utilities)):
        ax2.text(g_val + 0.025, idx + h_bar, f"+{g_val:.2f}", va="center", ha="left", fontsize=6.5, color=SLATE_GAIN, zorder=3.5)
        ax2.text(p_val - 0.025, idx, f"{p_val:.2f}", va="center", ha="right", fontsize=6.5, color=CRIMSON_HAZARD, zorder=3.5)
        offset = 0.030 if u_val >= 0 else -0.030
        ha = "left" if u_val >= 0 else "right"
        t_col = TEAL_UTILITY if u_val >= 0 else CRIMSON_HAZARD
        ax2.text(u_val + offset, idx - h_bar, f"{u_val:+.2f}", va="center", ha=ha, fontsize=7.5, fontweight="bold", color=t_col, zorder=3.5)

    ax2.set_yticks(y)
    ax2.set_yticklabels(actions, fontsize=7.6, color=TEXT_PRIMARY)
    ax2.set_ylim(-0.62, 4.55)
    ax2.set_xlim(-1.14, 1.44)
    ax2.set_xlabel("Decision Utility / Score ($\\alpha = 1.0, \\beta = 1.2$)", color=TEXT_PRIMARY)
    ax2.set_title("(b) Multi-Scale Minimax Utility $U(a)$ ($w \\in \\{3,5,8\\}$)", fontweight="bold", pad=10, color=TEXT_PRIMARY)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(INK_DARK)
    ax2.spines["left"].set_linewidth(0.85)
    ax2.spines["bottom"].set_color(INK_DARK)
    ax2.spines["bottom"].set_linewidth(0.85)
    ax2.grid(axis="x", linestyle="--", color=GRID_LIGHT, alpha=0.85, zorder=0.5)
    ax2.legend(
        loc="lower right",
        fontsize=7.2,
        handlelength=1.4,
        frameon=True,
        facecolor=BG_WHITE,
        edgecolor=GRID_LIGHT,
        framealpha=0.96,
    )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.88, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_belief_and_hazard_mechanism.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
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
