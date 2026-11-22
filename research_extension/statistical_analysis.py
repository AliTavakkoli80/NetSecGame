"""
statistical_analysis.py — Paired Statistical Significance & Information-Theoretic Telemetry.

Computes on paired evaluation episodes (`seeds 42..191`):
1. Exact McNemar's test p-values for binary Win Rate differences.
2. Wilcoxon signed-rank test p-values and Cliff's delta effect sizes for Return & Steps.
3. Sliding-window Shannon Action Entropy H_t = -sum p_i log2(p_i) trajectories
   and MITRE ATT&CK phase-arrival step distributions from *_steps.csv traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest, norm, wilcoxon

CANONICAL_PHASES = (
    "ScanNetwork",
    "FindServices",
    "ExploitService",
    "FindData",
    "ExfiltrateData",
)


def exact_mcnemar_test(wins_a: List[bool], wins_b: List[bool]) -> Dict[str, Any]:
    """Compute exact two-sided McNemar's test from paired boolean outcomes."""
    n = min(len(wins_a), len(wins_b))
    b = sum(1 for i in range(n) if wins_a[i] and not wins_b[i])
    c = sum(1 for i in range(n) if not wins_a[i] and wins_b[i])
    discordant = b + c
    p_val = float(binomtest(min(b, c), n=discordant, p=0.5).pvalue) if discordant > 0 else 1.0
    return {"n_paired": n, "a_only_wins": b, "b_only_wins": c, "p_value": round(p_val, 6)}


def two_proportion_z_test(p1_pct: float, n1: int, p2_pct: float, n2: int) -> Dict[str, Any]:
    """
    Compute an unpaired two-sided two-proportion z-test between independent
    published win rates (e.g., vs. Rigaki et al., ESWA 2026 baselines).
    """
    p1 = max(0.0, min(1.0, p1_pct / 100.0))
    p2 = max(0.0, min(1.0, p2_pct / 100.0))
    if n1 <= 0 or n2 <= 0:
        return {"z_stat": 0.0, "p_value": 1.0, "n1": n1, "n2": n2}
    p_pool = (p1 * n1 + p2 * n2) / float(n1 + n2)
    denom = math.sqrt(max(1e-12, p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)))
    z_stat = (p1 - p2) / denom
    p_val = float(2.0 * norm.sf(abs(z_stat)))
    return {
        "n1": n1,
        "n2": n2,
        "p1_percent": round(p1_pct, 2),
        "p2_percent": round(p2_pct, 2),
        "z_stat": round(z_stat, 4),
        "p_value": float(f"{p_val:.6e}"),
    }


# Published Rigaki et al. (ESWA 2026, Tables 4 & 5) baselines.
# Effective sample sizes are either stated (n=150 for Three-Networks Hackphyr)
# or inferred from the published 95% CI half-width (e.g., n=30 for GPT-4, n=200 for Full+Def Hackphyr).
ESWA_2026_PUBLISHED_BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "regime1a_full_nodefender": {
        "Hackphyr_7B": {"win_pct": 89.10, "ci95": 4.74, "n": 166},
        "GPT4": {"win_pct": 100.00, "ci95": 0.00, "n": 30},
    },
    "regime1b_full_defender": {
        "Hackphyr_7B": {"win_pct": 44.00, "ci95": 6.88, "n": 200},
        "GPT4": {"win_pct": 53.33, "ci95": 17.85, "n": 30},
    },
    "regime2a_threenets_nodefender": {
        "Hackphyr_7B": {"win_pct": 50.34, "ci95": 8.00, "n": 150},
        "GPT4": {"win_pct": 82.35, "ci95": 13.64, "n": 30},
    },
    "regime2b_threenets_defender": {
        "Hackphyr_7B": {"win_pct": 23.33, "ci95": 6.77, "n": 150},
        "GPT4": {"win_pct": 36.36, "ci95": 17.21, "n": 30},
    },
}


def compute_eswa_parity_return_stats(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Recompute episode returns under ESWA 2026 reward parity:
    detection penalty = -50 (vs. -10 in local YAML -> -40 adjustment),
    timeout penalty = 0 beyond -1/step (vs. -10 in local YAML -> +10 adjustment).
    """
    if not episodes:
        return {"eswa_parity_mean_return": 0.0, "eswa_parity_return_ci95": 0.0}
    vals = []
    for ep in episodes:
        r = float(ep.get("total_reward", 0.0))
        if ep.get("detected", False):
            r -= 40.0
        if ep.get("timeout", False):
            r += 10.0
        vals.append(r)
    n = len(vals)
    mean_v = float(np.mean(vals))
    sd_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * sd_v / math.sqrt(n) if n > 1 else 0.0
    return {
        "eswa_parity_mean_return": round(mean_v, 2),
        "eswa_parity_return_ci95": round(ci95, 2),
    }


def cliffs_delta(x: List[float], y: List[float]) -> float:
    """Compute non-parametric Cliff's delta effect size in [-1.0, 1.0]."""
    if not x or not y:
        return 0.0
    x_arr = np.asarray(x, dtype=float)[:, None]
    y_arr = np.asarray(y, dtype=float)[None, :]
    greater = float(np.sum(x_arr > y_arr))
    less = float(np.sum(x_arr < y_arr))
    return round((greater - less) / float(len(x) * len(y)), 4)


def paired_wilcoxon_and_delta(vals_a: List[float], vals_b: List[float]) -> Dict[str, Any]:
    """Compute paired Wilcoxon signed-rank p-value and Cliff's delta."""
    n = min(len(vals_a), len(vals_b))
    a = vals_a[:n]
    b = vals_b[:n]
    diffs = [a[i] - b[i] for i in range(n)]
    if all(abs(d) < 1e-12 for d in diffs):
        p_val = 1.0
    else:
        try:
            res = wilcoxon(a, b, zero_method="wilcox")
            p_val = float(res.pvalue)
            if math.isnan(p_val):
                p_val = 1.0
        except ValueError:
            p_val = 1.0
    return {
        "n_paired": n,
        "mean_a": round(float(np.mean(a)), 3) if a else 0.0,
        "mean_b": round(float(np.mean(b)), 3) if b else 0.0,
        "p_value": round(p_val, 6),
        "cliffs_delta": cliffs_delta(a, b),
    }


def shannon_entropy_bits(actions: List[str]) -> float:
    """Compute Shannon entropy H = -sum p_i log2(p_i) in bits."""
    if not actions:
        return 0.0
    total = float(len(actions))
    counts = Counter(actions)
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        if p > 0.0:
            ent -= p * math.log2(p)
    return ent


def compute_entropy_and_phase_arrival(
    steps_csv_path: Path, window_size: int = 5
) -> Dict[str, Any]:
    """
    Parse *_steps.csv to compute:
    - Mean sliding-window Shannon Action Entropy across steps
    - Per-step mean Shannon entropy trajectory (t=1..100)
    - First-arrival step distribution (median, IQR) per canonical kill-chain phase
    """
    if not steps_csv_path.exists():
        return {
            "mean_entropy_bits": 0.0,
            "entropy_trajectory": [],
            "phase_arrival_steps": {},
        }

    episodes: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    with open(steps_csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ep_id = int(row["episode_id"])
            step_num = int(row["step_num"])
            act = str(row["action_type"]).strip()
            episodes[ep_id].append((step_num, act))

    entropies_by_step: Dict[int, List[float]] = defaultdict(list)
    all_step_entropies: List[float] = []
    phase_first_step: Dict[str, List[int]] = defaultdict(list)

    for ep_id, seq in episodes.items():
        seq.sort(key=lambda x: x[0])
        seen_phases: set = set()
        hist: List[str] = []
        for step_num, act in seq:
            hist.append(act)
            win = hist[-window_size:]
            h_t = shannon_entropy_bits(win)
            entropies_by_step[step_num].append(h_t)
            all_step_entropies.append(h_t)
            if act in CANONICAL_PHASES and act not in seen_phases:
                seen_phases.add(act)
                phase_first_step[act].append(step_num)

    trajectory = [
        {
            "step": s,
            "mean_entropy": round(float(np.mean(vals)), 4),
            "ci95_entropy": round(1.96 * float(np.std(vals, ddof=1)) / math.sqrt(len(vals)), 4)
            if len(vals) > 1
            else 0.0,
            "n": len(vals),
        }
        for s, vals in sorted(entropies_by_step.items())
    ]

    phase_stats: Dict[str, Dict[str, Any]] = {}
    for ph in CANONICAL_PHASES:
        vals = phase_first_step.get(ph, [])
        if vals:
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            phase_stats[ph] = {
                "reached_episodes": len(vals),
                "median_step": round(float(med), 2),
                "iqr_steps": round(float(q75 - q25), 2),
                "mean_step": round(float(np.mean(vals)), 2),
            }
        else:
            phase_stats[ph] = {"reached_episodes": 0, "median_step": None, "iqr_steps": None, "mean_step": None}

    return {
        "mean_entropy_bits": round(float(np.mean(all_step_entropies)), 4) if all_step_entropies else 0.0,
        "entropy_trajectory": trajectory,
        "phase_arrival_steps": phase_stats,
    }


def _load_metrics_payload(metrics_json: Path) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    with open(metrics_json, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    summary = payload.get("summary", {})
    episodes_list = payload.get("episodes", [])
    return summary, {int(ep["episode_id"]): ep for ep in episodes_list}


def _backfill_autonomy_from_steps(summary: Dict[str, Any], steps_csv_path: Path) -> Dict[str, Any]:
    """Ensure first_pass_valid_rate_percent and post_reflexion_valid_rate_percent are populated."""
    out = dict(summary)
    if "first_pass_valid_rate_percent" in out and "post_reflexion_valid_rate_percent" in out:
        return out
    if steps_csv_path.exists():
        total = fp = pr = 0
        with open(steps_csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total += 1
                raw_fallback = row.get("raw_slm_valid", "True")
                if str(row.get("first_pass_valid", raw_fallback)).strip().lower() == "true":
                    fp += 1
                if str(row.get("post_reflexion_valid", raw_fallback)).strip().lower() == "true":
                    pr += 1
        if total > 0:
            out.setdefault("first_pass_valid_rate_percent", round(100.0 * fp / total, 2))
            out.setdefault("post_reflexion_valid_rate_percent", round(100.0 * pr / total, 2))
            return out
    raw_val = float(out.get("raw_slm_valid_rate_percent", 0.0))
    out.setdefault("first_pass_valid_rate_percent", raw_val)
    out.setdefault("post_reflexion_valid_rate_percent", raw_val)
    return out


def analyze_suite(suite_root: Path) -> Dict[str, Any]:
    """Run full paired statistical tests, unpaired external comparisons, ESWA-parity returns, entropy, and consolidated summary."""
    report: Dict[str, Any] = {"regimes": {}}
    consolidated_by_regime: Dict[str, List[Dict[str, Any]]] = {}
    all_summaries: List[Dict[str, Any]] = []
    regime_dirs = sorted(p for p in suite_root.iterdir() if p.is_dir())

    canonical_order = [
        "Baseline_RandomAttacker",
        "Baseline_PureHeuristic",
        "Ablation_PureHeuristic_WithStealth",
        "Ablation_SLM_7B_NF4_RawReAct",
        "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth",
        "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth",
        "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth",
    ]
    order_idx = {name: i for i, name in enumerate(canonical_order)}

    for rdir in regime_dirs:
        regime_name = rdir.name
        configs: Dict[str, Dict[int, Dict[str, Any]]] = {}
        summaries_map: Dict[str, Dict[str, Any]] = {}
        entropy_info: Dict[str, Any] = {}
        eswa_parity_returns: Dict[str, Dict[str, float]] = {}

        for mjson in sorted(rdir.glob("*_metrics.json")):
            exp_name = mjson.name.replace("_metrics.json", "")
            summary, ep_map = _load_metrics_payload(mjson)
            steps_csv = rdir / f"{exp_name}_steps.csv"
            summary = _backfill_autonomy_from_steps(summary, steps_csv)
            eswa_stats = compute_eswa_parity_return_stats(list(ep_map.values()))
            summary.update(eswa_stats)
            summaries_map[exp_name] = summary
            configs[exp_name] = ep_map
            eswa_parity_returns[exp_name] = eswa_stats
            entropy_info[exp_name] = compute_entropy_and_phase_arrival(steps_csv)

        ordered_summaries = [
            summaries_map[k]
            for k in sorted(summaries_map.keys(), key=lambda name: (order_idx.get(name, 999), name))
        ]
        consolidated_by_regime[regime_name] = ordered_summaries
        all_summaries.extend(ordered_summaries)

        comparisons: Dict[str, Any] = {}
        anchor = "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth"
        targets = [
            "Ablation_SLM_7B_NF4_BeliefGraph_NoStealth",
            "Baseline_PureHeuristic",
            "Ablation_PureHeuristic_WithStealth",
            "Ablation_SLM_7B_NF4_RawReAct",
            "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth",
        ]
        if anchor in configs:
            ep_a = configs[anchor]
            for tgt in targets:
                if tgt not in configs:
                    continue
                ep_b = configs[tgt]
                shared_ids = sorted(set(ep_a.keys()) & set(ep_b.keys()))
                if not shared_ids:
                    continue
                wins_a = [bool(ep_a[i]["win"]) for i in shared_ids]
                wins_b = [bool(ep_b[i]["win"]) for i in shared_ids]
                ret_a = [float(ep_a[i]["total_reward"]) for i in shared_ids]
                ret_b = [float(ep_b[i]["total_reward"]) for i in shared_ids]
                stp_a = [float(ep_a[i]["steps_taken"]) for i in shared_ids]
                stp_b = [float(ep_b[i]["steps_taken"]) for i in shared_ids]
                comparisons[f"{anchor}__vs__{tgt}"] = {
                    "mcnemar_win": exact_mcnemar_test(wins_a, wins_b),
                    "wilcoxon_return": paired_wilcoxon_and_delta(ret_a, ret_b),
                    "wilcoxon_steps": paired_wilcoxon_and_delta(stp_a, stp_b),
                }

        external_comparisons: Dict[str, Any] = {}
        ext_refs = ESWA_2026_PUBLISHED_BASELINES.get(regime_name, {})
        for our_name in [
            "Proposed_SLM_7B_NF4_BeliefGraph_WithStealth",
            "Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth",
        ]:
            if our_name not in summaries_map:
                continue
            our_win = float(summaries_map[our_name].get("win_rate_percent", 0.0))
            our_n = int(summaries_map[our_name].get("total_episodes", 150))
            for ext_name, ext_data in ext_refs.items():
                external_comparisons[f"{our_name}__vs__{ext_name}"] = two_proportion_z_test(
                    our_win, our_n, float(ext_data["win_pct"]), int(ext_data["n"])
                )

        report["regimes"][regime_name] = {
            "paired_comparisons": comparisons,
            "external_unpaired_z_tests": external_comparisons,
            "eswa_parity_returns": eswa_parity_returns,
            "entropy_and_phases": entropy_info,
        }

    consolidated_path = suite_root / "consolidated_150ep_summary.json"
    with open(consolidated_path, "w", encoding="utf-8") as fh:
        json.dump({"by_regime": consolidated_by_regime, "all_summaries": all_summaries}, fh, indent=2)

    out_path = suite_root / "statistical_analysis_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired statistical tests and entropy trajectories.")
    parser.add_argument("--suite-root", type=str, default="logs/benchmarks/final_paper_150ep")
    args = parser.parse_args()
    report = analyze_suite(Path(args.suite_root))
    print(f"Saved statistical analysis report for {len(report['regimes'])} regimes.")


if __name__ == "__main__":
    main()
