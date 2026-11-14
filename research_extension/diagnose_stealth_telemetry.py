"""
diagnose_stealth_telemetry.py — Diagnostic inspector for SIEM detections & wrapper overrides.

Parses step-level and episode-level benchmark CSVs to report:
1. Per-episode SIEM detection triggers (episode, step, triggering action, W=5 window).
2. Breakdown of wrapper overrides (invalid JSON/params vs. redundant/repeated vs. other).
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path
from typing import Dict, List


def analyze_steps_csv(steps_csv_path: str, window_size: int = 5) -> Dict[str, object]:
    """Analyze a single *_steps.csv file for detection triggers and override causes."""
    rows_by_ep: Dict[int, List[Dict[str, str]]] = {}
    with open(steps_csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ep_id = int(row["episode_id"])
            rows_by_ep.setdefault(ep_id, []).append(row)

    detections = []
    det_action_counts: Counter = Counter()
    override_causes: Counter = Counter()
    total_steps = 0
    total_overrides = 0

    for ep_id in sorted(rows_by_ep.keys()):
        ep_rows = sorted(rows_by_ep[ep_id], key=lambda r: int(r["step_num"]))
        history: List[str] = []
        for row in ep_rows:
            total_steps += 1
            act = row["action_type"]
            history.append(act)
            if row.get("detected", "").strip().lower() == "true":
                window = history[-window_size:]
                detections.append(
                    {
                        "episode_id": ep_id,
                        "step_num": int(row["step_num"]),
                        "action_type": act,
                        "window": window,
                        "stealth_risk": float(row.get("stealth_risk_score", 0.0)),
                    }
                )
                det_action_counts[act] += 1

            if row.get("was_overridden", "").strip().lower() == "true":
                total_overrides += 1
                raw_valid = row.get("raw_slm_valid", "").strip().lower() == "true"
                valid_syntax = row.get("valid_syntax", "").strip().lower() == "true"
                redundant = row.get("redundant_action", "").strip().lower() == "true"
                if not valid_syntax:
                    override_causes["invalid_json_syntax"] += 1
                elif not raw_valid:
                    override_causes["invalid_params_or_illegal_state_action"] += 1
                elif redundant:
                    override_causes["redundant_already_executed_action"] += 1
                else:
                    override_causes["stealth_soft_masked_or_other"] += 1

    return {
        "file": str(steps_csv_path),
        "total_steps": total_steps,
        "total_overrides": total_overrides,
        "override_rate_percent": round(100.0 * total_overrides / total_steps, 2) if total_steps else 0.0,
        "detections": detections,
        "detection_counts_by_action": dict(det_action_counts),
        "override_causes": dict(override_causes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SIEM detections and wrapper overrides from step CSVs.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="logs/benchmarks/regime2_interleaved_10ep/regime2_defender/*WithStealth*_steps.csv",
        help="Glob pattern matching *_steps.csv files",
    )
    args = parser.parse_args()

    matched = sorted(glob.glob(args.pattern))
    if not matched:
        print(f"No files matched pattern: {args.pattern}")
        return

    for path in matched:
        report = analyze_steps_csv(path)
        print(f"\n=== DIAGNOSTIC REPORT: {Path(path).name} ===")
        print(
            f"Total Steps: {report['total_steps']} | Overridden: {report['total_overrides']} "
            f"({report['override_rate_percent']:.2f}%)"
        )
        print("Override Cause Breakdown:")
        for cause, cnt in sorted(report["override_causes"].items()):
            pct = (100.0 * cnt / report["total_overrides"]) if report["total_overrides"] else 0.0
            print(f"  - {cause}: {cnt} ({pct:.1f}% of overrides)")
        print(f"Detected Episodes ({len(report['detections'])}):")
        print(f"  Triggers by ActionType: {report['detection_counts_by_action']}")
        for det in report["detections"]:
            print(
                f"  - Ep {det['episode_id']} Step {det['step_num']}: {det['action_type']} "
                f"(risk={det['stealth_risk']:.2f}) | W=5 Window={det['window']}"
            )


if __name__ == "__main__":
    main()
