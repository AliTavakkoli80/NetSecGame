"""
metrics_logger.py — Quantitative Telemetry, Autonomy Tracking, and Statistical Reporting.

Records step-level and episode-level evaluation metrics including:
- Mission Success (Win Rate %), SIEM Detection Rate %, Average Return, Average Steps
- Statistical Dispersion: Standard Deviation (SD) and 95% Confidence Intervals (+/- 1.96 * SE)
- LLM Autonomy Telemetry: Raw SLM Valid Action Rate (%) and Wrapper Override Rate (%)
- Temporal Stealth Pacing Telemetry: Mean Stealth Risk Score
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _compute_stats(values: List[float]) -> TupleLike:
    """Return (mean, sample_sd, ci95_half_width) across observations."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = sum(values) / float(n)
    if n == 1:
        return mean_val, 0.0, 0.0
    variance = sum((v - mean_val) ** 2 for v in values) / float(n - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(float(n))
    ci95 = 1.96 * se
    return mean_val, sd, ci95


TupleLike = tuple  # Type alias kept simple for 3.12 compatibility


@dataclass
class StepMetric:
    """Granular telemetry recorded at each environment step."""

    episode_id: int
    step_num: int
    action_type: str
    target: str = ""
    valid_syntax: bool = True
    valid_action: bool = True
    redundant_action: bool = False
    reward: float = 0.0
    latency_ms: float = 0.0
    tokens_generated: int = 0
    detected: bool = False
    raw_action_str: str = ""
    raw_slm_valid: bool = True
    first_pass_valid: bool = True
    post_reflexion_valid: bool = True
    was_overridden: bool = False
    stealth_risk_score: float = 0.0
    stealth_deferred_count: int = 0


@dataclass
class EpisodeMetric:
    """Aggregated episode telemetry including autonomy and stealth indicators."""

    episode_id: int
    model_name: str
    scenario: str
    steps_taken: int = 0
    total_reward: float = 0.0
    win: bool = False
    detected: bool = False
    timeout: bool = False
    syntax_errors: int = 0
    invalid_actions: int = 0
    redundant_actions: int = 0
    raw_slm_valid_count: int = 0
    first_pass_valid_count: int = 0
    post_reflexion_valid_count: int = 0
    override_count: int = 0
    total_stealth_risk: float = 0.0
    total_stealth_deferred: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    raw_slm_valid_rate_percent: float = 0.0
    first_pass_valid_rate_percent: float = 0.0
    post_reflexion_valid_rate_percent: float = 0.0
    override_rate_percent: float = 0.0
    mean_stealth_risk: float = 0.0
    steps: List[StepMetric] = field(default_factory=list)


class MetricsLogger:
    """Structured telemetry logger with 95% CI computation and JSON/CSV export."""

    def __init__(self, output_dir: str = "logs/benchmarks", experiment_name: str = "exp") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.episodes: List[EpisodeMetric] = []
        self._current_episode: Optional[EpisodeMetric] = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_episode(self, episode_id: int, model_name: str, scenario: str) -> None:
        """Begin telemetry collection for a fresh episode."""
        self._current_episode = EpisodeMetric(
            episode_id=episode_id,
            model_name=model_name,
            scenario=scenario,
        )

    def record_step(
        self,
        step_num: int,
        action_type: str,
        target: str = "",
        valid_syntax: bool = True,
        valid_action: bool = True,
        redundant_action: bool = False,
        reward: float = 0.0,
        latency_ms: float = 0.0,
        tokens_generated: int = 0,
        detected: bool = False,
        raw_action_str: str = "",
        raw_slm_valid: bool = True,
        first_pass_valid: Optional[bool] = None,
        post_reflexion_valid: Optional[bool] = None,
        was_overridden: bool = False,
        stealth_risk_score: float = 0.0,
        stealth_deferred_count: int = 0,
    ) -> None:
        """Record telemetry for a single executed step in the active episode."""
        if self._current_episode is None:
            raise RuntimeError("start_episode() must be called before record_step().")

        fpv = bool(raw_slm_valid) if first_pass_valid is None else bool(first_pass_valid)
        prv = bool(raw_slm_valid) if post_reflexion_valid is None else bool(post_reflexion_valid)
        step = StepMetric(
            episode_id=self._current_episode.episode_id,
            step_num=step_num,
            action_type=action_type,
            target=target,
            valid_syntax=valid_syntax,
            valid_action=valid_action,
            redundant_action=redundant_action,
            reward=float(reward),
            latency_ms=float(latency_ms),
            tokens_generated=int(tokens_generated),
            detected=bool(detected),
            raw_action_str=raw_action_str,
            raw_slm_valid=bool(raw_slm_valid),
            first_pass_valid=fpv,
            post_reflexion_valid=prv,
            was_overridden=bool(was_overridden),
            stealth_risk_score=float(stealth_risk_score),
            stealth_deferred_count=int(stealth_deferred_count),
        )
        ep = self._current_episode
        ep.steps.append(step)
        ep.steps_taken += 1
        ep.total_reward += float(reward)
        ep.total_latency_ms += float(latency_ms)
        ep.total_tokens += int(tokens_generated)
        ep.total_stealth_risk += float(stealth_risk_score)
        ep.total_stealth_deferred += int(stealth_deferred_count)
        if not valid_syntax:
            ep.syntax_errors += 1
        if not valid_action:
            ep.invalid_actions += 1
        if redundant_action:
            ep.redundant_actions += 1
        if raw_slm_valid:
            ep.raw_slm_valid_count += 1
        if fpv:
            ep.first_pass_valid_count += 1
        if prv:
            ep.post_reflexion_valid_count += 1
        if was_overridden:
            ep.override_count += 1

    def end_episode(self, win: bool = False, detected: bool = False, timeout: bool = False) -> EpisodeMetric:
        """Finalize episode counters, compute per-episode autonomy rates, and store it."""
        if self._current_episode is None:
            raise RuntimeError("No active episode to finalize.")

        ep = self._current_episode
        ep.win = bool(win)
        ep.detected = bool(detected)
        ep.timeout = bool(timeout)

        if ep.steps_taken > 0:
            ep.raw_slm_valid_rate_percent = round(100.0 * ep.raw_slm_valid_count / ep.steps_taken, 2)
            ep.first_pass_valid_rate_percent = round(100.0 * ep.first_pass_valid_count / ep.steps_taken, 2)
            ep.post_reflexion_valid_rate_percent = round(100.0 * ep.post_reflexion_valid_count / ep.steps_taken, 2)
            ep.override_rate_percent = round(100.0 * ep.override_count / ep.steps_taken, 2)
            ep.mean_stealth_risk = round(ep.total_stealth_risk / ep.steps_taken, 4)

        self.episodes.append(ep)
        self._current_episode = None
        return ep

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregate means, SDs, and 95% confidence intervals across episodes."""
        n = len(self.episodes)
        if n == 0:
            return {"total_episodes": 0}

        win_vals = [100.0 if e.win else 0.0 for e in self.episodes]
        det_vals = [100.0 if e.detected else 0.0 for e in self.episodes]
        timeout_vals = [100.0 if e.timeout else 0.0 for e in self.episodes]
        return_vals = [float(e.total_reward) for e in self.episodes]
        step_vals = [float(e.steps_taken) for e in self.episodes]
        raw_valid_vals = [float(e.raw_slm_valid_rate_percent) for e in self.episodes]
        override_vals = [float(e.override_rate_percent) for e in self.episodes]
        stealth_vals = [float(e.mean_stealth_risk) for e in self.episodes]

        win_mean, win_sd, win_ci = _compute_stats(win_vals)
        det_mean, det_sd, det_ci = _compute_stats(det_vals)
        to_mean, _, _ = _compute_stats(timeout_vals)
        ret_mean, ret_sd, ret_ci = _compute_stats(return_vals)
        stp_mean, stp_sd, stp_ci = _compute_stats(step_vals)

        # Also compute global micro-averaged step rates alongside macro CI across episodes.
        total_steps = sum(e.steps_taken for e in self.episodes)
        total_latency = sum(e.total_latency_ms for e in self.episodes)
        total_tokens = sum(e.total_tokens for e in self.episodes)
        total_raw_valid = sum(e.raw_slm_valid_count for e in self.episodes)
        total_first_pass = sum(e.first_pass_valid_count for e in self.episodes)
        total_post_reflexion = sum(e.post_reflexion_valid_count for e in self.episodes)
        total_overrides = sum(e.override_count for e in self.episodes)
        total_stealth = sum(e.total_stealth_risk for e in self.episodes)

        _, raw_sd, raw_ci = _compute_stats(raw_valid_vals)
        _, ovr_sd, ovr_ci = _compute_stats(override_vals)
        _, stl_sd, stl_ci = _compute_stats(stealth_vals)

        raw_valid_micro = (100.0 * total_raw_valid / total_steps) if total_steps > 0 else 0.0
        first_pass_micro = (100.0 * total_first_pass / total_steps) if total_steps > 0 else 0.0
        post_reflexion_micro = (100.0 * total_post_reflexion / total_steps) if total_steps > 0 else 0.0
        override_micro = (100.0 * total_overrides / total_steps) if total_steps > 0 else 0.0
        stealth_micro = (total_stealth / total_steps) if total_steps > 0 else 0.0

        return {
            "experiment_name": self.experiment_name,
            "model_name": self.episodes[0].model_name,
            "scenario": self.episodes[0].scenario,
            "total_episodes": n,
            "win_rate_percent": round(win_mean, 2),
            "win_rate_sd": round(win_sd, 2),
            "win_rate_ci95": round(win_ci, 2),
            "detection_rate_percent": round(det_mean, 2),
            "detection_rate_sd": round(det_sd, 2),
            "detection_rate_ci95": round(det_ci, 2),
            "timeout_rate_percent": round(to_mean, 2),
            "mean_return": round(ret_mean, 2),
            "std_return": round(ret_sd, 2),
            "return_ci95": round(ret_ci, 2),
            "mean_steps": round(stp_mean, 2),
            "std_steps": round(stp_sd, 2),
            "steps_ci95": round(stp_ci, 2),
            "raw_slm_valid_rate_percent": round(raw_valid_micro, 2),
            "raw_slm_valid_rate_sd": round(raw_sd, 2),
            "raw_slm_valid_rate_ci95": round(raw_ci, 2),
            "first_pass_valid_rate_percent": round(first_pass_micro, 2),
            "post_reflexion_valid_rate_percent": round(post_reflexion_micro, 2),
            "wrapper_override_rate_percent": round(override_micro, 2),
            "wrapper_override_rate_sd": round(ovr_sd, 2),
            "wrapper_override_rate_ci95": round(ovr_ci, 2),
            "mean_stealth_risk_score": round(stealth_micro, 4),
            "stealth_risk_sd": round(stl_sd, 4),
            "stealth_risk_ci95": round(stl_ci, 4),
            "avg_latency_ms_per_step": round(total_latency / total_steps, 2) if total_steps else 0.0,
            "avg_tokens_per_step": round(total_tokens / total_steps, 2) if total_steps else 0.0,
            "total_syntax_errors": sum(e.syntax_errors for e in self.episodes),
            "total_invalid_actions": sum(e.invalid_actions for e in self.episodes),
            "total_redundant_actions": sum(e.redundant_actions for e in self.episodes),
            "total_raw_slm_valid_steps": total_raw_valid,
            "total_overridden_steps": total_overrides,
            "total_stealth_deferred": sum(e.total_stealth_deferred for e in self.episodes),
            "avg_stealth_deferred_per_step": round(
                sum(e.total_stealth_deferred for e in self.episodes) / total_steps, 2
            ) if total_steps else 0.0,
            "total_steps": total_steps,
        }

    def export_json(self, filename: Optional[str] = None) -> Path:
        """Write full summary and per-episode breakdown to JSON."""
        fname = filename or f"{self.experiment_name}_{self.timestamp}.json"
        path = self.output_dir / fname
        payload = {
            "summary": self.get_summary(),
            "episodes": [
                {k: v for k, v in asdict(ep).items() if k != "steps"}
                for ep in self.episodes
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def append_episode_jsonl(self, ep: EpisodeMetric, jsonl_path: Path) -> None:
        """Flush a single completed episode (with step trace) to JSONL for crash-safe resume."""
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(ep)) + "\n")
            fh.flush()

    def load_episodes_jsonl(self, jsonl_path: Path) -> int:
        """Load previously flushed episodes from JSONL and return the count loaded."""
        if not jsonl_path.exists():
            return 0
        loaded: Dict[int, EpisodeMetric] = {}
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    steps = [StepMetric(**st) for st in raw.pop("steps", [])]
                    ep = EpisodeMetric(**raw, steps=steps)
                    loaded[ep.episode_id] = ep
                except Exception:
                    continue
        self.episodes = [loaded[k] for k in sorted(loaded.keys())]
        return len(self.episodes)

    def export_csv(self, filename: Optional[str] = None) -> Path:
        """Write episode-level summary CSV and granular step-level trace CSV."""
        base = filename.replace(".csv", "") if filename else f"{self.experiment_name}_{self.timestamp}"
        ep_path = self.output_dir / f"{base}_episodes.csv"
        step_path = self.output_dir / f"{base}_steps.csv"

        ep_fields = [
            "episode_id",
            "model_name",
            "scenario",
            "steps_taken",
            "total_reward",
            "win",
            "detected",
            "timeout",
            "syntax_errors",
            "invalid_actions",
            "redundant_actions",
            "raw_slm_valid_count",
            "override_count",
            "raw_slm_valid_rate_percent",
            "override_rate_percent",
            "mean_stealth_risk",
            "total_stealth_deferred",
            "total_latency_ms",
            "total_tokens",
        ]
        with open(ep_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ep_fields)
            writer.writeheader()
            for ep in self.episodes:
                row = {k: getattr(ep, k) for k in ep_fields}
                writer.writerow(row)

        step_fields = [
            "episode_id",
            "step_num",
            "action_type",
            "target",
            "valid_syntax",
            "valid_action",
            "redundant_action",
            "raw_slm_valid",
            "first_pass_valid",
            "post_reflexion_valid",
            "was_overridden",
            "stealth_risk_score",
            "stealth_deferred_count",
            "reward",
            "latency_ms",
            "tokens_generated",
            "detected",
            "raw_action_str",
        ]
        with open(step_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=step_fields)
            writer.writeheader()
            for ep in self.episodes:
                for st in ep.steps:
                    writer.writerow(asdict(st))

        return ep_path
