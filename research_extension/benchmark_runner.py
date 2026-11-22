"""
benchmark_runner.py — Protocol-Parity NetSecGame Evaluation Harness.

Evaluates:
1. RandomAttacker (uniform valid-action baseline matching Rigaki et al., ESWA 2026)
2. PureHeuristicBaseline (domain-agnostic non-LLM BFS/DFS rule agent)
3. ModernSLMAttacker variants (Qwen2.5-Coder-1.5B, Qwen2.5-Coder-7B-NF4, and Config B Raw ReAct)

Enforces paired per-episode topology/IP/goal randomization (`randomize_topology=True`,
identical episode seeds across agents), dynamic `goal_description` parsing, 100-step horizon
parity, GPU VRAM cleanup between models, and 95% CI + Raw SLM Autonomy telemetry reporting.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing
import pickle
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from netsecgame import Action, AgentRole, BaseAgent, generate_valid_actions
from netsecgame.game.worlds.NetSecGame import NetSecGame
from netsecgame.game_components import AgentStatus
from NetSecGameAgents.agents.llm_utils import create_action_from_response

from .base_agent import HuggingFaceSLMBackend, ModernSLMAttacker, SLMConfig
from .memory_module import SemanticMemoryStore, normalize_ip
from .metrics_logger import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BenchmarkRunner")


def _run_server_process(host: str, port: int, task_config_file: str, seed: int) -> None:
    """Target entrypoint for running NetSecGame server in an isolated subprocess."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.getLogger().setLevel(logging.WARNING)
    game_server = NetSecGame(host, port, task_config_file, seed=seed)
    game_server.run()


def _load_max_steps_from_yaml(config_path: str, default_max_steps: int = 100) -> int:
    """Inspect scenario YAML for max_steps; default to 100 (ESWA 2026 parity)."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if "max_steps" in cfg:
            return int(cfg["max_steps"])
        coord = cfg.get("coordinator", {})
        agents = coord.get("agents", {}) if isinstance(coord, dict) else {}
        attacker = agents.get("Attacker", {}) if isinstance(agents, dict) else {}
        if isinstance(attacker, dict) and "max_steps" in attacker:
            return int(attacker["max_steps"])
        env = cfg.get("env", {})
        if isinstance(env, dict) and "max_steps" in env:
            return int(env["max_steps"])
    except Exception as exc:
        logger.warning("Could not parse max_steps from %s (%s); defaulting to %d.", config_path, exc, default_max_steps)
    return default_max_steps


def _parse_episode_terminal_flags(obs: Any, step_count: int, max_steps: int) -> Tuple[bool, bool, bool]:
    """
    Determine (win, detected, timeout) strictly from NetSecGame Observation.info.
    """
    info = getattr(obs, "info", {}) or {}
    end_reason = info.get("end_reason", "")
    end_str = str(getattr(end_reason, "value", end_reason)).lower()

    win = (end_reason == AgentStatus.Success) or ("success" in end_str)
    timeout = (end_reason == AgentStatus.TimeoutReached) or ("timeout" in end_str) or (not getattr(obs, "end", False) and step_count >= max_steps)
    detected = (not win and not timeout) and (
        (end_reason == AgentStatus.Fail) or ("fail" in end_str) or ("detect" in end_str)
    )
    return win, detected, timeout


def _extract_target_label(params: Dict[str, Any]) -> str:
    """Extract human-readable target descriptor from canonical action parameters."""
    if not params:
        return ""
    for key in ("target_host", "target_network", "source_host"):
        if key in params and params[key]:
            return str(params[key])
    return ""


class PureHeuristicBaseline:
    """
    Domain-agnostic non-LLM deterministic BFS/DFS rule baseline.
    Uses SemanticMemoryStore strictly from live observations without any hardcoded
    IPs, subnets, service names, or data identifiers.
    """

    PHASE_ORDER = {
        "ExfiltrateData": 0,
        "FindData": 1,
        "ExploitService": 2,
        "FindServices": 3,
        "ScanNetwork": 4,
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, enable_stealth_pacing: bool = False) -> None:
        self.agent = BaseAgent(host, port, AgentRole.Attacker)
        self.semantic_memory = SemanticMemoryStore(stealth_window=5, enable_stealth_pacing=enable_stealth_pacing)

    def connect(self) -> Any:
        for attempt in range(4):
            res = self.agent.register()
            if getattr(self.agent, "_socket", None) is not None:
                return res
            time.sleep(2.0)
        raise RuntimeError("PureHeuristicBaseline failed to connect to NetSecGame server.")

    def reset_episode(self, seed: int) -> Any:
        self.semantic_memory.reset_episode()
        return self.agent.request_game_reset(
            request_trajectory=False,
            randomize_topology=True,
            seed=seed,
        )

    def select_action(self, obs: Any, goal_description: str = "") -> Tuple[Optional[Action], Dict[str, Any], float]:
        """Select highest-priority unexplored valid action in canonical BFS/DFS order."""
        self.semantic_memory.update_from_observation(obs, goal_description=goal_description)
        candidates = self.semantic_memory.get_valid_action_space(obs.state, include_executed=False)
        if not candidates:
            candidates = self.semantic_memory.get_valid_action_space(obs.state, include_executed=True)
        if not candidates:
            return None, {}, 0.0

        def _priority_key(cand: Dict[str, Any]) -> Tuple[int, int, str]:
            act = cand["action"]
            params = cand["parameters"]
            base_prio = self.PHASE_ORDER.get(act, 99)

            sub_prio = 1
            if act == "ExfiltrateData":
                if self.semantic_memory.goal_target_ip and params.get("target_host") == self.semantic_memory.goal_target_ip:
                    sub_prio = 0
            elif act == "FindServices":
                th = params.get("target_host", "")
                if th not in self.semantic_memory.probed_service_hosts and th not in self.semantic_memory.controlled_hosts:
                    sub_prio = 0
            elif act == "ExploitService":
                th = params.get("target_host", "")
                if th not in self.semantic_memory.controlled_hosts:
                    sub_prio = 0
            else:
                sub_prio = 0

            return (base_prio, sub_prio, str(sorted(params.items())))

        if self.semantic_memory.enable_stealth_pacing:
            # SemanticMemoryStore.get_valid_action_space already ranks candidates by descending U(a)
            candidates_sorted = candidates
        else:
            candidates_sorted = sorted(candidates, key=_priority_key)
        for cand in candidates_sorted:
            ok, netsec_act = create_action_from_response(cand, obs.state)
            if ok and netsec_act is not None:
                risk = self.semantic_memory.score_action_stealth_risk(cand["action"], cand["parameters"])
                return netsec_act, cand, risk
        return None, {}, 0.0

    def step(self, action: Action) -> Any:
        return self.agent.make_step(action)

    def terminate(self) -> None:
        self.agent.terminate_connection()


class BenchmarkHarness:
    """Orchestrates NetSecGame server lifecycle and reproducible multi-agent evaluations."""

    def __init__(
        self,
        task_config: str,
        host: str = "127.0.0.1",
        port: int = 9000,
        output_dir: str = "logs/benchmarks",
        seed: int = 42,
    ) -> None:
        self.task_config = str(Path(task_config).resolve())
        self.scenario_name = Path(task_config).stem
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.seed = int(seed)
        self.max_steps = _load_max_steps_from_yaml(self.task_config, default_max_steps=100)
        self._server_proc: Optional[multiprocessing.Process] = None

    def start_server(self) -> None:
        """Spawn the NetSecGame TCP server subprocess."""
        if self._server_proc is not None and self._server_proc.is_alive():
            self.stop_server()
        ctx = multiprocessing.get_context("spawn")
        self._server_proc = ctx.Process(
            target=_run_server_process,
            args=(self.host, self.port, self.task_config, self.seed),
            daemon=True,
        )
        self._server_proc.start()
        # Allow NetSecGame TCP server time to bind without making a dummy probe connection
        # that would temporarily occupy NetSecGame's single-agent max_connections slot.
        time.sleep(4.5)
        if not self._server_proc.is_alive():
            raise RuntimeError(f"NetSecGame server exited prematurely for {self.task_config}")
        logger.info("NetSecGame server active on %s:%d (max_steps=%d)", self.host, self.port, self.max_steps)

    def stop_server(self) -> None:
        """Terminate the NetSecGame server subprocess cleanly."""
        if self._server_proc is not None:
            if self._server_proc.is_alive():
                self._server_proc.terminate()
                self._server_proc.join(timeout=5)
                if self._server_proc.is_alive():
                    self._server_proc.kill()
            self._server_proc = None
            self.port += 1
            time.sleep(0.5)

    def run_slm_agent(
        self,
        config: SLMConfig,
        experiment_name: str,
        num_episodes: int = 5,
        backend: Optional[HuggingFaceSLMBackend] = None,
        deterministic_filenames: bool = False,
    ) -> Tuple[Dict[str, Any], MetricsLogger]:
        """Run `num_episodes` randomized evaluations with a configured ModernSLMAttacker."""
        owns_backend = backend is None
        active_backend = backend or HuggingFaceSLMBackend(config)
        metrics = MetricsLogger(output_dir=self.output_dir, experiment_name=experiment_name)
        jsonl_path = Path(self.output_dir) / f"{experiment_name}_episodes.jsonl" if deterministic_filenames else None
        start_ep = 0
        if jsonl_path is not None and jsonl_path.exists():
            loaded_cnt = metrics.load_episodes_jsonl(jsonl_path)
            if loaded_cnt > 0:
                start_ep = loaded_cnt
                logger.info(
                    "[%s] Resuming from JSONL checkpoint at episode %d/%d",
                    experiment_name, start_ep, num_episodes,
                )
        attacker: Optional[ModernSLMAttacker] = None

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if start_ep < num_episodes:
            self.start_server()
            try:
                attacker = ModernSLMAttacker(config=config, backend=active_backend, host=self.host, port=self.port)
                for attempt in range(4):
                    obs = attacker.connect()
                    if getattr(attacker, "_socket", None) is not None:
                        break
                    time.sleep(2.0)
                for ep in range(start_ep, num_episodes):
                    ep_seed = self.seed + ep
                    attacker.reset_memory()
                    obs = attacker.reset_environment(randomize_topology=True, seed=ep_seed)
                    goal_desc = str((getattr(obs, "info", {}) or {}).get("goal_description", "") or "")
                    init_nets = sorted(str(n) for n in getattr(obs.state, "known_networks", ()))
                    init_ctrl = sorted(normalize_ip(h) for h in getattr(obs.state, "controlled_hosts", ()))
                    if num_episodes < 25 or ep < 3:
                        logger.info(
                            "[%s | ep=%d seed=%d] Start hosts=%s nets=%s | Goal=%s",
                            experiment_name, ep, ep_seed, init_ctrl, init_nets, goal_desc,
                        )

                    metrics.start_episode(episode_id=ep, model_name=config.model_name, scenario=self.scenario_name)
                    step_num = 0
                    win = False
                    detected = False
                    timeout = False

                    while not getattr(obs, "end", False) and step_num < self.max_steps:
                        step_num += 1
                        goal_desc = str((getattr(obs, "info", {}) or {}).get("goal_description", "") or goal_desc)
                        action, action_dict, latency_ms, tokens = attacker.plan_next_action(obs, goal_description=goal_desc)
                        meta = attacker.last_decision_metadata

                        if action is None:
                            # Unmasked invalid SLM action: count penalized step without mutating state.
                            reward = -1.0
                            target_lbl = _extract_target_label(meta.get("raw_action", {}).get("parameters", {}))
                            attacker.episodic_memory.add_step(
                                step_num=step_num,
                                action_type=meta.get("raw_action", {}).get("action", "INVALID"),
                                parameters=meta.get("raw_action", {}).get("parameters", {}),
                                outcome_status="invalid",
                                reward=reward,
                                state_delta_summary="Invalid action rejected before execution",
                                detected=False,
                                stealth_risk=0.0,
                            )
                            metrics.record_step(
                                step_num=step_num,
                                action_type=meta.get("raw_action", {}).get("action", "INVALID"),
                                target=target_lbl,
                                valid_syntax=meta["raw_valid_syntax"],
                                valid_action=False,
                                redundant_action=False,
                                reward=reward,
                                latency_ms=latency_ms,
                                tokens_generated=tokens,
                                detected=False,
                                raw_action_str=meta["raw_response"],
                                raw_slm_valid=False,
                                was_overridden=False,
                                stealth_risk_score=0.0,
                            )
                            if step_num >= self.max_steps:
                                timeout = True
                                break
                            continue

                        prev_hosts = len(getattr(obs.state, "known_hosts", ()))
                        prev_ctrl = len(getattr(obs.state, "controlled_hosts", ()))
                        prev_srvs = sum(len(v) for v in (getattr(obs.state, "known_services", {}) or {}).values())
                        prev_data = sum(len(v) for v in (getattr(obs.state, "known_data", {}) or {}).values())

                        next_obs = attacker.step(action)
                        reward = float(getattr(next_obs, "reward", 0.0))

                        new_hosts = len(getattr(next_obs.state, "known_hosts", ()))
                        new_ctrl = len(getattr(next_obs.state, "controlled_hosts", ()))
                        new_srvs = sum(len(v) for v in (getattr(next_obs.state, "known_services", {}) or {}).values())
                        new_data = sum(len(v) for v in (getattr(next_obs.state, "known_data", {}) or {}).values())
                        state_changed = (
                            new_hosts > prev_hosts
                            or new_ctrl > prev_ctrl
                            or new_srvs > prev_srvs
                            or new_data > prev_data
                            or reward > 0
                        )

                        win, detected, timeout = _parse_episode_terminal_flags(next_obs, step_num, self.max_steps)
                        outcome_str = "success" if state_changed or reward >= 0 else "no_effect"
                        delta_summary = (
                            f"hosts:{prev_hosts}->{new_hosts}, ctrl:{prev_ctrl}->{new_ctrl}, "
                            f"srvs:{prev_srvs}->{new_srvs}, data:{prev_data}->{new_data}"
                        )

                        act_type = action_dict.get("action", action.action_type.value)
                        act_params = action_dict.get("parameters", {})
                        stealth_risk = float(meta.get("stealth_risk_score", 0.0))

                        attacker.semantic_memory.record_action_outcome(
                            action_type=act_type,
                            parameters=act_params,
                            outcome_status=outcome_str,
                            reward=reward,
                            state_changed=state_changed,
                        )
                        attacker.episodic_memory.add_step(
                            step_num=step_num,
                            action_type=act_type,
                            parameters=act_params,
                            outcome_status=outcome_str,
                            reward=reward,
                            state_delta_summary=delta_summary,
                            detected=detected,
                            stealth_risk=stealth_risk,
                        )

                        metrics.record_step(
                            step_num=step_num,
                            action_type=act_type,
                            target=_extract_target_label(act_params),
                            valid_syntax=meta["raw_valid_syntax"],
                            valid_action=True,
                            redundant_action=meta["raw_redundant"],
                            reward=reward,
                            latency_ms=latency_ms,
                            tokens_generated=tokens,
                            detected=detected,
                            raw_action_str=meta["raw_response"],
                            raw_slm_valid=meta["raw_slm_valid"],
                            first_pass_valid=meta.get("first_pass_valid", meta["raw_slm_valid"]),
                            post_reflexion_valid=meta.get("post_reflexion_valid", meta["raw_slm_valid"]),
                            was_overridden=meta["was_overridden"],
                            stealth_risk_score=stealth_risk,
                            stealth_deferred_count=int(meta.get("stealth_deferred_count", 0)),
                        )
                        obs = next_obs

                    ep_metric = metrics.end_episode(win=win, detected=detected, timeout=timeout)
                    if jsonl_path is not None:
                        metrics.append_episode_jsonl(ep_metric, jsonl_path)
                    if num_episodes < 25 or (ep + 1) % 25 == 0 or (ep + 1) == num_episodes:
                        so_far = metrics.get_summary()
                        logger.info(
                            "[%s] Progress %d/%d | EpWin=%s Det=%s Steps=%d Ret=%.1f | CumWin=%.1f%% CumDet=%.1f%% FirstPass=%.1f%% PostRef=%.1f%%",
                            experiment_name,
                            ep + 1,
                            num_episodes,
                            ep_metric.win,
                            ep_metric.detected,
                            ep_metric.steps_taken,
                            ep_metric.total_reward,
                            so_far.get("win_rate_percent", 0.0),
                            so_far.get("detection_rate_percent", 0.0),
                            so_far.get("first_pass_valid_rate_percent", 0.0),
                            so_far.get("post_reflexion_valid_rate_percent", 0.0),
                        )
            finally:
                try:
                    attacker.terminate()
                except Exception:
                    pass
                self.stop_server()

        if owns_backend:
            del active_backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
        json_name = f"{experiment_name}_metrics.json" if deterministic_filenames else None
        csv_name = experiment_name if deterministic_filenames else None
        json_path = metrics.export_json(filename=json_name)
        metrics.export_csv(filename=csv_name)
        summary = metrics.get_summary()
        if torch.cuda.is_available():
            summary["peak_vram_mb"] = peak_vram_mb
            with open(json_path, "r+", encoding="utf-8") as fh:
                payload = json.load(fh)
                payload["summary"]["peak_vram_mb"] = peak_vram_mb
                fh.seek(0)
                json.dump(payload, fh, indent=2)
                fh.truncate()
        return summary, metrics

    def run_baseline_random_agent(
        self,
        experiment_name: str = "Baseline_RandomAttacker",
        num_episodes: int = 5,
        deterministic_filenames: bool = False,
    ) -> Tuple[Dict[str, Any], MetricsLogger]:
        """Run uniform RandomAttacker baseline on identical per-episode randomized seeds."""
        metrics = MetricsLogger(output_dir=self.output_dir, experiment_name=experiment_name)
        agent: Optional[BaseAgent] = None
        self.start_server()

        try:
            agent = BaseAgent(self.host, self.port, AgentRole.Attacker)
            for attempt in range(4):
                agent.register()
                if getattr(agent, "_socket", None) is not None:
                    break
                time.sleep(2.0)
            if getattr(agent, "_socket", None) is None:
                raise RuntimeError("RandomAttacker failed to connect to NetSecGame server.")
            for ep in range(num_episodes):
                ep_seed = self.seed + ep
                rng = random.Random(ep_seed)
                obs = agent.request_game_reset(request_trajectory=False, randomize_topology=True, seed=ep_seed)
                metrics.start_episode(episode_id=ep, model_name="RandomAttacker", scenario=self.scenario_name)

                step_num = 0
                win = detected = timeout = False
                while not getattr(obs, "end", False) and step_num < self.max_steps:
                    step_num += 1
                    valid_set = list(generate_valid_actions(obs.state))
                    if not valid_set:
                        break
                    chosen = rng.choice(sorted(valid_set, key=lambda a: str(a)))
                    next_obs = agent.make_step(chosen)
                    reward = float(getattr(next_obs, "reward", 0.0))
                    win, detected, timeout = _parse_episode_terminal_flags(next_obs, step_num, self.max_steps)
                    metrics.record_step(
                        step_num=step_num,
                        action_type=chosen.action_type.value,
                        target=str(chosen.parameters.get("target_host", chosen.parameters.get("target_network", ""))),
                        reward=reward,
                        detected=detected,
                        raw_slm_valid=True,
                        was_overridden=False,
                    )
                    obs = next_obs
                metrics.end_episode(win=win, detected=detected, timeout=timeout)
                if num_episodes >= 25 and ((ep + 1) % 250 == 0 or (ep + 1) == num_episodes):
                    so_far = metrics.get_summary()
                    logger.info(
                        "[%s] Progress %d/%d | CumWin=%.2f%% CumDet=%.2f%%",
                        experiment_name,
                        ep + 1,
                        num_episodes,
                        so_far.get("win_rate_percent", 0.0),
                        so_far.get("detection_rate_percent", 0.0),
                    )
        finally:
            try:
                agent.terminate_connection()
            except Exception:
                pass
            self.stop_server()

        json_name = f"{experiment_name}_metrics.json" if deterministic_filenames else None
        csv_name = experiment_name if deterministic_filenames else None
        metrics.export_json(filename=json_name)
        metrics.export_csv(filename=csv_name)
        return metrics.get_summary(), metrics

    def run_baseline_heuristic_agent(
        self,
        experiment_name: str = "Baseline_PureHeuristic",
        num_episodes: int = 5,
        deterministic_filenames: bool = False,
        enable_stealth_pacing: bool = False,
    ) -> Tuple[Dict[str, Any], MetricsLogger]:
        """Run deterministic domain-agnostic PureHeuristicBaseline under identical episode seeds."""
        metrics = MetricsLogger(output_dir=self.output_dir, experiment_name=experiment_name)
        jsonl_path = Path(self.output_dir) / f"{experiment_name}_episodes.jsonl" if deterministic_filenames else None
        if jsonl_path is not None and jsonl_path.exists():
            jsonl_path.unlink()
        baseline: Optional[PureHeuristicBaseline] = None
        self.start_server()

        try:
            baseline = PureHeuristicBaseline(
                host=self.host, port=self.port, enable_stealth_pacing=enable_stealth_pacing
            )
            baseline.connect()
            model_lbl = "PureHeuristicWithStealth" if enable_stealth_pacing else "PureHeuristicBaseline"
            for ep in range(num_episodes):
                ep_seed = self.seed + ep
                obs = baseline.reset_episode(seed=ep_seed)
                goal_desc = str((getattr(obs, "info", {}) or {}).get("goal_description", "") or "")
                metrics.start_episode(episode_id=ep, model_name=model_lbl, scenario=self.scenario_name)

                step_num = 0
                win = detected = timeout = False
                while not getattr(obs, "end", False) and step_num < self.max_steps:
                    step_num += 1
                    goal_desc = str((getattr(obs, "info", {}) or {}).get("goal_description", "") or goal_desc)
                    action, action_dict, risk = baseline.select_action(obs, goal_description=goal_desc)
                    if action is None:
                        break
                    prev_score = (
                        len(getattr(obs.state, "known_hosts", ())),
                        len(getattr(obs.state, "controlled_hosts", ())),
                        sum(len(v) for v in (getattr(obs.state, "known_services", {}) or {}).values()),
                        sum(len(v) for v in (getattr(obs.state, "known_data", {}) or {}).values()),
                    )
                    next_obs = baseline.step(action)
                    reward = float(getattr(next_obs, "reward", 0.0))
                    new_score = (
                        len(getattr(next_obs.state, "known_hosts", ())),
                        len(getattr(next_obs.state, "controlled_hosts", ())),
                        sum(len(v) for v in (getattr(next_obs.state, "known_services", {}) or {}).values()),
                        sum(len(v) for v in (getattr(next_obs.state, "known_data", {}) or {}).values()),
                    )
                    baseline.semantic_memory.record_action_outcome(
                        action_type=action_dict["action"],
                        parameters=action_dict["parameters"],
                        outcome_status="success" if (new_score > prev_score or reward > 0) else "no_effect",
                        reward=reward,
                        state_changed=(new_score > prev_score or reward > 0),
                    )
                    win, detected, timeout = _parse_episode_terminal_flags(next_obs, step_num, self.max_steps)
                    metrics.record_step(
                        step_num=step_num,
                        action_type=action_dict["action"],
                        target=_extract_target_label(action_dict["parameters"]),
                        reward=reward,
                        detected=detected,
                        raw_slm_valid=True,
                        was_overridden=False,
                        stealth_risk_score=risk,
                    )
                    obs = next_obs
                ep_rec = metrics.end_episode(win=win, detected=detected, timeout=timeout)
                if jsonl_path is not None:
                    metrics.append_episode_jsonl(ep_rec, jsonl_path)
                if num_episodes >= 25 and ((ep + 1) % 25 == 0 or (ep + 1) == num_episodes):
                    so_far = metrics.get_summary()
                    logger.info(
                        "[%s] Progress %d/%d | CumWin=%.1f%% CumDet=%.1f%% MeanSteps=%.1f",
                        experiment_name,
                        ep + 1,
                        num_episodes,
                        so_far.get("win_rate_percent", 0.0),
                        so_far.get("detection_rate_percent", 0.0),
                        so_far.get("mean_steps", 0.0),
                    )
        finally:
            try:
                baseline.terminate()
            except Exception:
                pass
            self.stop_server()

        json_name = f"{experiment_name}_metrics.json" if deterministic_filenames else None
        csv_name = experiment_name if deterministic_filenames else None
        metrics.export_json(filename=json_name)
        metrics.export_csv(filename=csv_name)
        return metrics.get_summary(), metrics


def _load_completed_checkpoint(
    output_dir: Path,
    experiment_name: str,
    expected_episodes: int,
) -> Optional[Dict[str, Any]]:
    """Return cached summary dict if checkpoint JSON exists with exact episode count."""
    ckpt_path = output_dir / f"{experiment_name}_metrics.json"
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        summary = payload.get("summary", {})
        if int(summary.get("total_episodes", -1)) == expected_episodes:
            logger.info("[SKIP] Loaded checkpoint %s (%d episodes)", ckpt_path, expected_episodes)
            return summary
    except Exception as exc:
        logger.warning("Ignoring unreadable checkpoint %s: %s", ckpt_path, exc)
    return None


def print_comparison_table(summaries: List[Dict[str, Any]]) -> None:
    """Pretty-print ESWA-style summary comparison table with 95% CIs and autonomy rates."""
    if not summaries:
        print("No experiment summaries recorded.")
        return

    header = (
        f"{'Experiment':<40} | {'Win % (±CI95)':<16} | {'Detect % (±CI95)':<17} | "
        f"{'Return (±CI95)':<17} | {'Steps (±CI95)':<15} | {'RawValid%':<9} | {'Override%':<9} | {'Risk':<6} | {'Def':<4}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        win_str = f"{s.get('win_rate_percent', 0):.1f}±{s.get('win_rate_ci95', 0):.1f}"
        det_str = f"{s.get('detection_rate_percent', 0):.1f}±{s.get('detection_rate_ci95', 0):.1f}"
        ret_str = f"{s.get('mean_return', 0):.1f}±{s.get('return_ci95', 0):.1f}"
        stp_str = f"{s.get('mean_steps', 0):.1f}±{s.get('steps_ci95', 0):.1f}"
        raw_v = f"{s.get('raw_slm_valid_rate_percent', 0):.1f}%"
        ovr = f"{s.get('wrapper_override_rate_percent', 0):.1f}%"
        rsk = f"{s.get('mean_stealth_risk_score', 0):.2f}"
        dfd = f"{s.get('total_stealth_deferred', 0)}"
        print(
            f"{str(s.get('experiment_name', 'exp')):<40} | {win_str:<16} | {det_str:<17} | "
            f"{ret_str:<17} | {stp_str:<15} | {raw_v:<9} | {ovr:<9} | {rsk:<6} | {dfd:<4}"
        )
    print("=" * len(header) + "\n")


def _model_presets(selected: Optional[List[str]] = None) -> List[Tuple[str, SLMConfig]]:
    """Build standard SLM model configs cleanly."""
    all_presets = {
        "1.5b": (
            "SLM_Qwen2.5_1.5B_BeliefGraph",
            SLMConfig(
                model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                quantization=None,
                temperature=0.0,
                max_new_tokens=160,
                use_memory=True,
                enable_action_masking=True,
                memory_window=5,
            ),
        ),
        "7b": (
            "SLM_Qwen2.5_7B_NF4_BeliefGraph",
            SLMConfig(
                model_name="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
                quantization="4bit",
                temperature=0.0,
                max_new_tokens=160,
                use_memory=True,
                enable_action_masking=True,
                memory_window=5,
            ),
        ),
    }
    keys = selected if selected else ["1.5b", "7b"]
    return [all_presets[k.lower()] for k in keys if k.lower() in all_presets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyber SLM Agent Benchmark Runner (ESWA 2026 Parity)")
    parser.add_argument(
        "--mode",
        type=str,
        default="adversarial_suite",
        choices=[
            "sanity_check",
            "adversarial_suite",
            "tri_ablation",
            "baselines_only",
            "regime_eval",
            "final_paper_150ep",
        ],
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scenarios/scenario_full_defender.yaml",
        help="Path to NetSecGame scenario YAML",
    )
    parser.add_argument(
        "--nodefender-config",
        type=str,
        default="scenarios/scenario_full_nodefender.yaml",
        help="Path to No-Defender scenario YAML (used in regime_eval mode)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output-dir", type=str, default="logs/benchmarks")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--regime2-only", action="store_true", help="In regime_eval mode, skip Regime 1 and run Regime 2 only")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model keys: 1.5b 7b")
    args = parser.parse_args()

    harness = BenchmarkHarness(
        task_config=args.config,
        port=args.port,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    summaries: List[Dict[str, Any]] = []

    if args.mode == "final_paper_150ep":
        suite_root = Path(args.output_dir) / "final_paper_150ep"
        suite_root.mkdir(parents=True, exist_ok=True)
        slm_episodes = args.episodes if args.episodes != 5 else 150
        random_episodes = 2000 if slm_episodes == 150 else max(args.episodes * 10, 50)

        regimes = [
            ("regime1a_full_nodefender", "scenarios/scenario_full_nodefender.yaml"),
            ("regime1b_full_defender", "scenarios/scenario_full_defender.yaml"),
            ("regime2a_threenets_nodefender", "scenarios/scenario_three_nets_nodefender.yaml"),
            ("regime2b_threenets_defender", "scenarios/scenario_three_nets_defender.yaml"),
        ]
        by_regime: Dict[str, List[Dict[str, Any]]] = {r[0]: [] for r in regimes}
        next_port = args.port

        # 1. Baselines (RandomAttacker 2000 eps + PureHeuristic 150 eps)
        if not args.skip_baselines:
            for regime_dir, yaml_path in regimes:
                out_dir = suite_root / regime_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                harness_r = BenchmarkHarness(
                    task_config=yaml_path, port=next_port, output_dir=str(out_dir), seed=args.seed
                )
                for exp_name, ep_cnt, runner_fn in [
                    ("Baseline_RandomAttacker", random_episodes, harness_r.run_baseline_random_agent),
                    ("Baseline_PureHeuristic", slm_episodes, harness_r.run_baseline_heuristic_agent),
                    (
                        "Ablation_PureHeuristic_WithStealth",
                        slm_episodes,
                        lambda **kw: harness_r.run_baseline_heuristic_agent(enable_stealth_pacing=True, **kw),
                    ),
                ]:
                    cached = _load_completed_checkpoint(out_dir, exp_name, ep_cnt)
                    if cached is not None:
                        summary = cached
                    else:
                        summary, _ = runner_fn(
                            experiment_name=exp_name,
                            num_episodes=ep_cnt,
                            deterministic_filenames=True,
                        )
                        next_port = harness_r.port + 2
                        harness_r.port = next_port
                    by_regime[regime_dir].append(summary)
                    summaries.append(summary)

        # 2. SLM Model Groups (each model loaded once and reused across all 4 regimes)
        _, cfg_7b_base = _model_presets(["7b"])[0]
        _, cfg_1_5b_base = _model_presets(["1.5b"])[0]
        slm_groups = [
            (
                cfg_7b_base,
                [
                    ("Ablation_SLM_7B_NF4_RawReAct", False, False, False),
                    ("Ablation_SLM_7B_NF4_BeliefGraph_NoStealth", True, True, False),
                    ("Proposed_SLM_7B_NF4_BeliefGraph_WithStealth", True, True, True),
                ],
            ),
            (
                cfg_1_5b_base,
                [
                    ("Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth", True, True, True),
                ],
            ),
        ]

        for base_cfg, variant_specs in slm_groups:
            pending: List[Tuple[str, str, str, SLMConfig]] = []
            for regime_dir, yaml_path in regimes:
                out_dir = suite_root / regime_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                for exp_name, use_mem, use_mask, use_stealth in variant_specs:
                    cached = _load_completed_checkpoint(out_dir, exp_name, slm_episodes)
                    if cached is not None:
                        by_regime[regime_dir].append(cached)
                        summaries.append(cached)
                    else:
                        slm_cfg = SLMConfig(
                            model_name=base_cfg.model_name,
                            quantization=base_cfg.quantization,
                            temperature=base_cfg.temperature,
                            max_new_tokens=base_cfg.max_new_tokens,
                            use_memory=use_mem,
                            enable_action_masking=use_mask,
                            enable_stealth_pacing=use_stealth,
                            memory_window=5,
                        )
                        pending.append((regime_dir, yaml_path, exp_name, slm_cfg))

            if pending:
                backend = HuggingFaceSLMBackend(base_cfg)
                try:
                    for regime_dir, yaml_path, exp_name, slm_cfg in pending:
                        out_dir = suite_root / regime_dir
                        harness_r = BenchmarkHarness(
                            task_config=yaml_path, port=next_port, output_dir=str(out_dir), seed=args.seed
                        )
                        summary, _ = harness_r.run_slm_agent(
                            slm_cfg,
                            experiment_name=exp_name,
                            num_episodes=slm_episodes,
                            backend=backend,
                            deterministic_filenames=True,
                        )
                        next_port = harness_r.port + 2
                        by_regime[regime_dir].append(summary)
                        summaries.append(summary)
                finally:
                    del backend
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        for regime_dir, _ in regimes:
            print(f"\n>>> FINAL 150-EPISODE SUMMARY: {regime_dir} <<<")
            print_comparison_table(by_regime[regime_dir])

        consolidated_path = suite_root / "consolidated_150ep_summary.json"
        with open(consolidated_path, "w", encoding="utf-8") as fh:
            json.dump({"by_regime": by_regime, "all_summaries": summaries}, fh, indent=2)
        logger.info("Saved consolidated summary to %s", consolidated_path)
        return

    if args.mode == "regime_eval":
        # Regime 1 (Without Defender - Table 4 parity) & Regime 2 (With Defender - Table 5 parity)
        _, base_cfg = _model_presets(args.models or ["7b"])[0]
        backend = HuggingFaceSLMBackend(base_cfg)
        try:
            next_port = args.port
            if not args.regime2_only:
                # --- Regime 1: Without Defender ---
                harness_nd = BenchmarkHarness(
                    task_config=args.nodefender_config,
                    port=args.port,
                    output_dir=str(Path(args.output_dir) / "regime1_nodefender"),
                    seed=args.seed,
                )
                r1_summaries: List[Dict[str, Any]] = []
                r1_heur, _ = harness_nd.run_baseline_heuristic_agent(
                    experiment_name="R1_NoDef_PureHeuristicBaseline",
                    num_episodes=args.episodes,
                )
                r1_summaries.append(r1_heur)

                cfg_r1_slm = SLMConfig(
                    model_name=base_cfg.model_name,
                    quantization=base_cfg.quantization,
                    temperature=base_cfg.temperature,
                    use_memory=True,
                    enable_action_masking=True,
                    enable_stealth_pacing=False,
                    memory_window=5,
                )
                r1_slm, _ = harness_nd.run_slm_agent(
                    cfg_r1_slm,
                    "R1_NoDef_SLM_7B_NF4_BeliefGraph",
                    num_episodes=args.episodes,
                    backend=backend,
                )
                r1_summaries.append(r1_slm)
                print("\n>>> REGIME 1: WITHOUT DEFENDER (ESWA Table 4 Parity) <<<")
                print_comparison_table(r1_summaries)
                summaries.extend(r1_summaries)
                next_port = harness_nd.port + 2

            # --- Regime 2: With Defender ---
            harness_def = BenchmarkHarness(
                task_config=args.config,
                port=next_port,
                output_dir=str(Path(args.output_dir) / "regime2_defender"),
                seed=args.seed,
            )
            r2_summaries: List[Dict[str, Any]] = []
            r2_heur, _ = harness_def.run_baseline_heuristic_agent(
                experiment_name="R2_Def_PureHeuristicBaseline",
                num_episodes=args.episodes,
            )
            r2_summaries.append(r2_heur)

            cfg_r2_nostealth = SLMConfig(
                model_name=base_cfg.model_name,
                quantization=base_cfg.quantization,
                temperature=base_cfg.temperature,
                use_memory=True,
                enable_action_masking=True,
                enable_stealth_pacing=False,
                memory_window=5,
            )
            r2_slm_no, _ = harness_def.run_slm_agent(
                cfg_r2_nostealth,
                "R2_Def_SLM_7B_BeliefGraph_NoStealth",
                num_episodes=args.episodes,
                backend=backend,
            )
            r2_summaries.append(r2_slm_no)

            cfg_r2_stealth = SLMConfig(
                model_name=base_cfg.model_name,
                quantization=base_cfg.quantization,
                temperature=base_cfg.temperature,
                use_memory=True,
                enable_action_masking=True,
                enable_stealth_pacing=True,
                memory_window=5,
            )
            r2_slm_yes, _ = harness_def.run_slm_agent(
                cfg_r2_stealth,
                "R2_Def_SLM_7B_BeliefGraph_WithStealth",
                num_episodes=args.episodes,
                backend=backend,
            )
            r2_summaries.append(r2_slm_yes)
            print("\n>>> REGIME 2: WITH DEFENDER (ESWA Table 5 Parity) <<<")
            print_comparison_table(r2_summaries)
            summaries.extend(r2_summaries)
        finally:
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    if args.mode == "tri_ablation":
        # Reuse loaded 7B/1.5B or chosen model across Config A, Config B (Raw ReAct), Config C.
        model_label, base_cfg = _model_presets(args.models or ["1.5b"])[0]
        configs = [
            (
                f"{model_label}_ConfigA_BeliefGraph_Masking",
                SLMConfig(
                    model_name=base_cfg.model_name,
                    quantization=base_cfg.quantization,
                    use_memory=True,
                    enable_action_masking=True,
                    memory_window=5,
                ),
            ),
            (
                f"{model_label}_ConfigB_RawReAct",
                SLMConfig(
                    model_name=base_cfg.model_name,
                    quantization=base_cfg.quantization,
                    use_memory=False,
                    enable_action_masking=False,
                    memory_window=5,
                ),
            ),
            (
                f"{model_label}_ConfigC_BeliefGraph_NoMasking",
                SLMConfig(
                    model_name=base_cfg.model_name,
                    quantization=base_cfg.quantization,
                    use_memory=True,
                    enable_action_masking=False,
                    memory_window=5,
                ),
            ),
        ]
        backend = HuggingFaceSLMBackend(base_cfg)
        try:
            for exp_name, cfg in configs:
                summary, _ = harness.run_slm_agent(cfg, exp_name, num_episodes=args.episodes, backend=backend)
                summaries.append(summary)
        finally:
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    else:
        if not args.skip_baselines:
            rand_sum, _ = harness.run_baseline_random_agent(num_episodes=args.episodes)
            summaries.append(rand_sum)
            heur_sum, _ = harness.run_baseline_heuristic_agent(num_episodes=args.episodes)
            summaries.append(heur_sum)

        if args.mode != "baselines_only":
            for exp_name, cfg in _model_presets(args.models):
                summary, _ = harness.run_slm_agent(cfg, exp_name, num_episodes=args.episodes)
                summaries.append(summary)

    print_comparison_table(summaries)


if __name__ == "__main__":
    main()
