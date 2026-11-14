"""
Unit tests for research_extension:
1. Topological Belief Graph classification & dynamic valid-action derivation
2. Temporal Stealth Pacing (W=5 sliding-window density & consecutive streak)
3. Zero-hardcoding static AST audit across all research_extension/*.py modules
4. MetricsLogger autonomy telemetry & 95% CI calculations
5. ModernSLMAttacker autonomy vs Config B (no override) vs masked fallback
6. PureHeuristicBaseline domain-agnostic selection
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest
from netsecgame import Data, GameState, IP, Network, Observation, Service

from research_extension.base_agent import MockSLMBackend, ModernSLMAttacker, SLMConfig
from research_extension.benchmark_runner import PureHeuristicBaseline
from research_extension.memory_module import (
    EpistemicState,
    EpisodicMemoryBuffer,
    SemanticMemoryStore,
    extract_goal_target_ip,
)
from research_extension.metrics_logger import MetricsLogger


def _make_sample_state() -> GameState:
    return GameState(
        controlled_hosts={IP("172.16.44.10")},
        known_hosts={IP("172.16.44.10"), IP("172.16.44.20"), IP("172.16.44.30")},
        known_networks={Network("172.16.44.0", 24)},
        known_services={
            IP("172.16.44.10"): {Service("local_marker", "passive", "1.0", True)},
            IP("172.16.44.20"): {Service("redis_custom", "remote", "7.0", False)},
        },
        known_data={IP("172.16.44.10"): {Data("Alice", "SecretKey")}},
        known_blocks={},
    )


def test_zero_hardcoding_ast_scan() -> None:
    """Ensure research_extension/*.py contains zero hardcoded topology/service/data strings."""
    pkg_dir = Path(__file__).resolve().parent.parent / "research_extension"
    forbidden_terms = {
        "microsoft-ds",
        "active-directory",
        "can_attack_start_here",
        "datafromserver",
        "three_net_scenario",
        "scenario1",
        "bash_vuln",
    }
    cidr_re = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")
    ipv4_re = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

    for py_file in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                low = val.lower()
                for term in forbidden_terms:
                    assert term not in low, f"Hardcoded term '{term}' found in {py_file.name}: {val!r}"
                assert not cidr_re.search(val), f"Hardcoded CIDR found in {py_file.name}: {val!r}"
                for ip_hit in ipv4_re.findall(val):
                    assert ip_hit == "127.0.0.1", f"Hardcoded non-localhost IP '{ip_hit}' found in {py_file.name}"


def test_topological_belief_graph_and_pruning() -> None:
    state = _make_sample_state()
    goal = "Exfiltrate data to '172.16.99.5'"
    obs = Observation(state=state, reward=0.0, end=False, info={"goal_description": goal})

    sem = SemanticMemoryStore(stealth_window=5)
    belief = sem.classify_topology(obs)

    assert belief["goal_target_ip"] == "172.16.99.5"
    assert belief["subnets"]["172.16.44.0/24"]["state"] == EpistemicState.UNEXPLORED.value
    assert belief["hosts"]["172.16.44.10"]["state"] == EpistemicState.COMPROMISED.value
    assert belief["hosts"]["172.16.44.20"]["state"] == EpistemicState.MAPPED.value
    assert belief["hosts"]["172.16.44.30"]["state"] == EpistemicState.UNEXPLORED.value

    valid = sem.get_valid_action_space(state)
    # Passive service on controlled host should not yield ExploitService; remote service on .20 should.
    assert any(
        a["action"] == "ExploitService"
        and a["parameters"]["target_host"] == "172.16.44.20"
        and a["parameters"]["target_service"] == "redis_custom"
        for a in valid
    )
    assert not any(
        a["action"] == "ExploitService" and a["parameters"]["target_host"] == "172.16.44.10"
        for a in valid
    )

    # Prune ONLY exact executed action; other valid actions remain untouched.
    scan_action = {
        "source_host": "172.16.44.10",
        "target_network": "172.16.44.0/24",
    }
    sem.record_action_outcome("ScanNetwork", scan_action, outcome_status="success", state_changed=True)
    assert sem.is_action_redundant("ScanNetwork", scan_action)
    valid_after = sem.get_valid_action_space(state)
    assert not any(a["action"] == "ScanNetwork" for a in valid_after)
    assert any(a["action"] == "ExploitService" for a in valid_after)

    # Probe .30 and fail -> should transition .30 to EXHAUSTED when no services found.
    sem.record_action_outcome(
        "FindServices",
        {"source_host": "172.16.44.10", "target_host": "172.16.44.30"},
        outcome_status="no_effect",
        state_changed=False,
    )
    belief_after = sem.classify_topology(obs)
    assert belief_after["subnets"]["172.16.44.0/24"]["state"] == EpistemicState.MAPPED.value
    assert belief_after["hosts"]["172.16.44.30"]["state"] == EpistemicState.EXHAUSTED.value


def test_temporal_stealth_pacing_and_prompt() -> None:
    episodic = EpisodicMemoryBuffer(max_capacity=10, stealth_window=5)
    sem = SemanticMemoryStore(stealth_window=5)

    for step in range(1, 5):
        episodic.add_step(
            step_num=step,
            action_type="ScanNetwork",
            parameters={"target_network": f"10.10.{step}.0/24", "source_host": "10.10.0.2"},
            outcome_status="success",
            reward=-1.0,
            state_delta_summary="hosts:+1",
        )
        sem.record_action_outcome(
            "ScanNetwork",
            {"target_network": f"10.10.{step}.0/24", "source_host": "10.10.0.2"},
            outcome_status="success",
        )

    assert episodic.get_action_density("ScanNetwork") == pytest.approx(0.8)
    assert episodic.get_consecutive_streak("ScanNetwork") == 4

    scan_risk = sem.score_action_stealth_risk("ScanNetwork", episodic_memory=episodic)
    find_data_risk = sem.score_action_stealth_risk("FindData", episodic_memory=episodic)
    assert scan_risk > find_data_risk
    assert scan_risk >= 0.65

    state = _make_sample_state()
    obs = Observation(state=state, reward=0.0, end=False, info={"goal_description": "Exfiltrate data to '172.16.44.99'"})
    prompt = sem.format_context_prompt(obs, episodic_memory=episodic)
    assert "TOPOLOGICAL BELIEF GRAPH" in prompt
    assert "TEMPORAL STEALTH ADVISORY" in prompt
    assert "PERMISSIBLE UNEXPLORED ACTIONS" in prompt
    assert "172.16.44.99" in prompt
    # Ensure no forced single-action JSON answer appears in prompt.
    assert "recommended action:" not in prompt.lower()
    assert extract_goal_target_ip("No address here") is None


def test_metrics_logger_ci95_and_autonomy(tmp_path: Path) -> None:
    ml = MetricsLogger(output_dir=str(tmp_path), experiment_name="unit_ci_test")

    # Episode 0: win, 10 steps, return 90, 8/10 raw valid, 2/10 overridden
    ml.start_episode(0, "TestModel", "test_scen")
    for s in range(1, 11):
        ml.record_step(
            step_num=s,
            action_type="FindServices",
            reward=10.0 if s == 10 else -1.0,
            raw_slm_valid=(s <= 8),
            was_overridden=(s > 8),
            stealth_risk_score=0.25,
        )
    ml.end_episode(win=True, detected=False, timeout=False)

    # Episode 1: detected, 4 steps, return -50, 2/4 raw valid, 2/4 overridden
    ml.start_episode(1, "TestModel", "test_scen")
    for s in range(1, 5):
        ml.record_step(
            step_num=s,
            action_type="ScanNetwork",
            reward=-47.0 if s == 4 else -1.0,
            detected=(s == 4),
            raw_slm_valid=(s <= 2),
            was_overridden=(s > 2),
            stealth_risk_score=0.75,
        )
    ml.end_episode(win=False, detected=True, timeout=False)

    summary = ml.get_summary()
    assert summary["win_rate_percent"] == 50.0
    assert summary["detection_rate_percent"] == 50.0
    assert summary["mean_steps"] == 7.0
    # Sample SD of [10, 4] = sqrt(18) = 4.24264; CI95 = 1.96 * SD / sqrt(2) = 1.96 * 3 = 5.88
    expected_steps_ci = round(1.96 * (math.sqrt(18.0) / math.sqrt(2.0)), 2)
    assert summary["steps_ci95"] == expected_steps_ci
    # Total steps = 14; raw_valid = 10 -> 71.43%; overrides = 4 -> 28.57%
    assert summary["raw_slm_valid_rate_percent"] == pytest.approx(71.43, abs=0.05)
    assert summary["wrapper_override_rate_percent"] == pytest.approx(28.57, abs=0.05)


def test_attacker_autonomy_and_config_b_no_override() -> None:
    state = _make_sample_state()
    obs = Observation(state=state, reward=0.0, end=False, info={"goal_description": "Exfiltrate data to '172.16.44.99'"})

    # 1. Valid SLM response must NOT be overridden.
    valid_json = '{"action": "ExploitService", "parameters": {"target_host": "172.16.44.20", "target_service": "redis_custom", "source_host": "172.16.44.10"}}'
    cfg_a = SLMConfig(backend="mock", use_memory=True, enable_action_masking=True)
    agent_a = ModernSLMAttacker(cfg_a, MockSLMBackend([valid_json]))
    act, act_dict, _, _ = agent_a.plan_next_action(obs)
    assert act is not None
    assert act_dict["action"] == "ExploitService"
    assert agent_a.last_decision_metadata["raw_slm_valid"] is True
    assert agent_a.last_decision_metadata["was_overridden"] is False

    # 2. Config B (enable_action_masking=False) with invalid text MUST NOT override.
    cfg_b = SLMConfig(backend="mock", use_memory=False, enable_action_masking=False)
    agent_b = ModernSLMAttacker(cfg_b, MockSLMBackend(["I think I will think about it without json"]))
    act_b, _, _, _ = agent_b.plan_next_action(obs)
    assert act_b is None
    assert agent_b.last_decision_metadata["raw_slm_valid"] is False
    assert agent_b.last_decision_metadata["was_overridden"] is False

    # 3. Masked agent with invalid text or missing required parameter DOES trigger domain-agnostic fallback.
    agent_c = ModernSLMAttacker(cfg_a, MockSLMBackend(["broken output", '{"action": "FindServices", "parameters": {"source_host": "172.16.44.10"}}']))
    act_c, _, _, _ = agent_c.plan_next_action(obs)
    assert act_c is not None
    assert agent_c.last_decision_metadata["raw_slm_valid"] is False
    assert agent_c.last_decision_metadata["was_overridden"] is True

    act_d, _, _, _ = agent_c.plan_next_action(obs)
    assert act_d is not None
    assert agent_c.last_decision_metadata["raw_slm_valid"] is False
    assert agent_c.last_decision_metadata["was_overridden"] is True


def test_pure_heuristic_baseline_priority() -> None:
    state = _make_sample_state()
    obs = Observation(state=state, reward=0.0, end=False, info={"goal_description": "Exfiltrate data to '172.16.44.99'"})
    baseline = PureHeuristicBaseline()
    act, act_dict, _ = baseline.select_action(obs)
    assert act is not None
    # With no second controlled host to exfiltrate yet, FindData takes priority over ExploitService.
    assert act_dict["action"] == "FindData"
    assert act_dict["parameters"]["target_host"] == "172.16.44.10"


def test_active_stealth_pacing_soft_masking_and_deadlock_safety() -> None:
    state = GameState(
        controlled_hosts={IP("172.16.44.10")},
        known_hosts={IP("172.16.44.10"), IP("172.16.44.20")},
        known_networks={Network("172.16.44.0", 24), Network("172.16.45.0", 24)},
        known_services={IP("172.16.44.20"): {Service("redis_custom", "remote", "7.0", False)}},
        known_data={},
        known_blocks={},
    )
    sem_paced = SemanticMemoryStore(stealth_window=5, enable_stealth_pacing=True)
    sem_unpaced = SemanticMemoryStore(stealth_window=5, enable_stealth_pacing=False)

    # Record 1 ScanNetwork -> another ScanNetwork would be consecutive=2 and density=0.40 >= 0.25!
    first_scan = {"source_host": "172.16.44.10", "target_network": "172.16.44.0/24"}
    sem_paced.record_action_outcome("ScanNetwork", first_scan, outcome_status="success")
    sem_unpaced.record_action_outcome("ScanNetwork", first_scan, outcome_status="success")

    paced_actions = sem_paced.get_valid_action_space(state)
    unpaced_actions = sem_unpaced.get_valid_action_space(state)

    assert not any(a["action"] == "ScanNetwork" for a in paced_actions)
    assert sem_paced.last_stealth_deferred_count == 3
    assert any(a["action"] == "ScanNetwork" for a in unpaced_actions)
    assert sem_unpaced.last_stealth_deferred_count == 0

    # Deadlock prevention: if ONLY ScanNetwork remains, active stealth pacing must NOT deadlock.
    scan_only_state = GameState(
        controlled_hosts={IP("172.16.44.10")},
        known_hosts={IP("172.16.44.10")},
        known_networks={Network("172.16.44.0", 24), Network("172.16.45.0", 24)},
        known_services={},
        known_data={},
        known_blocks={},
    )
    # Mark FindServices and FindData as already executed so only ScanNetwork remains.
    sem_paced.record_action_outcome(
        "FindServices", {"source_host": "172.16.44.10", "target_host": "172.16.44.10"}, "success"
    )
    sem_paced.record_action_outcome(
        "FindData", {"source_host": "172.16.44.10", "target_host": "172.16.44.10"}, "success"
    )
    # Also ensure ScanNetwork is still high risk by adding one more ScanNetwork in window.
    fallback_actions = sem_paced.get_valid_action_space(scan_only_state)
    assert len(fallback_actions) == 1
    assert fallback_actions[0]["action"] == "ScanNetwork"
    assert sem_paced.last_stealth_deadlock_bypass is True


def test_depth_first_interleaved_streak_breaking() -> None:
    """Verify consecutive FindServices bursts are broken by interleaving ExploitService."""
    state = GameState(
        controlled_hosts={IP("172.16.44.10")},
        known_hosts={IP("172.16.44.10"), IP("172.16.44.20"), IP("172.16.44.30")},
        known_networks={Network("172.16.44.0", 24)},
        known_services={IP("172.16.44.20"): {Service("redis_custom", "remote", "7.0", False)}},
        known_data={},
        known_blocks={},
    )
    sem = SemanticMemoryStore(stealth_window=5, enable_stealth_pacing=True)
    # Last action was FindServices on .20 -> another FindServices on .30 must be masked
    # because ExploitService on .20 is available!
    sem.record_action_outcome(
        "FindServices", {"source_host": "172.16.44.10", "target_host": "172.16.44.20"}, "success"
    )
    actions = sem.get_valid_action_space(state)
    action_types = [a["action"] for a in actions]
    assert "ExploitService" in action_types
    assert "FindServices" not in action_types
    assert "ScanNetwork" not in action_types
    # Depth-First Kill-Chain Locality: FindData (tier 1) precedes ExploitService (tier 2)
    assert actions[0]["action"] == "FindData"
    assert actions[1]["action"] == "ExploitService"
