"""
memory_module.py — Domain-Agnostic Topological Belief Graph & Temporal Stealth Memory.

Provides:
1. EpisodicMemoryBuffer: Sliding-window action trace and temporal detectability metrics
   (window density d_t(a), consecutive streak c_t(a), and stealth risk estimation)
   for evading sliding-window SIEM detectors.
2. SemanticMemoryStore: Purely observation-driven Topological Belief Graph classifying
   subnets, hosts, services, and data into generic epistemic states:
   UNEXPLORED, MAPPED, COMPROMISED, and EXHAUSTED.
   Computes the exact permissible unexplored action space without any domain,
   topology, IP, service, or data hardcoding.
"""

from __future__ import annotations

import ipaddress
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import groupby
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

# Matches IPv4 addresses parsed dynamically from environment goal descriptions.
_IPV4_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Canonicalize alias action names accepted by NetSecGame llm_utils.
_ACTION_ALIAS_MAP = {
    "ScanServices": "FindServices",
}


class EpistemicState(str, Enum):
    """Generic epistemic status for topological entities."""

    UNEXPLORED = "UNEXPLORED"
    MAPPED = "MAPPED"
    COMPROMISED = "COMPROMISED"
    EXHAUSTED = "EXHAUSTED"


def canonicalize_action_type(action_type: Any) -> str:
    """Normalize an ActionType enum or string to its canonical NetSecGame name."""
    raw = getattr(action_type, "value", None)
    if raw is None:
        raw = str(action_type)
    raw = raw.replace("ActionType.", "").strip()
    return _ACTION_ALIAS_MAP.get(raw, raw)


def normalize_ip(obj: Any) -> str:
    """Normalize an IP object or string to its canonical string."""
    if obj is None:
        return ""
    ip_attr = getattr(obj, "ip", None)
    if ip_attr is not None:
        return str(ip_attr).strip()
    return str(obj).strip()


def normalize_network(obj: Any) -> str:
    """Normalize a Network object or CIDR string to 'ip/mask'."""
    if obj is None:
        return ""
    ip_attr = getattr(obj, "ip", None)
    mask_attr = getattr(obj, "mask", None)
    if ip_attr is not None and mask_attr is not None:
        return f"{str(ip_attr).strip()}/{int(mask_attr)}"
    return str(obj).strip()


def normalize_service(obj: Any) -> str:
    """Normalize a Service object or string to lowercase service name."""
    if obj is None:
        return ""
    name_attr = getattr(obj, "name", None)
    if name_attr is not None:
        return str(name_attr).strip().lower()
    return str(obj).strip().lower()


def normalize_data(obj: Any) -> Tuple[str, str]:
    """Normalize a Data object or dict to a canonical (owner, id) tuple."""
    if obj is None:
        return ("", "")
    if isinstance(obj, dict):
        return (str(obj.get("owner", "")).strip(), str(obj.get("id", "")).strip())
    owner = getattr(obj, "owner", None)
    data_id = getattr(obj, "id", None)
    if owner is not None or data_id is not None:
        return (str(owner or "").strip(), str(data_id or "").strip())
    return ("", str(obj).strip())


def canonicalize_parameters(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a clean JSON-serializable dictionary with canonicalized parameter values."""
    if not parameters:
        return {}
    clean: Dict[str, Any] = {}
    for key, val in parameters.items():
        if key in ("source_host", "target_host", "blocked_host", "blocker_host"):
            clean[key] = normalize_ip(val)
        elif key == "target_network":
            clean[key] = normalize_network(val)
        elif key == "target_service":
            clean[key] = normalize_service(val)
        elif key == "data":
            owner, data_id = normalize_data(val)
            clean[key] = {"owner": owner, "id": data_id}
        else:
            clean[key] = str(val).strip() if not isinstance(val, (int, float, bool)) else val
    return clean


def canonical_action_key(action_type: Any, parameters: Optional[Dict[str, Any]]) -> Tuple[Any, ...]:
    """Build a deterministic hashable tuple key for deduplicating exact actions."""
    act = canonicalize_action_type(action_type)
    params = canonicalize_parameters(parameters)
    items: List[Tuple[str, Any]] = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, dict):
            items.append((k, tuple(sorted((str(dk), str(dv)) for dk, dv in v.items()))))
        else:
            items.append((k, str(v)))
    return (act, tuple(items))


def extract_goal_target_ip(goal_description: str) -> Optional[str]:
    """
    Dynamically parse the exfiltration destination host IP from observation goal text.
    Never uses hardcoded defaults; returns None if no IPv4 host appears in the goal.
    """
    if not goal_description:
        return None
    targeted = re.search(
        r"(?:to|host|target|destination)\s*['\"]?((?:\d{1,3}\.){3}\d{1,3})['\"]?",
        goal_description,
        flags=re.IGNORECASE,
    )
    if targeted:
        return targeted.group(1)
    matches = _IPV4_REGEX.findall(goal_description)
    return matches[-1] if matches else None


def _ip_in_network(ip_str: str, net_str: str) -> bool:
    """Check membership of an IPv4 string inside a CIDR subnet string."""
    try:
        return ipaddress.IPv4Address(ip_str) in ipaddress.IPv4Network(net_str, strict=False)
    except ValueError:
        return False


def _extract_state(obs_or_state: Any) -> Any:
    """Support passing either an Observation (with .state) or a raw GameState."""
    if obs_or_state is None:
        return None
    return getattr(obs_or_state, "state", obs_or_state)


def _normalize_blocks(raw_blocks: Any) -> Dict[str, Set[str]]:
    """Normalize state.known_blocks mapping into {ip_str: set_of_ip_strs}."""
    blocks: Dict[str, Set[str]] = {}
    if not raw_blocks or not isinstance(raw_blocks, dict):
        return blocks
    for host, peers in raw_blocks.items():
        h_str = normalize_ip(host)
        if not h_str:
            continue
        peer_set = blocks.setdefault(h_str, set())
        if isinstance(peers, (set, list, tuple)):
            for p in peers:
                p_str = normalize_ip(p)
                if p_str:
                    peer_set.add(p_str)
        elif peers is not None:
            p_str = normalize_ip(peers)
            if p_str:
                peer_set.add(p_str)
    return blocks


def _is_blocked(src: str, dst: str, blocks: Dict[str, Set[str]]) -> bool:
    """Check symmetrically whether src and dst have an observed firewall block."""
    if not src or not dst:
        return False
    return dst in blocks.get(src, set()) or src in blocks.get(dst, set())


@dataclass
class StepMemory:
    """Compact record of a single executed step in episodic memory."""

    step_num: int
    action_type: str
    parameters: Dict[str, Any]
    outcome_status: str
    reward: float
    state_delta_summary: str
    detected: bool = False
    stealth_risk: float = 0.0


class EpisodicMemoryBuffer:
    """
    Bounded episodic buffer tracking recent interaction traces and temporal
    SIEM-detectability dynamics over a sliding window of size W.
    """

    def __init__(self, max_capacity: int = 10, stealth_window: int = 5) -> None:
        self.max_capacity = max_capacity
        self.stealth_window = max(1, int(stealth_window))
        self.buffer: Deque[StepMemory] = deque(maxlen=max_capacity)
        # Keep the full episode action sequence so density/streak/episode totals stay exact.
        self.episode_action_history: List[str] = []
        self.episode_action_counts: Dict[str, int] = {}

    def add_step(
        self,
        step_num: int,
        action_type: str,
        parameters: Dict[str, Any],
        outcome_status: str,
        reward: float,
        state_delta_summary: str,
        detected: bool = False,
        stealth_risk: float = 0.0,
    ) -> None:
        """Append a step outcome to the episodic buffer and temporal trace."""
        canon_action = canonicalize_action_type(action_type)
        canon_params = canonicalize_parameters(parameters)
        entry = StepMemory(
            step_num=step_num,
            action_type=canon_action,
            parameters=canon_params,
            outcome_status=outcome_status,
            reward=reward,
            state_delta_summary=state_delta_summary,
            detected=detected,
            stealth_risk=stealth_risk,
        )
        self.buffer.append(entry)
        self.episode_action_history.append(canon_action)
        self.episode_action_counts[canon_action] = self.episode_action_counts.get(canon_action, 0) + 1

    def get_recent_steps(self, n: int = 5) -> List[StepMemory]:
        """Return the n most recent episodic transitions."""
        if n <= 0:
            return []
        return list(self.buffer)[-n:]

    def get_action_repetition_count(self, action_type: str, target: Optional[str] = None) -> int:
        """Count occurrences in the recent episodic buffer matching action_type (and optional target)."""
        canon = canonicalize_action_type(action_type)
        count = 0
        for step in self.buffer:
            if step.action_type != canon:
                continue
            if target is None:
                count += 1
            else:
                tgt = str(target).strip()
                p = step.parameters
                if (
                    p.get("target_host") == tgt
                    or p.get("target_network") == tgt
                    or p.get("source_host") == tgt
                ):
                    count += 1
        return count

    def get_window_actions(self, window_size: Optional[int] = None) -> List[str]:
        """Return the last W executed action types in the episode."""
        w = self.stealth_window if window_size is None else max(1, int(window_size))
        return list(self.episode_action_history[-w:])

    def get_action_density(
        self,
        action_type: str,
        window_size: Optional[int] = None,
        prospective: bool = False,
    ) -> float:
        """
        Compute sliding-window density d_t(a) = count_W(a) / W.
        If prospective=True, computes density if `action_type` were taken next.
        """
        w = self.stealth_window if window_size is None else max(1, int(window_size))
        canon = canonicalize_action_type(action_type)
        if prospective:
            tail = self.episode_action_history[-(w - 1) :] if w > 1 else []
            window = tail + [canon]
        else:
            window = self.episode_action_history[-w:]
        return window.count(canon) / float(w)

    def get_consecutive_streak(self, action_type: str, prospective: bool = False) -> int:
        """
        Compute consecutive streak length c_t(a).
        If prospective=True, returns the streak length if `action_type` is taken next.
        """
        canon = canonicalize_action_type(action_type)
        streak = 0
        for act in reversed(self.episode_action_history):
            if act == canon:
                streak += 1
            else:
                break
        return streak + 1 if prospective else streak

    def clear(self) -> None:
        """Reset episodic history at episode start."""
        self.buffer.clear()
        self.episode_action_history.clear()
        self.episode_action_counts.clear()


class SemanticMemoryStore:
    """
    Observation-driven Topological Belief Graph & Temporal Stealth Pacing engine.

    Maintains canonicalized entity beliefs across an episode without any
    scenario-, IP-, service-, or data-specific hardcoding.
    """

    # Generic structural thresholds matching sliding-window SIEM sensitivity profiles.
    STEALTH_DENSITY_ALERT = 0.40  # >= 2 actions of the same type in W=5
    CONSECUTIVE_STREAK_ALERT = 2  # >= 2 back-to-back actions of the same type

    # Sliding-window SIEM trigger boundaries (calibrated to GlobalDefender W=5).
    SIEM_RATIO_THRESHOLDS: Dict[str, float] = {
        "ScanNetwork": 0.25,
        "FindServices": 0.30,
        "ExploitService": 0.25,
        "FindData": 0.50,
        "ExfiltrateData": 0.25,
    }
    SIEM_CONSECUTIVE_THRESHOLDS: Dict[str, int] = {
        "ScanNetwork": 2,
        "FindServices": 3,
        "ExfiltrateData": 2,
    }

    def __init__(self, stealth_window: int = 5, enable_stealth_pacing: bool = True) -> None:
        self.stealth_window = max(1, int(stealth_window))
        self.enable_stealth_pacing = bool(enable_stealth_pacing)
        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset all per-episode topological, execution, and temporal stealth state."""
        self.goal_description: str = ""
        self.goal_target_ip: Optional[str] = None

        # Canonical sets of observed entities.
        self.known_networks: Set[str] = set()
        self.known_hosts: Set[str] = set()
        self.controlled_hosts: Set[str] = set()
        self.known_services: Dict[str, Set[str]] = {}  # host_ip -> set of non-passive service names
        self.known_data: Dict[str, Set[Tuple[str, str]]] = {}  # host_ip -> set of (owner, id)
        self.known_blocks: Dict[str, Set[str]] = {}

        # Execution telemetry keyed by canonical tuple.
        self.executed_actions: Set[Tuple[Any, ...]] = set()
        self.failed_actions: Set[Tuple[Any, ...]] = set()
        self.succeeded_actions: Set[Tuple[Any, ...]] = set()
        self.exhausted_entities: Set[str] = set()

        # Target-level attempt tracking for epistemic node classification.
        self.scanned_networks: Set[str] = set()
        self.probed_service_hosts: Set[str] = set()
        self.searched_data_hosts: Set[str] = set()
        self.failed_services_by_host: Dict[str, Set[str]] = {}

        # Sliding-window action sequence mirrored here so SemanticMemoryStore can score stealth standalone.
        self.action_history: List[str] = []
        self.episode_action_counts: Dict[str, int] = {}
        self.last_stealth_deferred_count: int = 0
        self.last_stealth_deadlock_bypass: bool = False

    def update_from_observation(self, obs_or_state: Any, goal_description: str = "") -> None:
        """Ingest current NetSecGame Observation or GameState into canonical belief sets."""
        if obs_or_state is None:
            return
        if not goal_description:
            info = getattr(obs_or_state, "info", None)
            if isinstance(info, dict):
                goal_description = str(info.get("goal_description", "") or "")
        if goal_description:
            self.goal_description = goal_description
            parsed_ip = extract_goal_target_ip(goal_description)
            if parsed_ip:
                self.goal_target_ip = parsed_ip

        state = _extract_state(obs_or_state)
        if state is None:
            return

        for net in getattr(state, "known_networks", ()) or ():
            net_str = normalize_network(net)
            if net_str:
                self.known_networks.add(net_str)

        for host in getattr(state, "known_hosts", ()) or ():
            h_str = normalize_ip(host)
            if h_str:
                self.known_hosts.add(h_str)

        for host in getattr(state, "controlled_hosts", ()) or ():
            h_str = normalize_ip(host)
            if h_str:
                self.controlled_hosts.add(h_str)
                self.known_hosts.add(h_str)

        raw_services = getattr(state, "known_services", {}) or {}
        if isinstance(raw_services, dict):
            for host, srv_list in raw_services.items():
                h_str = normalize_ip(host)
                if not h_str:
                    continue
                bucket = self.known_services.setdefault(h_str, set())
                for srv in srv_list or ():
                    # Skip local-only capability markers (is_local=True) generically.
                    if bool(getattr(srv, "is_local", False)):
                        continue
                    s_name = normalize_service(srv)
                    if s_name:
                        bucket.add(s_name)

        raw_data = getattr(state, "known_data", {}) or {}
        if isinstance(raw_data, dict):
            for host, data_items in raw_data.items():
                h_str = normalize_ip(host)
                if not h_str:
                    continue
                bucket = self.known_data.setdefault(h_str, set())
                for item in data_items or ():
                    owner, data_id = normalize_data(item)
                    if owner or data_id:
                        bucket.add((owner, data_id))

        self.known_blocks = _normalize_blocks(getattr(state, "known_blocks", {}))

    def record_action_outcome(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        outcome_status: str,
        reward: float = 0.0,
        state_changed: bool = True,
    ) -> None:
        """Record an executed action's canonical key, success/failure status, and temporal footprint."""
        canon_action = canonicalize_action_type(action_type)
        canon_params = canonicalize_parameters(parameters)
        key = canonical_action_key(canon_action, canon_params)

        self.executed_actions.add(key)
        self.action_history.append(canon_action)
        self.episode_action_counts[canon_action] = self.episode_action_counts.get(canon_action, 0) + 1

        is_success = str(outcome_status).strip().lower() in ("success", "true", "ok") or reward > 0
        if is_success:
            self.succeeded_actions.add(key)
        else:
            self.failed_actions.add(key)

        target_net = canon_params.get("target_network", "")
        target_host = canon_params.get("target_host", "")
        target_service = canon_params.get("target_service", "")

        if canon_action == "ScanNetwork" and target_net:
            self.scanned_networks.add(target_net)
            if not state_changed or not is_success:
                self.exhausted_entities.add(target_net)
        elif canon_action == "FindServices" and target_host:
            self.probed_service_hosts.add(target_host)
            if not is_success and target_host not in self.known_services:
                self.exhausted_entities.add(target_host)
        elif canon_action == "ExploitService" and target_host and target_service:
            if not is_success:
                self.failed_services_by_host.setdefault(target_host, set()).add(target_service)
        elif canon_action == "FindData" and target_host:
            self.searched_data_hosts.add(target_host)

    def is_action_redundant(self, action_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Return True ONLY when the exact canonical (ActionType, Parameters) pair
        has already been executed or the specific parameter target failed preconditions.
        """
        key = canonical_action_key(action_type, parameters)
        return key in self.executed_actions

    def _sync_history(self, episodic_memory: Optional[EpisodicMemoryBuffer]) -> List[str]:
        """Use EpisodicMemoryBuffer history if provided, otherwise internal action_history."""
        if episodic_memory is not None and episodic_memory.episode_action_history:
            return episodic_memory.episode_action_history
        return self.action_history

    def compute_temporal_metrics(
        self,
        action_type: str,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
    ) -> Dict[str, float]:
        """Compute current and prospective sliding-window density d_t(a) and consecutive streak c_t(a)."""
        canon = canonicalize_action_type(action_type)
        history = self._sync_history(episodic_memory)
        w = self.stealth_window

        current_window = history[-w:]
        next_window = (history[-(w - 1) :] if w > 1 else []) + [canon]

        current_density = current_window.count(canon) / float(w)
        projected_density = next_window.count(canon) / float(w)

        current_streak = 0
        for act in reversed(history):
            if act == canon:
                current_streak += 1
            else:
                break
        projected_streak = current_streak + 1
        episode_count = sum(1 for a in history if a == canon)

        return {
            "current_density": current_density,
            "projected_density": projected_density,
            "current_streak": float(current_streak),
            "projected_streak": float(projected_streak),
            "episode_count": float(episode_count),
            "window_len": float(len(history)),
        }

    HAZARD_SCALES: Tuple[int, ...] = (3, 5, 8)
    PROGRESS_GAIN_WEIGHTS: Dict[str, float] = {
        "ExfiltrateData": 1.00,
        "FindData": 0.80,
        "ExploitService": 0.62,
        "FindServices": 0.42,
        "ScanNetwork": 0.22,
    }

    def compute_multiscale_hazard(
        self,
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
    ) -> float:
        """
        Compute worst-case normalized sliding-window hazard max_{w in {3,5,8}} R_w(a | H_t)
        so the planner requires zero prior knowledge of the monitor's exact window W.
        """
        canon = canonicalize_action_type(action_type)
        history = self._sync_history(episodic_memory)
        max_hazard = 0.0

        for w in self.HAZARD_SCALES:
            next_window = (history[-(w - 1) :] if w > 1 else []) + [canon]
            density_w = next_window.count(canon) / float(w)

            consec_w = 0
            for act in reversed(next_window):
                if act == canon:
                    consec_w += 1
                else:
                    break

            if consec_w >= 3 or (consec_w >= 2 and canon in ("ScanNetwork", "ExfiltrateData", "FindData")):
                consec_hazard = 1.0
            elif consec_w >= 2:
                consec_hazard = 0.55
            else:
                consec_hazard = 0.0

            # ExfiltrateData spacing guard across window w
            exfil_guard = 1.0 if (canon == "ExfiltrateData" and history[-w:].count("ExfiltrateData") >= 1) else 0.0
            r_w = min(1.0, max(density_w, consec_hazard, exfil_guard))
            if r_w > max_hazard:
                max_hazard = r_w

        if parameters is not None and self.is_action_redundant(canon, parameters):
            max_hazard = min(1.0, max_hazard + 0.35)

        return round(min(1.0, max(0.0, max_hazard)), 4)

    def compute_minimax_hazard_utility(
        self,
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
        alpha: float = 1.0,
        beta: float = 1.2,
    ) -> Tuple[float, float, float]:
        """
        Evaluate U(a) = alpha * ProgressGain(a | G_t) - beta * max_{w in {3,5,8}} R_w(a | H_t).
        Returns (utility, progress_gain, max_hazard).
        """
        canon = canonicalize_action_type(action_type)
        p = parameters or {}
        gain = self.PROGRESS_GAIN_WEIGHTS.get(canon, 0.10)
        if canon == "ExfiltrateData" and self.goal_target_ip and p.get("target_host") == self.goal_target_ip:
            gain += 0.08
        elif canon == "FindServices":
            th = p.get("target_host", "")
            if th and th not in self.controlled_hosts and th not in self.probed_service_hosts:
                gain += 0.06
        if parameters is not None and self.is_action_redundant(canon, parameters):
            gain -= 1.0

        max_hazard = self.compute_multiscale_hazard(
            canon, parameters=parameters, episodic_memory=episodic_memory
        )
        utility = round(alpha * gain - beta * max_hazard, 4)
        return utility, round(gain, 4), max_hazard

    def score_action_stealth_risk(
        self,
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
    ) -> float:
        """
        Estimate normalized multi-scale SIEM detection hazard in [0.0, 1.0]
        via max_{w in {3,5,8}} R_w(a | H_t) and redundancy penalty.
        """
        return self.compute_multiscale_hazard(
            action_type=action_type,
            parameters=parameters,
            episodic_memory=episodic_memory,
        )

    def is_action_high_stealth_risk(
        self,
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
    ) -> bool:
        """
        Return True if executing `action_type` next would breach the W=5 sliding-window
        SIEM density or consecutive-streak threshold (matching GlobalDefender).
        """
        canon = canonicalize_action_type(action_type)
        history = self._sync_history(episodic_memory)
        w = self.stealth_window
        next_window = (history[-(w - 1) :] if w > 1 else []) + [canon]

        projected_density = next_window.count(canon) / float(w)
        ratio_thresh = self.SIEM_RATIO_THRESHOLDS.get(canon, 1.0)
        if projected_density >= ratio_thresh:
            return True

        # Frequency spacing for ExfiltrateData: avoid clustering >= 2 within last W steps
        if canon == "ExfiltrateData" and history[-w:].count("ExfiltrateData") >= 1:
            return True

        consec_thresh = self.SIEM_CONSECUTIVE_THRESHOLDS.get(canon)
        if consec_thresh is not None:
            max_consec = max(
                (sum(1 for _ in grp) for act, grp in groupby(next_window) if act == canon),
                default=0,
            )
            if max_consec >= consec_thresh:
                return True

        if parameters is not None and self.is_action_redundant(canon, parameters):
            return True

        return False

    def get_temporal_stealth_advisory(
        self,
        valid_actions: List[Dict[str, Any]],
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the current sliding window (W=5) across candidate actions and produce
        a Temporal Stealth Advisory highlighting noisy streaks vs low-footprint options.
        """
        history = self._sync_history(episodic_memory)
        window = history[-self.stealth_window :]

        density_by_type: Dict[str, float] = {}
        streak_by_type: Dict[str, int] = {}
        for act_type in sorted(set(window)):
            density_by_type[act_type] = round(window.count(act_type) / float(self.stealth_window), 2)
            streak = 0
            for prev in reversed(history):
                if prev == act_type:
                    streak += 1
                else:
                    break
            streak_by_type[act_type] = streak

        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
        elevated_types: Set[str] = set()
        low_risk_types: Set[str] = set()

        for cand in valid_actions:
            act_type = cand["action"]
            score = self.score_action_stealth_risk(
                act_type, cand.get("parameters", {}), episodic_memory=episodic_memory
            )
            cand_with_risk = {**cand, "stealth_risk": score}
            scored_candidates.append((score, cand_with_risk))
            if score >= 0.45:
                elevated_types.add(act_type)
            else:
                low_risk_types.add(act_type)

        if len(history) < self.stealth_window:
            window_status = f"WARMUP ({len(history)}/{self.stealth_window} actions logged)"
        elif elevated_types:
            window_status = "ELEVATED SIEM PRESSURE"
        else:
            window_status = "NOMINAL"

        return {
            "window_size": self.stealth_window,
            "recent_window": window,
            "density_by_type": density_by_type,
            "streak_by_type": streak_by_type,
            "window_status": window_status,
            "elevated_risk_types": sorted(elevated_types),
            "low_risk_types": sorted(low_risk_types),
            "scored_candidates": [c for _, c in scored_candidates],
        }

    def get_valid_action_space(
        self,
        obs_or_state: Any,
        include_executed: bool = False,
        apply_stealth_filter: Optional[bool] = None,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
    ) -> List[Dict[str, Any]]:
        """
        Derive all syntactically and semantically permissible (ActionType, Parameters)
        tuples strictly from the current state and general NetSecGame rules.
        Prunes ONLY exact (ActionType, Parameters) pairs already in executed_actions.
        """
        state = _extract_state(obs_or_state)
        if state is None:
            return []

        known_nets = sorted({normalize_network(n) for n in (getattr(state, "known_networks", ()) or ()) if normalize_network(n)})
        controlled = sorted({normalize_ip(h) for h in (getattr(state, "controlled_hosts", ()) or ()) if normalize_ip(h)})
        known_hosts = sorted(
            {normalize_ip(h) for h in (getattr(state, "known_hosts", ()) or ()) if normalize_ip(h)} | set(controlled)
        )
        blocks = _normalize_blocks(getattr(state, "known_blocks", {}))

        raw_services = getattr(state, "known_services", {}) or {}
        services_by_host: Dict[str, List[str]] = {}
        if isinstance(raw_services, dict):
            for host, srv_list in raw_services.items():
                h_str = normalize_ip(host)
                if not h_str:
                    continue
                valid_srvs = []
                for srv in srv_list or ():
                    if bool(getattr(srv, "is_local", False)):
                        continue
                    s_name = normalize_service(srv)
                    if s_name and s_name not in valid_srvs:
                        valid_srvs.append(s_name)
                if valid_srvs:
                    services_by_host[h_str] = sorted(valid_srvs)

        raw_data = getattr(state, "known_data", {}) or {}
        data_by_host: Dict[str, List[Tuple[str, str]]] = {}
        if isinstance(raw_data, dict):
            for host, d_list in raw_data.items():
                h_str = normalize_ip(host)
                if not h_str:
                    continue
                items = []
                for d in d_list or ():
                    pair = normalize_data(d)
                    if (pair[0] or pair[1]) and pair not in items:
                        items.append(pair)
                if items:
                    data_by_host[h_str] = sorted(items)

        if not controlled:
            return []

        controlled_set = set(controlled)
        candidates: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[Any, ...]] = set()

        def _maybe_add(action_name: str, params: Dict[str, Any]) -> None:
            canon_params = canonicalize_parameters(params)
            key = canonical_action_key(action_name, canon_params)
            if key in seen_keys:
                return
            if not include_executed and key in self.executed_actions:
                return
            seen_keys.add(key)
            candidates.append({"action": action_name, "parameters": canon_params})

        # 1. ExfiltrateData: any data on a controlled host -> another controlled host (goal target first if known)
        ordered_exfil_targets = sorted(
            controlled,
            key=lambda h: (0 if self.goal_target_ip and h == self.goal_target_ip else 1, h),
        )
        for src in controlled:
            for owner, data_id in data_by_host.get(src, []):
                for dst in ordered_exfil_targets:
                    if dst == src or _is_blocked(src, dst, blocks):
                        continue
                    _maybe_add(
                        "ExfiltrateData",
                        {
                            "source_host": src,
                            "target_host": dst,
                            "data": {"owner": owner, "id": data_id},
                        },
                    )

        # 2. FindData: search any controlled host for local data
        for src in controlled:
            _maybe_add(
                "FindData",
                {
                    "source_host": src,
                    "target_host": src,
                },
            )

        # 3. ExploitService: exploit non-passive services on uncontrolled known hosts
        for dst in known_hosts:
            if dst in controlled_set:
                continue
            for srv_name in services_by_host.get(dst, []):
                for src in controlled:
                    if _is_blocked(src, dst, blocks):
                        continue
                    _maybe_add(
                        "ExploitService",
                        {
                            "source_host": src,
                            "target_host": dst,
                            "target_service": srv_name,
                        },
                    )

        # 4. FindServices: probe known hosts (prioritize uncontrolled / unprobed hosts)
        for dst in known_hosts:
            for src in controlled:
                if _is_blocked(src, dst, blocks):
                    continue
                _maybe_add(
                    "FindServices",
                    {
                        "source_host": src,
                        "target_host": dst,
                    },
                )

        # 5. ScanNetwork: scan known CIDR subnets from a controlled host
        for net in known_nets:
            # Prefer a controlled host on the same subnet first so pivot order is natural.
            ordered_srcs = sorted(controlled, key=lambda h: (0 if _ip_in_network(h, net) else 1, h))
            for src in ordered_srcs:
                _maybe_add(
                    "ScanNetwork",
                    {
                        "source_host": src,
                        "target_network": net,
                    },
                )

        use_filter = self.enable_stealth_pacing if apply_stealth_filter is None else bool(apply_stealth_filter)
        if not use_filter or not candidates:
            self.last_stealth_deferred_count = 0
            self.last_stealth_deadlock_bypass = False
            return candidates

        history = self._sync_history(episodic_memory)
        last_action = history[-1] if history else None

        # Principle (a): Consecutive-Streak Breaking — break recon bursts when exploit/data actions exist.
        recon_types = ("FindServices", "ScanNetwork")
        depth_types = ("ExfiltrateData", "FindData", "ExploitService")
        if last_action in recon_types and any(c["action"] in depth_types for c in candidates):
            pool = [c for c in candidates if c["action"] not in recon_types]
        else:
            pool = candidates

        # Soft-mask actions breaching W=5 SIEM thresholds when lower-risk options exist in `pool`.
        safe_actions = [
            c for c in pool
            if not self.is_action_high_stealth_risk(
                c["action"], c.get("parameters", {}), episodic_memory=episodic_memory
            )
        ]
        if safe_actions:
            active_pool = safe_actions
            self.last_stealth_deadlock_bypass = False
        else:
            active_pool = pool
            self.last_stealth_deadlock_bypass = True

        self.last_stealth_deferred_count = len(candidates) - len(active_pool)

        # Principle (b) Depth-First Kill-Chain Locality + Principle (c) Sliding-Window Least-Frequent Action Type.
        kill_chain_tier = {
            "ExfiltrateData": 0,
            "FindData": 1,
            "ExploitService": 2,
            "FindServices": 3,
            "ScanNetwork": 4,
        }
        recent_w = history[-self.stealth_window :]

        def _schedule_key(cand: Dict[str, Any]) -> Tuple[float, int, str]:
            act = cand["action"]
            p = cand["parameters"]
            utility, _, _ = self.compute_minimax_hazard_utility(
                act, p, episodic_memory=episodic_memory
            )
            tier = kill_chain_tier.get(act, 99)
            return (-utility, tier, str(sorted(p.items())))

        return sorted(active_pool, key=_schedule_key)

    def classify_topology(
        self,
        obs_or_state: Any,
        goal_description: str = "",
    ) -> Dict[str, Any]:
        """
        Classify every discovered subnet, host, service, and data item into generic
        epistemic states (UNEXPLORED, MAPPED, COMPROMISED, EXHAUSTED).
        """
        self.update_from_observation(obs_or_state, goal_description=goal_description)
        state = _extract_state(obs_or_state)

        known_nets = sorted(self.known_networks)
        controlled = sorted(self.controlled_hosts)
        known_hosts = sorted(self.known_hosts)
        valid_actions = (
            self.get_valid_action_space(state, apply_stealth_filter=False)
            if state is not None
            else []
        )

        # Map which hosts/nets still have remaining permissible actions.
        remaining_actions_by_net: Dict[str, int] = {}
        remaining_exploits_by_host: Dict[str, int] = {}
        remaining_find_services_by_host: Dict[str, int] = {}
        remaining_find_data_by_host: Dict[str, int] = {}
        remaining_exfil_by_host: Dict[str, int] = {}

        for act in valid_actions:
            atype = act["action"]
            p = act["parameters"]
            if atype == "ScanNetwork":
                net = p.get("target_network", "")
                remaining_actions_by_net[net] = remaining_actions_by_net.get(net, 0) + 1
            elif atype == "ExploitService":
                th = p.get("target_host", "")
                remaining_exploits_by_host[th] = remaining_exploits_by_host.get(th, 0) + 1
            elif atype == "FindServices":
                th = p.get("target_host", "")
                remaining_find_services_by_host[th] = remaining_find_services_by_host.get(th, 0) + 1
            elif atype == "FindData":
                th = p.get("target_host", "")
                remaining_find_data_by_host[th] = remaining_find_data_by_host.get(th, 0) + 1
            elif atype == "ExfiltrateData":
                sh = p.get("source_host", "")
                remaining_exfil_by_host[sh] = remaining_exfil_by_host.get(sh, 0) + 1

        subnets: Dict[str, Dict[str, Any]] = {}
        for net in known_nets:
            member_hosts = [h for h in known_hosts if _ip_in_network(h, net)]
            if net not in self.scanned_networks:
                status = EpistemicState.UNEXPLORED.value
            elif remaining_actions_by_net.get(net, 0) == 0 and all(
                (h in controlled or remaining_exploits_by_host.get(h, 0) == 0)
                for h in member_hosts
            ):
                status = EpistemicState.EXHAUSTED.value
            else:
                status = EpistemicState.MAPPED.value
            subnets[net] = {
                "state": status,
                "hosts": member_hosts,
                "remaining_scans": remaining_actions_by_net.get(net, 0),
            }

        hosts: Dict[str, Dict[str, Any]] = {}
        for h in known_hosts:
            srvs = sorted(self.known_services.get(h, set()))
            data_items = sorted(self.known_data.get(h, set()))
            if h in self.controlled_hosts:
                # If searched for data and no exfil remains, distinguish MAPPED/EXHAUSTED detail cleanly while state is COMPROMISED.
                h_state = EpistemicState.COMPROMISED.value
                if h in self.searched_data_hosts and remaining_exfil_by_host.get(h, 0) == 0 and h != self.goal_target_ip:
                    detail = "data_searched"
                elif h == self.goal_target_ip:
                    detail = "goal_destination"
                elif remaining_find_data_by_host.get(h, 0) > 0:
                    detail = "unsearched_for_data"
                else:
                    detail = "exfil_pending"
            else:
                failed_srvs = self.failed_services_by_host.get(h, set())
                has_unexploited_srv = any(s not in failed_srvs for s in srvs)
                if h not in self.probed_service_hosts and not srvs:
                    h_state = EpistemicState.UNEXPLORED.value
                    detail = "unprobed"
                elif srvs and has_unexploited_srv and remaining_exploits_by_host.get(h, 0) > 0:
                    h_state = EpistemicState.MAPPED.value
                    detail = "exploitable_surface_mapped"
                elif h in self.probed_service_hosts and not has_unexploited_srv:
                    h_state = EpistemicState.EXHAUSTED.value
                    detail = "all_services_exhausted"
                else:
                    h_state = EpistemicState.MAPPED.value
                    detail = "probed"

            hosts[h] = {
                "state": h_state,
                "detail": detail,
                "services": srvs,
                "data": [{"owner": o, "id": d} for o, d in data_items],
                "is_goal_target": bool(self.goal_target_ip and h == self.goal_target_ip),
            }

        services: Dict[str, Dict[str, str]] = {}
        for h in known_hosts:
            for s in sorted(self.known_services.get(h, set())):
                key = f"{h}:{s}"
                if h in self.controlled_hosts:
                    services[key] = EpistemicState.COMPROMISED.value
                elif s in self.failed_services_by_host.get(h, set()) and remaining_exploits_by_host.get(h, 0) == 0:
                    services[key] = EpistemicState.EXHAUSTED.value
                else:
                    services[key] = EpistemicState.UNEXPLORED.value

        data_nodes: Dict[str, Dict[str, Any]] = {}
        for h, items in sorted(self.known_data.items()):
            for owner, did in sorted(items):
                dkey = f"{h}:{owner}:{did}"
                exfil_done = any(
                    a[0] == "ExfiltrateData"
                    and any(k == "source_host" and v == h for k, v in a[1])
                    and any(k == "data" and ("id", did) in v for k, v in a[1])
                    for a in self.succeeded_actions
                )
                if exfil_done:
                    d_state = EpistemicState.EXHAUSTED.value
                elif h == self.goal_target_ip:
                    d_state = EpistemicState.MAPPED.value
                else:
                    d_state = EpistemicState.UNEXPLORED.value
                data_nodes[dkey] = {
                    "state": d_state,
                    "host": h,
                    "owner": owner,
                    "id": did,
                }

        return {
            "goal_target_ip": self.goal_target_ip,
            "subnets": subnets,
            "hosts": hosts,
            "services": services,
            "data": data_nodes,
        }

    def _select_prompt_actions(
        self,
        scored_actions: List[Dict[str, Any]],
        max_display: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Keep prompt action choices concise and diverse by prioritizing distinct
        (action_type, primary_target) signatures before showing alternate source pivots.
        Never forces a single action.
        """
        primary: List[Dict[str, Any]] = []
        secondary: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Any, ...]] = set()

        for act in scored_actions:
            atype = act["action"]
            p = act["parameters"]
            sig: Tuple[Any, ...]
            if atype == "ScanNetwork":
                sig = (atype, p.get("target_network"))
            elif atype == "FindServices":
                sig = (atype, p.get("target_host"))
            elif atype == "ExploitService":
                sig = (atype, p.get("target_host"), p.get("target_service"))
            elif atype == "FindData":
                sig = (atype, p.get("target_host"))
            elif atype == "ExfiltrateData":
                sig = (atype, p.get("source_host"), p.get("target_host"), p.get("data", {}).get("id"))
            else:
                sig = (atype, str(p))

            if sig not in seen_signatures:
                seen_signatures.add(sig)
                primary.append(act)
            else:
                secondary.append(act)

        combined = primary + secondary
        return combined[:max_display]

    def format_context_prompt(
        self,
        obs_or_state: Any,
        episodic_memory: Optional[EpisodicMemoryBuffer] = None,
        goal_description: str = "",
    ) -> str:
        """
        Build a structured domain-agnostic prompt containing:
        1. Goal Objective (parsed dynamically from observations)
        2. Topological Belief Graph (UNEXPLORED / MAPPED / COMPROMISED / EXHAUSTED)
        3. Temporal Stealth Advisory (sliding-window SIEM density & risk)
        4. Permissible Unexplored Actions (choice set, zero single-action directives)
        5. Recent Episodic Trace
        """
        belief = self.classify_topology(obs_or_state, goal_description=goal_description)
        valid_actions = self.get_valid_action_space(obs_or_state, episodic_memory=episodic_memory)
        deferred_count = self.last_stealth_deferred_count
        advisory = self.get_temporal_stealth_advisory(valid_actions, episodic_memory=episodic_memory)
        display_actions = self._select_prompt_actions(advisory["scored_candidates"])

        lines: List[str] = []

        # 1. Mission Objective
        goal_text = self.goal_description or goal_description or "Compromise network assets and complete mission objective."
        target_note = f" | Parsed Exfiltration Destination: {self.goal_target_ip}" if self.goal_target_ip else ""
        lines.append(f"=== MISSION OBJECTIVE ===\nGoal: {goal_text}{target_note}")

        # 2. Topological Belief Graph
        lines.append("\n=== TOPOLOGICAL BELIEF GRAPH ===")
        lines.append("Subnets:")
        if belief["subnets"]:
            for net, info in belief["subnets"].items():
                hosts_str = ", ".join(info["hosts"]) if info["hosts"] else "none discovered"
                lines.append(f"  - {net} [{info['state']}] (known hosts: {hosts_str})")
        else:
            lines.append("  - None discovered")

        lines.append("Hosts:")
        if belief["hosts"]:
            for h, info in belief["hosts"].items():
                tags = [info["state"], info["detail"]]
                if info["is_goal_target"]:
                    tags.append("EXFIL_TARGET")
                srv_str = f"services={info['services']}" if info["services"] else "services=[]"
                data_str = (
                    f"data={[f'{d['owner']}:{d['id']}' for d in info['data']]}"
                    if info["data"]
                    else "data=[]"
                )
                lines.append(f"  - {h} [{' | '.join(tags)}] {srv_str} {data_str}")
        else:
            lines.append("  - None discovered")

        if self.known_blocks:
            blk_parts = [f"{h} <-> {sorted(peers)}" for h, peers in sorted(self.known_blocks.items()) if peers]
            if blk_parts:
                lines.append(f"Observed Firewall Blocks: {'; '.join(blk_parts)}")

        # 3. Temporal Stealth Advisory
        lines.append("\n=== TEMPORAL STEALTH ADVISORY (SLIDING WINDOW W=5) ===")
        lines.append(
            f"SIEM Pressure: {advisory['window_status']} | "
            f"Recent Window: {advisory['recent_window'] or ['<empty>']}"
        )
        if advisory["density_by_type"]:
            dens_str = ", ".join(
                f"{k}: d={advisory['density_by_type'][k]:.2f}, streak={advisory['streak_by_type'][k]}"
                for k in sorted(advisory["density_by_type"].keys())
            )
            lines.append(f"Window Metrics: {dens_str}")
        if advisory["elevated_risk_types"]:
            lines.append(
                f"High-Detection-Risk Action Types (burst/density alert): {advisory['elevated_risk_types']}."
            )
        if advisory["low_risk_types"]:
            lines.append(
                f"Low-Footprint Diversified Options Available: {advisory['low_risk_types']}."
            )
        if deferred_count > 0:
            lines.append(
                f"Active Stealth Pacing: {deferred_count} high-risk action(s) exceeding W={self.stealth_window} "
                "SIEM thresholds have been temporarily deferred to preserve stealth."
            )
        lines.append(
            "Tradeoff Guidance: Prefer depth-first, alternating action types — exploit mapped services "
            "and collect data on discovered/compromised targets before scanning additional hosts or "
            "subnets — to evade SIEM burst detection."
        )

        # 4. Permissible Unexplored Actions (Stealth-Scheduled when pacing is active)
        section_header = (
            "=== STEALTH-SCHEDULED PERMISSIBLE UNEXPLORED ACTIONS ==="
            if self.enable_stealth_pacing
            else "=== PERMISSIBLE UNEXPLORED ACTIONS ==="
        )
        lines.append(f"\n{section_header}")
        if display_actions:
            for idx, act in enumerate(display_actions, start=1):
                lines.append(
                    f"  {idx}. action={act['action']} | parameters={act['parameters']} "
                    f"| stealth_risk={act.get('stealth_risk', 0.0):.2f}"
                )
            omitted = len(advisory["scored_candidates"]) - len(display_actions)
            if omitted > 0:
                lines.append(f"  ... ({omitted} additional valid source-pivot permutations omitted for brevity)")
        else:
            lines.append("  - No unexplored permissible actions remain in the current belief graph.")

        # 5. Recent Episodic Trace
        if episodic_memory is not None:
            recent = episodic_memory.get_recent_steps(5)
            lines.append("\n=== RECENT EPISODIC TRACE ===")
            if recent:
                for step in recent:
                    det_tag = " [DETECTED]" if step.detected else ""
                    lines.append(
                        f"  Step {step.step_num}: {step.action_type} {step.parameters} -> "
                        f"{step.outcome_status} (r={step.reward:.1f}, risk={step.stealth_risk:.2f}{det_tag}) | "
                        f"{step.state_delta_summary}"
                    )
            else:
                lines.append("  - Episode start (no prior actions).")

        lines.append(
            "\nSelect ONE entry from the permissible actions list above (do not repeat actions from "
            "the Recent Episodic Trace or omit required parameter keys) and output strictly a single "
            "JSON object with keys \"action\" and \"parameters\"."
        )
        return "\n".join(lines)
