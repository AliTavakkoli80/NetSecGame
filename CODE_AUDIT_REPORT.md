# Methodological & Code Audit Report: `research_extension/`, Scenario Configurations, and Benchmark Traces

**Audit Standard:** Elsevier *Expert Systems with Applications* (ESWA) Peer Review & AI Systems Methodology Audit  
**Scope:** Read-only inspection of `research_extension/`, `scenarios/`, `logs/benchmarks/`, and comparative parity against `NetSecGame` / `NetSecGameAgents` (Rigaki et al., ESWA 2026)  
**Overall Verdict:** **DESK-REJECT RISK (CRITICAL FLAWS PRESENT)** — Do **NOT** launch large-scale experiments or draft paper claims until all **CRITICAL** findings below are remediated.

---

## Executive Verdict Summary

| Audit Domain | Verdict | Key Finding |
| :--- | :---: | :--- |
| **1. LLM Autonomy & Heuristic Hijacking** | **CRITICAL** | The Python wrapper operates as a **double-lock deterministic Finite State Machine (FSM)**: `get_tactical_phase_and_guidance()` injects the exact next JSON action into the prompt (`Schema: {...}`), and `prune_and_recover_action()` unilaterally substitutes the same 1-element `candidates[0]` on fallback. Both `1.5B` and `7B-NF4` execute identical deterministic trajectories. |
| **1a. Fallback/Override Rate Telemetry** | **CRITICAL** | **Unobservable (`N/A`) in `logs/benchmarks/*.csv` and `*_metrics.json`.** Neither `StepMetric` nor `plan_next_action()` records whether `prune_and_recover_action()` fired; `benchmark_runner.py` overwrites `is_valid=True` and recomputes `is_redundant` on the post-override action. |
| **2. Hardcoding & Scenario Leakage** | **CRITICAL** | `memory_module.py` and `benchmark_runner.py` bake in specific IPs (`213.47.23.195`, `192.168.1.1`, `192.168.1.2`, `192.168.2.2`, `192.168.3.2`), subnets (`192.168.1/2/3.`), service priorities (`microsoft-ds`, `active-directory`), target data (`User1`/`DataFromServer1`, `admin`/`passwords`), and topology-specific pivot rules (`FindData` suppression on `192.168.1.2`). |
| **3. Experimental Protocol Parity (Rigaki et al.)** | **CRITICAL** | Static seeds (`seed=42`), `use_dynamic_addresses: False`, `randomize_topology=False`, static goals/start hosts, truncated horizon (`max_steps: 50` vs. `100`), tiny sample size (`5` episodes vs. `150`), no 95% confidence intervals, and `RandomAttacker` omitted from all `GlobalDefender` scenarios. |
| **4. Code Correctness Bugs** | **MODERATE** | Duplicate `reset_episode()` shadows clearing of `failed_exploits`/`initial_foothold`; `str` vs. `IP` dict lookups always miss (`known_d.get(h)`, `known_s.get(h)`); `Config_B_RawSLM` in `benchmark_runner.py` omits `enable_action_masking=False`; `DEFAULT_SYSTEM_PROMPT` calls an Attacker a `"defensive cyber security agent"`. |
| **5. Infrastructure & Core Interfaces** | **PASS** | `HuggingFaceSLMBackend` quantization/CUDA timing, `MetricsLogger` VRAM/TPS capture, `_extract_json_action()` balanced-brace parsing, and `NetSecGame` `GlobalDefender` integration (`use_global_defender: True`) are structurally sound once autonomy/protocol issues are fixed. |

---

## 1. Heuristic Hijacking vs. Genuine SLM Reasoning (LLM Autonomy Audit)

### 1.1 Finding [CRITICAL]: Double-Lock Deterministic FSM Masquerading as SLM Reasoning

Inspection of `research_extension/base_agent.py` and `research_extension/memory_module.py` confirms that the agent does **not** perform autonomous multi-step reasoning or general-purpose action masking. Instead, two coupled mechanisms enforce a deterministic 5-phase Finite State Machine (FSM):

1. **Lock 1 — Pre-Inference Prompt Spoon-Feeding (`format_context_prompt()` -> `get_tactical_phase_and_guidance()`):**
   - In `research_extension/base_agent.py` (lines 422–427), whenever `use_memory=True`, `plan_next_action()` calls `self.semantic_memory.format_context_prompt(self.memory_buffer, observation.state, goal_desc)`.
   - In `research_extension/memory_module.py` (lines 470–473), `format_context_prompt()` invokes `self.get_tactical_phase_and_guidance(state, goal_description)` and injects the exact next decision into the user prompt:
     - Line 472: `lines.append(f"TACTICAL DIRECTIVE ({phase_name.upper()}):")`
     - Line 473: `lines.append(f">>> {guidance}\n")`
   - Every branch of `get_tactical_phase_and_guidance()` (`memory_module.py` lines 269–452) computes the exact single next action (`target_network`, `target_host`, `source_host`, `target_service`, `data`) and embeds a verbatim copy-pasteable JSON string (`Schema: {"action": ..., "parameters": ...}`) directly inside `guidance` (lines 307, 329, 344, 391, 422, 446), even stating in Phase 5 (line 308): `"All other actions are strictly masked."`
   - Combined with greedy decoding (`SLMConfig.do_sample: bool = False` in `research_extension/base_agent.py` line 41), the SLM is reduced to a **JSON copy-echo transcriber** rather than a reasoning policy.

2. **Lock 2 — Post-Inference Unilateral Override (`prune_and_recover_action()`):**
   - In `research_extension/base_agent.py` (lines 458–471), when `self.enable_action_masking` is `True`, any SLM output that fails validation (`not is_valid` or `action is None`) or repeats a prior action (`is_redundant`) triggers:
     ```python
     pruned_act, pruned_dict, prune_msg = self.prune_and_recover_action(observation.state, goal_desc)
     ```
   - `prune_and_recover_action()` (`base_agent.py` lines 403–414) calls `self.semantic_memory.get_tactical_phase_and_guidance(state, eff_goal)` (line 409) and returns `candidates[0]` (lines 410–413).
   - Crucially, `get_tactical_phase_and_guidance()` **never returns a mask over multiple valid actions**: in all six tactical branches (`memory_module.py` lines 310–317, 331–335, 346–350, 393–400, 424–427, 448–451), `candidates` is a **1-element list** containing the single hardcoded FSM transition (or `[]` at line 454). This is not action masking; it is a **hardcoded fallback policy**.

---

### 1.2 Finding [CRITICAL]: Fallback / Override Rate Is Unobservable in All Saved Benchmark Logs

Parsing all 11 step-level CSV files (1,048 step rows) and 11 JSON metric files in `logs/benchmarks/` reveals a **complete telemetry blindspot**:

- **Missing Telemetry Columns:** `StepMetric` (`research_extension/metrics_logger.py` lines 24–39) and `export_csv()` (lines 254–259) only record:
  `episode_idx, step_idx, action_type, is_valid, state_changed, reward, inference_latency_ms, env_step_latency_ms, gpu_memory_allocated_mb, gpu_memory_peak_mb, token_count, tokens_per_second, is_redundant, timestamp`.
- **Silent Overwrite Before Telemetry Capture:**
  - `ModernSLMAttacker.plan_next_action()` (`research_extension/base_agent.py` lines 463–473) overwrites `action = pruned_act` and `parsed_dict = pruned_dict`, emits only an unpersisted `self.logger.info(...)` line (no file handler is configured in `benchmark_runner.py`), and returns `(action, parsed_dict, latency_sec, token_count)` with **no override flag and no raw SLM output**.
  - `BenchmarkHarness.run_slm_agent()` (`research_extension/benchmark_runner.py` lines 112–149) then sets `is_valid = True` whenever `action is not None` (line 119) and recomputes `is_redundant` on the **post-override** `action_name` and `raw_dict` (lines 134–136).

**Computed Telemetry Across `logs/benchmarks/*.csv`:**

| Benchmark CSV | Episodes | Total Steps | Reported `is_valid=True` | Reported `is_redundant=True` | Override / Fallback Column Present? | Observable Fallback Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Adv_FullScenario_Defender_Qwen1.5B_steps.csv` | 5 | 25 | 25/25 (100.0%) | 0/25 (0.0%) | **No** | **Unobservable (`N/A`)** |
| `Adv_FullScenario_Defender_Qwen7B_NF4_steps.csv` | 5 | 25 | 25/25 (100.0%) | 0/25 (0.0%) | **No** | **Unobservable (`N/A`)** |
| `Adv_ThreeNets_Defender_Qwen1.5B_steps.csv` | 5 | 34 | 34/34 (100.0%) | 0/34 (0.0%) | **No** | **Unobservable (`N/A`)** |
| `Adv_ThreeNets_Defender_Qwen7B_NF4_steps.csv` | 5 | 34 | 34/34 (100.0%) | 0/34 (0.0%) | **No** | **Unobservable (`N/A`)** |
| `Condition_1_Qwen1.5B_Masked_steps.csv` | 5 | 40 | 40/40 (100.0%) | 0/40 (0.0%) | **No** | **Unobservable (`N/A`)** |
| `Condition_2_Qwen7B_NF4_Masked_steps.csv` | 5 | 40 | 40/40 (100.0%) | 0/40 (0.0%) | **No** | **Unobservable (`N/A`)** |
| `Config_A_RandomBaseline_steps.csv` | 5 | 250 | 250/250 (100.0%) | 175/250 (70.0%) | **No** | `N/A` (Random baseline) |
| `Config_B_RawSLM_steps.csv` | 5 | 250 | 250/250 (100.0%) | 240/250 (96.0%) | **No** | `N/A` (Pre-masking run) |
| `Config_C_MemoryAugmentedSLM_steps.csv` | 5 | 250 | 250/250 (100.0%) | 240/250 (96.0%) | **No** | `N/A` (Pre-masking run) |
| `baseline_random_steps.csv` | 2 | 100 | 100/100 (100.0%) | *(col absent)* | **No** | `N/A` |
| `slm_mock_steps.csv` | 2 | 100 | 100/100 (100.0%) | *(col absent)* | **No** | `N/A` |

**Peer-Review Impact:** Reporting `100.0% Action Validity Rate` and `0.0% Redundant Action Rate` for Conditions 1–2 and `Adv_*` while silently replacing invalid/redundant SLM outputs via `prune_and_recover_action()` without logging the raw SLM failure/override rate constitutes **metric misrepresentation** and will trigger immediate rejection. Notably, in `Config_B_RawSLM_steps.csv` and `Config_C_MemoryAugmentedSLM_steps.csv` (which ran at `10:26–10:29` prior to the `11:36–12:43` action-masking edits), the 1.5B SLM achieved **0.0% Win Rate** and **96.0% redundancy (240/250 steps)**, stalling in `ScanNetwork` (`Config_B`) or `FindServices`/`ScanNetwork` loops (`Config_C`).

---

### 1.3 Why `Qwen2.5-Coder-1.5B` and `Qwen2.5-Coder-7B-NF4` Yielded Identical Metrics

Across the adversarial suite (`logs/benchmarks/adversarial_suite_summary.json`), both `Qwen2.5-Coder-1.5B` and `Qwen2.5-Coder-7B-NF4` produced **bit-for-bit identical behavioral outcomes**:
- **Full Scenario + Defender:** `100.0%` Win Rate, `0.0%` Detection Rate, `5.0` steps/episode, `+95.00` return for **both** models.
- **Three-Networks + Defender:** `40.0%` Win Rate (`2/5` wins), `60.0%` Detection Rate (`3/5` detected), `6.8` steps/episode, `+27.20` return for **both** models.

Step-by-step inspection of the CSV traces explains this exact equivalence:

1. **Identical Deterministic FSM Action Sequences:**
   - In `Adv_FullScenario_Defender_*_steps.csv`, every single episode (1–5) of both `1.5B` and `7B_NF4` executes the exact 5-step sequence: `['ScanNetwork', 'FindServices', 'ExploitService', 'FindData', 'ExfiltrateData']` with identical token counts per step (`[61, 58, 74, 58, 87]` for 1.5B; `[52, 49, 64, 49, 73]` for 7B-NF4).
   - In `Adv_ThreeNets_Defender_*_steps.csv`, **both** models fail at **Step 6** in Episodes **1, 4, and 5** (`['ScanNetwork', 'FindServices', 'ExploitService', 'ScanNetwork', 'FindServices', 'ExploitService']`, reward `-11`) and win at **Step 8** in Episodes **2 and 3** (`[..., 'FindData', 'ExfiltrateData']`, reward `99`).
   - Either the SLM greedily echoed (`do_sample=False`) the exact JSON `Schema` dictated in `TACTICAL DIRECTIVE`, or `prune_and_recover_action()` forced `candidates[0]`. In both cases, **model scale (1.5B vs. 7B) had zero causal effect on action selection**.

2. **Identical Fixed PRNG Stream (`seed=42`) in `GlobalDefender`:**
   - `BenchmarkHarness.__init__()` (`benchmark_runner.py` line 49) defaults to `seed=42`, and `start_server()` passes `seed=42` to `NetSecGame(..., seed=seed)` (lines 35, 65). `reset_environment()` (`base_agent.py` line 287) calls `request_game_reset()` without changing the seed.
   - In `ThreeNets_Defender`, `GlobalDefender.stochastic_with_threshold()` (`NetSecGame/netsecgame/game/global_defender.py` lines 61–103, `tw_size=5`) triggers `random.random()` twice per episode before Step 7:
     - **Step 5 (`FindServices`):** `tw_ratio = 2/5 = 0.40 >= 0.30` -> calls `random.random() < 0.075`. *(Note: `EXPERIMENT_RESULTS.md` line 384 overlooked this Step 5 roll.)*
     - **Step 6 (`ExploitService`):** `tw_ratio = 2/5 = 0.40 >= 0.25` -> calls `random.random() < 0.10`.
   - With `random.seed(42)`, the first 10 `random.random()` draws are:
     `[0.6394, 0.0250, 0.2750, 0.2232, 0.7365, 0.6767, 0.8922, 0.0869, 0.4219, 0.0298]`.
   - Because both models execute the exact same FSM sequence from the same `seed=42` server state, the 2nd (Ep 1 Step 6 = `0.0250 < 0.10`), 8th (Ep 4 Step 6 = `0.0869 < 0.10`), and 10th (Ep 5 Step 6 = `0.0298 < 0.10`) draws trigger detection in **Episodes 1, 4, and 5** for **both** `1.5B` and `7B-NF4`, while Episodes 2 and 3 pass! Thus, the `40.0%` win rate is a deterministic artifact of `seed=42` coupled to an FSM trajectory, not an evaluation of 1.5B vs. 7B reasoning.

*(Note on `Condition_1` vs. `Condition_2`: `Condition_1` ran at `11:34` and `Condition_2` at `11:46` on Nov 5, prior to the `12:42–12:46` edits to `memory_module.py`/`base_agent.py`/`benchmark_runner.py`. Even there, both executed 4 `ScanNetwork` steps followed by `FindServices -> ExploitService -> FindData -> ExfiltrateData` for identical 8.0 steps and +92.00 returns, differing only in which subnet index hit `192.168.1.0/24` first.)*

---

## 2. Hardcoding & Scenario Leakage Audit

Every file in `research_extension/` was scanned line-by-line for hardcoded IP addresses, CIDR ranges, service names, data identifiers, and topology-specific branching.

### 2.1 Complete Inventory of Hardcoded Domain Knowledge

| Severity | File & Line(s) | Hardcoded Symbol / Value | Scenario Leakage / Defect Description |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | `research_extension/memory_module.py:137` | `"213.47.23.195"` | Hardcodes C2 server IP to exclude it when inferring `self.initial_foothold` from `state.controlled_hosts`. Breaks if C2 IP changes (e.g., `54.123.53.21` in `netsecenv_conf_new_ips_1.yaml`) or `use_dynamic_addresses: True` is enabled. |
| **CRITICAL** | `research_extension/memory_module.py:256, 260` | `c2_ip = "213.47.23.195"`, fallback `"192.168.1.1"` | Hardcodes C2 destination IP for `ExfiltrateData` and fallback foothold IP instead of parsing goal/state dynamically. |
| **CRITICAL** | `research_extension/memory_module.py:262–266` | `is_three_nets = ("password" in goal_description.lower() or any("192.168.3." in net ...) or "192.168.2.2" in controlled)` | **Scenario Classifier Leakage:** Explicitly branches agent policy on scenario-specific IP prefixes (`192.168.3.`, `192.168.2.2`) and goal keywords (`"password"`). |
| **CRITICAL** | `research_extension/memory_module.py:274–303` | `"passwords"`, `"password"`, `"admin"`, `"DataFromServer1"`, `"User1"`, `"datafromserver1"` | Hardcodes target data `id` and `owner` strings for both scenarios, including fallback defaults (`"admin"`/`"passwords"` vs. `"User1"`/`"DataFromServer1"`) at lines 301–302. |
| **CRITICAL** | `research_extension/memory_module.py:321–325` | `if is_three_nets and h == "192.168.1.2": continue` | **Topology Leakage (Oracle Shortcut):** Explicitly suppresses `FindData` on intermediate pivot `192.168.1.2` so the agent skips decoy `DataFromServer1` and avoids extra steps/SIEM exposure. |
| **CRITICAL** | `research_extension/memory_module.py:337–350` | `"192.168.1.2"` -> `"192.168.3.0/24"` | **Topology Leakage (Oracle Pivot):** Hardcodes the exact lateral-movement edge (`192.168.1.2` scans `192.168.3.0/24`) immediately upon compromising `192.168.1.2`. |
| **CRITICAL** | `research_extension/memory_module.py:369` | `["microsoft-ds", "active-directory", "ms-wbt-server", "lanman", "smb"]` | Hardcodes the exact vulnerable/target service priority order from `scenario1` and `three_net_scenario`, bypassing decoy services like `http` or `ssh` without probing. |
| **CRITICAL** | `research_extension/memory_module.py:377–386` | `tgt_h == "192.168.1.2"` (prio 0), `"192.168.3.2"` (prio 1), `"192.168.3."` source `"192.168.1.2"` | Hardcodes target host priority and multi-hop routing source IP for `ExploitService`, skipping decoy hosts `192.168.1.3–1.6` and `192.168.3.3` on Step 3/6. |
| **CRITICAL** | `research_extension/memory_module.py:409–418` | `h == "192.168.1.2"` (prio 0), `"192.168.3.2"` (prio 1), `"192.168.1."` (prio 2), source `"192.168.1.2"` | Hardcodes `ScanServices` target priority to scan `192.168.1.2` first among 5 discovered hosts, directly preventing the `FindServices` burst that triggers `GlobalDefender` (`_TW_CONSECUTIVE_TYPE_THRESHOLD[FindServices] = 3`). |
| **CRITICAL** | `research_extension/memory_module.py:433–442` | `"192.168.1."` (prio 0), `"192.168.3."` (prio 1/10), `"192.168.2."` (prio 2), source `"192.168.1.2"` | Hardcodes `ScanNetwork` subnet priority so `192.168.1.0/24` is scanned on Step 1 instead of decoy subnets `192.168.0.0/24`, `192.168.2.0/24`, or `213.47.23.192/26`. |
| **CRITICAL** | `research_extension/benchmark_runner.py:504, 506, 535` | `"Exfiltrate DataFromServer1 from Samba server to C2 (213.47.23.195)"`, `"Exfiltrate password DB data to C2 (213.47.23.195)"` | Overrides the environment's `obs.info["goal_description"]` with custom runner-injected strings (which also leak `"Samba server"`, `"DataFromServer1"`, `"password DB"` and trigger `is_three_nets` at `memory_module.py:263`). |
| **MODERATE** | `research_extension/base_agent.py:250–254` | `{"target_network": "192.168.1.0/24", "source_host": "192.168.1.1"}` | Hardcoded fallback IP/subnet in `MockSLMBackend.generate()`. |
| **MODERATE** | `research_extension/base_agent.py:324, 342` | `"autonomous defensive cyber security agent"`, `"Scan network -> Scan services -> ..."` | `DEFAULT_SYSTEM_PROMPT` mislabels `ModernSLMAttacker` as a `"defensive"` agent and embeds a fixed linear attack script. |
| **PASS / Minor** | `research_extension/verify_7b.py:61` | `"192.168.1.0/24"` | Standalone smoke-test script for 7B VRAM loading; not part of agent decision logic. |

### 2.2 Additional Code Defects Discovered During Audit

1. **Duplicate `reset_episode()` Shadowing (`MODERATE`, `research_extension/memory_module.py:119–129` vs. `502–510`):**
   - `SemanticMemoryStore.reset_episode` is defined twice. The second definition at lines 502–510 overwrites the first at lines 119–129 and **omits** `self.failed_exploits.clear()` and `self.initial_foothold = None`. Under dynamic topologies or multi-episode runs, `initial_foothold` and `failed_exploits` leak across episodes.
2. **Broken `str` vs. `IP` Dictionary Lookups (`MODERATE`, `research_extension/memory_module.py:323, 406`):**
   - `known_d.get(h, set())` and `known_s.get(h, set())` pass `h: str`, whereas `GameState.known_data` and `GameState.known_services` are keyed by `netsecgame.game_components.IP`. Because `IP.__eq__` (`NetSecGame/netsecgame/game_components.py:75–87`) returns `NotImplemented` for `str`, these `.get()` calls always return empty `set()`.
3. **Broken Ablation Flag Default (`CRITICAL`, `research_extension/benchmark_runner.py:599–604` & `base_agent.py:353`):**
   - `ModernSLMAttacker.__init__` defaults `enable_action_masking: bool = True` (`base_agent.py:353`), and `benchmark_runner.py` lines 599–604 instantiate `Config_B_RawSLM` without passing `enable_action_masking=False`. Re-running `--mode tri_ablation` on the current codebase would silently run `Config_B_RawSLM` with FSM masking enabled!
4. **Dead Code in `update_from_observation()` (`MODERATE`, `research_extension/memory_module.py:141–151`):**
   - `known_networks` and `known_data` loops contain only `pass`.

---

## 3. Experimental Protocol Parity with Rigaki et al. (ESWA 2026)

Comparing `research_extension/benchmark_runner.py`, `research_extension/base_agent.py`, and `scenarios/*.yaml` against Rigaki et al. (ESWA 2026, Sections 6.2, 7.1, 7.2; Tables 4 & 5) and NetSecGame's reference configurations (`NetSecGame/NetSecGameAgents/agents/attackers/llm_react/netsecenv-tests_03..06.yaml`, `netsecenv_conf_new_ips_*.yaml`):

### 3.1 Protocol Comparison Matrix

| Protocol Dimension | Rigaki et al. (ESWA 2026) / NetSecGame Reference | Current Codebase & `scenarios/*.yaml` | Audit Status |
| :--- | :--- | :--- | :---: |
| **3(a) IP / Subnet Randomization** | `use_dynamic_addresses: True` (`netsecenv-tests_03..06.yaml`), topology randomized across episodes (`random_agent.py:62,117`) | `use_dynamic_addresses: False` in both `scenarios/*.yaml:43`; `reset_environment()` (`base_agent.py:287`) never passes `randomize_topology=True` | **CRITICAL** |
| **3(a) Goal & Start-Host Randomization** | `known_data: {C2: [random]}`, `controlled_hosts: [C2, random]`, `random_seed: 'random'` (`netsecenv-tests_06.yaml:32,75`, `netsecenv_conf_new_ips_*.yaml:37`) | Static `known_data` (`[[User1, DataFromServer1]]` / `[[admin, passwords]]`), static start hosts (`192.168.1.1` / `192.168.2.2`), fixed `seed=42` (`benchmark_runner.py:49`) | **CRITICAL** |
| **3(b) Max Steps per Episode** | `max_steps: 100` (Tables 4 & 5; `netsecenv-tests_03..06.yaml:62,79,80`; `notebooks/paper_return_win_rate_full.ipynb:34`) | `max_steps: 50` in `scenarios/*.yaml:5`, `example_task_configuration.yaml:10`, and `benchmark_runner.py:100,230` | **CRITICAL** |
| **3(b) Episode Count & Confidence Intervals** | `150` evaluation episodes per condition with `95%` confidence intervals | `episodes: 5` default (`benchmark_runner.py:84,208,347`); `metrics_logger.py:179–215` computes point means only (no SD/SE/95% CI) | **CRITICAL** |
| **3(c) `RandomAttacker` vs. SLM Parity** | Baselines evaluated on identical scenarios, defender settings, and step budgets | `RandomAttacker` only ran on `two_networks_tiny` (`use_global_defender: False`); **never run** on `scenario_full_defender.yaml` or `scenario_three_nets_defender.yaml` | **CRITICAL** |
| **3(c) `GlobalDefender` Thresholds** | `StochasticWithThreshold` (`tw_size=5`, ratios `0.25/0.30/0.25/0.50/0.25`, consecutive `2/3/2`, episode repeats `2/2`) | `use_global_defender: True` invokes `NetSecGame/netsecgame/game/global_defender.py:9–103` with identical thresholds and probabilities | **PASS** *(for SLM runs only; Random missing)* |

### 3.2 Detailed Parity Gaps

- **3(a) Static Seeds & Zero Randomization:**
  - `scenarios/scenario_full_defender.yaml` (lines 12, 17, 43) and `scenarios/scenario_three_nets_defender.yaml` (lines 12, 17, 43) fix every entity: C2 IP, start foothold IP, target data `(owner, id)`, and `use_dynamic_addresses: False`.
  - Even in `example_task_configuration.yaml` (line 49, where `use_dynamic_addresses: True`), `BaseModularAgent.reset_environment()` (`base_agent.py:287`) calls `self.network_agent.request_game_reset()` with default `randomize_topology=False, seed=None` (`NetSecGame/netsecgame/agents/base_agent.py:172–192`). As verified in `NetSecGame.py` lines 1399–1401, `_dynamic_ip_change()` only executes when `topology_change` is `True`. Consequently, **zero IP randomization occurred in any benchmark run**.
  - Furthermore, `BenchmarkHarness` passes static `seed=42` (`benchmark_runner.py:35,49,65`) to every server instance, making all 5 episodes in `FullScenario` and `ThreeNets` replay identical initial states and identical `GlobalDefender` PRNG streams.

- **3(b) Halved Horizon (`50` vs. `100`) & Underpowered Sample (`5` vs. `150`):**
  - Setting `max_steps: 50` cuts the step budget in half relative to Rigaki et al. Tables 4 & 5 (`100` steps), systematically depressing `RandomAttacker` and unmasked SLM win rates.
  - With `N=5` episodes, binomial uncertainty is vast (e.g., `2/5 = 40%` has a 95% Wilson CI of `[11.8%, 76.9%]`), and because `seed=42` is fixed across runs, the 5 episodes are not independent trials. `metrics_logger.py` exports no standard deviations, standard errors, or 95% confidence intervals.
  - Note also that `Condition_1`/`Condition_2` (`11:34–11:46`), `Config_A/B/C` (`10:23–10:29`), and `Adv_*` (`12:47–12:49`) were generated across different versions of `research_extension/*.py`, and `EXPERIMENT_RESULTS.md` line 376 claims `"Up to 80.0% on random seeds"` without any corresponding log file in `logs/benchmarks/`.

- **3(c) Baseline Asymmetry (`RandomAttacker` vs. `ModernSLMAttacker`):**
  - In `benchmark_runner.py`, `run_baseline_random_agent()` is only invoked in `tri_ablation` (lines 582–591, using `example_task_configuration.yaml` with `two_networks_tiny` and `use_global_defender: False`) and `baseline`/`both` (line 648).
  - The `adversarial_suite` mode (lines 491–560) runs `ModernSLMAttacker` on `scenario_full_defender.yaml` and `scenario_three_nets_defender.yaml` **without running `RandomAttacker`** under the same `GlobalDefender`, topologies, and step limits.

---

## 4. Refactoring Specification (Eliminating Hardcoding While Preserving Valid Action Masking)

To achieve publishable ESWA methodology without heuristic hijacking or scenario leakage, apply the following specification before running any full-scale experiments:

### 4.1 Separate Legitimate Action Masking from Policy Selection
1. **Remove `get_tactical_phase_and_guidance()` completely** from both prompt construction (`format_context_prompt()`) and fallback (`prune_and_recover_action()`). Do **not** inject `"TACTICAL DIRECTIVE"`, `"Schema: {...}"`, or phase-priority hints into the SLM prompt.
2. **Implement General-Purpose Action Masking (Precondition + Anti-Loop Filter Only):**
   - Build a domain-agnostic `get_unmasked_valid_actions(state, semantic_memory)` helper derived purely from `netsecgame.generate_valid_actions(state)` (or `state.known_networks`, `state.known_hosts`, `state.controlled_hosts`, `state.known_services`, `state.known_data`) filtered to exclude actions in `semantic_memory.executed_actions` (and optionally failed `(host, service)` exploits).
   - **Critical Constraint:** The mask must expose **all** remaining valid, non-redundant actions (or action-type categories / valid parameter options) **without** scenario-specific ranking (no IP priority, no service priority, no `192.168.1.2` pivot skipping).
   - **Do NOT silently substitute `candidates[0]` on invalid/redundant SLM output.** Instead:
     - *Option A (Preferred for Pure SLM Evaluation):* Reject invalid/redundant output, append a generic feedback message (e.g., `"Invalid parameters"` or `"Already executed"`) to `EpisodicMemoryBuffer`, optionally allow at most `k=1..2` re-prompts **within the same step budget** (or count as a wasted/invalid step matching `llm_agent_react.py`), and log the raw SLM failure.
     - *Option B (If Testing Constrained Decoding / Action Masking):* Present the **entire unmasked valid action set** (shuffled or sorted lexicographically, never by IP/service heuristic) to the SLM, or use logits/grammar-constrained decoding over the full valid set, so the SLM—not Python—chooses among valid targets, networks, services, and data items.

### 4.2 Full Telemetry & Provenance Logging (`metrics_logger.py`, `base_agent.py`, `benchmark_runner.py`)
Add mandatory per-step fields to `StepMetric` and CSV/JSON exports:
- `raw_slm_action_type: str` and `raw_slm_parameters_json: str`
- `raw_slm_parse_ok: bool`
- `raw_slm_valid_in_state: bool`
- `raw_slm_was_redundant: bool`
- `override_triggered: bool` (or `retry_count: int`)
- Compute and export **Override/Fallback Rate (%)**, **Raw SLM Validity Rate (%)**, and **Raw SLM Redundancy Rate (%)** in `MetricsLogger.get_summary()`.

### 4.3 Eliminate All Hardcoding & Fix Memory Bugs
1. **Goal & C2 Discovery:** Remove `goal_description` overrides at `benchmark_runner.py:504,506,535`; always pass `obs.info["goal_description"]` to the SLM. Never hardcode `213.47.23.195`, `DataFromServer1`, `passwords`, or service lists anywhere in `research_extension/`.
2. **Fix `SemanticMemoryStore`:**
   - Delete the duplicate `reset_episode()` at `memory_module.py:502–510` so `failed_exploits` and episode caches clear cleanly at lines 119–129.
   - Normalize dictionary keys (`str(ip)` or `IP(h)`) when checking `known_services` and `known_data`.
   - Fix `DEFAULT_SYSTEM_PROMPT` (`base_agent.py:324`) from `"defensive"` to `"offensive"` / `"penetration testing"` (matching `Attacker` role) and remove prescriptive step-by-step hints if evaluating emergent planning.
3. **Fix Ablation Configs:** Explicitly pass `enable_action_masking=False` in `Config_B_RawSLM` (`benchmark_runner.py:599–604`) and verify ablation flags via unit tests.

### 4.4 Align Experimental Protocol with Rigaki et al. (ESWA 2026)
1. **Scenario YAMLs (`scenarios/*.yaml`):**
   - Set `max_steps: 100`.
   - Set `use_dynamic_addresses: True`.
   - Randomize goals and start positions per episode (`known_data: {213.47.23.195: [random]}`, `controlled_hosts: [213.47.23.195, random]`, matching `NetSecGameAgents` test configs).
2. **Episode Reset & Seeding (`base_agent.py`, `benchmark_runner.py`):**
   - In `reset_environment()`, accept `seed` and `randomize_topology` and pass `request_game_reset(randomize_topology=True, seed=ep_seed)` (e.g., distinct deterministic seed schedule `seed = base_seed + ep` across conditions so all agents face the **same** sequence of randomized topologies/goals without repeating Episode 1 identically).
   - Run **N = 150 episodes** per condition.
3. **Statistical Reporting & Baseline Parity (`metrics_logger.py`, `benchmark_runner.py`):**
   - Add `RandomAttacker` (`run_baseline_random_agent`) to `adversarial_suite`, `full_defender`, and `three_nets_defender` under identical YAML configs, seeds, defender settings, and `max_steps=100`.
   - Compute and export **95% confidence intervals** (Wilson score CIs for Win Rate and Detection Rate; t-distribution / bootstrap CIs for Average Return and Average Steps) in `MetricsLogger.get_summary()`.
