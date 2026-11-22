# Empirical Benchmark Report: Stealth-Aware Topological Belief Graph & Multi-Scale Minimax Hazard Utility in NetSecGame

**Research Study**: Domain-Agnostic Stealth-Aware Topological Belief Graph Planning with Local Small Language Models  
**Reference Benchmark**: Rigaki et al., *"Building Adaptative and Transparent Cyber Agents with Local Language Models"*, Elsevier *Expert Systems with Applications* (ESWA 2026, Tables 4 & 5)  
**Evaluation Date**: November 2026  
**Hardware Accelerator**: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, Blackwell Architecture, CUDA 13.2)  
**Branch**: `feature/stealth-belief-graph-slm`

---

## 1. Executive Summary & Protocol Parity

This benchmark evaluates a **100% domain-agnostic** Sequential Planner combining:
1. **Topological Belief Graph (`research_extension/memory_module.py`)**: Compiles live observations into epistemic node/edge states (`UNEXPLORED`, `MAPPED`, `COMPROMISED`, `EXHAUSTED`) and prunes redundant `(ActionType, Parameters)` transitions.
2. **Multi-Scale Minimax Hazard Utility $U(a)$**: Ranks permissible actions without prior knowledge of the monitor's exact sliding window $W^*$ via:
   $$U(a) = \alpha \cdot \mathrm{ProgressGain}(a \mid \mathcal{G}_t) - \beta \cdot \max_{w \in \{3, 5, 8\}} R_w(a \mid \mathcal{H}_t)$$
   incorporating kill-chain depth-first locality, consecutive-streak breaking, and an `ExfiltrateData` spacing guard.
3. **1-Step Counterfactual Reflexion (`research_extension/agent_wrapper.py`)**: Grants the SLM one corrective turn upon invalid or redundant first-pass JSON before falling back to $\arg\max_a U(a)$.
4. **Fast Local Inference (`research_extension/slm_backend.py`)**: Balanced JSON-brace early stopping (`FirstJsonObjectStoppingCriteria`), static system-prompt prefix `DynamicCache` KV-caching, greedy decoding (`temperature=0.0`), and `max_new_tokens=160` (preventing nested-JSON truncation while halting immediately on the first balanced closing brace).

### Experimental Protocol (ESWA 2026 Parity)
- **Randomization**: `use_dynamic_addresses: True` and `randomize_topology=True` on every episode with fresh unseen IPs, subnets, and target objectives (`seeds 42..191` paired across all 150-episode agents; `seeds 42..2041` for 2,000-episode `RandomAttacker`).
- **Horizon**: `max_steps = 100` across all 4 scenario YAML configurations (`scenario_full_nodefender.yaml`, `scenario_full_defender.yaml`, `scenario_three_nets_nodefender.yaml`, `scenario_three_nets_defender.yaml`).
- **Scientific Integrity**: Zero hardcoded IP strings, subnet strings, service names, or scenario names anywhere in `research_extension/` (verified by `test_zero_hardcoding_ast_scan` in `tests/test_research_extension.py`, 15/15 unit tests passing).

---

## 2. Full 150-Episode / 2,000-Episode Benchmark Tables

> *Note on `FirstPass %`:* For unmasked `RawReAct` (`use_memory=False`, `enable_action_masking=False`) and non-LLM baselines, `FirstPass %` reflects state-legal JSON syntax without redundancy filtering or reflexion.

### 2.1 Regime 1A: Full Topology — Without Defender (`scenario_full_nodefender.yaml`)

| Configuration | Episodes | Win % (±CI95) | Detect % (±CI95) | Return (±CI95) | Steps (±CI95) | FirstPass % | PostReflexion % | Override % | Entropy $H_t$ (bits) | Peak VRAM (MB) | Latency (ms/step) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_RandomAttacker` | 2,000 | 4.65 ± 0.92 | 0.00 ± 0.00 | -103.55 ± 1.30 | 98.66 ± 0.33 | 100.00% | 100.00% | 0.00% | 1.300 | — | — |
| `Baseline_PureHeuristic` | 150 | 96.00 ± 3.15 | 0.00 ± 0.00 | +56.20 ± 6.95 | 39.40 ± 4.76 | 100.00% | 100.00% | 0.00% | 0.657 | — | — |
| `Ablation_PureHeuristic_WithStealth` | 150 | 100.00 ± 0.00 | 0.00 ± 0.00 | +74.99 ± 2.40 | 25.01 ± 2.40 | 100.00% | 100.00% | 0.00% | 1.531 | — | — |
| `Ablation_SLM_7B_NF4_RawReAct` | 150 | 0.00 ± 0.00 | 0.00 ± 0.00 | -110.00 ± 0.00 | 100.00 ± 0.00 | 100.00% | 100.00% | 0.00% | 0.000 | 5,550.68 | 2,323.85 |
| `Ablation_SLM_7B_NF4_BeliefGraph_NoStealth` | 150 | 66.67 ± 7.57 | 0.00 ± 0.00 | +1.63 ± 13.11 | 61.70 ± 5.50 | 46.13% | 64.05% | 35.95% | 0.792 | 5,979.34 | 2,411.04 |
| `Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth` | 150 | 95.33 ± 3.39 | 0.00 ± 0.00 | +50.87 ± 6.40 | 43.99 ± 3.50 | 39.88% | 65.72% | 34.28% | 1.681 | 3,301.64 | 1,301.12 |
| **`Proposed_SLM_7B_NF4_BeliefGraph_WithStealth`** | **150** | **98.67 ± 1.84** | **0.00 ± 0.00** | **+65.16 ± 4.26** | **33.37 ± 2.99** | **78.13%** | **86.06%** | **13.94%** | **1.756** | **6,065.35** | **1,759.42** |

### 2.2 Regime 1B: Full Topology — With Defender / Monitor On (`scenario_full_defender.yaml`)

| Configuration | Episodes | Win % (±CI95) | Detect % (±CI95) | Return (±CI95) | Steps (±CI95) | FirstPass % | PostReflexion % | Override % | Entropy $H_t$ (bits) | Peak VRAM (MB) | Latency (ms/step) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_RandomAttacker` | 2,000 | 0.25 ± 0.22 | 99.45 ± 0.32 | -32.24 ± 0.76 | 22.52 ± 0.73 | 100.00% | 100.00% | 0.00% | 1.208 | — | — |
| `Baseline_PureHeuristic` | 150 | 48.00 ± 8.02 | 52.00 ± 8.02 | +21.29 ± 8.54 | 21.51 ± 2.84 | 100.00% | 100.00% | 0.00% | 0.828 | — | — |
| `Ablation_PureHeuristic_WithStealth` | 150 | 71.33 ± 7.26 | 28.67 ± 7.26 | +51.05 ± 7.84 | 17.41 ± 1.48 | 100.00% | 100.00% | 0.00% | 1.620 | — | — |
| `Ablation_SLM_7B_NF4_RawReAct` | 150 | 0.00 ± 0.00 | 98.67 ± 1.84 | -35.53 ± 3.33 | 25.53 ± 3.33 | 100.00% | 100.00% | 0.00% | 0.000 | 5,542.57 | 1,021.48 |
| `Ablation_SLM_7B_NF4_BeliefGraph_NoStealth` | 150 | 26.67 ± 7.10 | 72.00 ± 7.21 | -6.08 ± 8.10 | 25.41 ± 3.10 | 50.84% | 73.77% | 26.23% | 1.006 | 5,973.57 | 2,088.69 |
| `Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth` | 150 | 48.67 ± 8.03 | 51.33 ± 8.03 | +16.01 ± 7.97 | 27.53 ± 2.81 | 36.26% | 62.15% | 37.85% | 1.654 | 3,300.38 | 1,269.42 |
| **`Proposed_SLM_7B_NF4_BeliefGraph_WithStealth`** | **150** | **63.33 ± 7.74** | **36.67 ± 7.74** | **+36.11 ± 7.71** | **23.55 ± 2.31** | **79.79%** | **87.97%** | **12.03%** | **1.749** | **6,054.33** | **1,669.35** |

### 2.3 Regime 2A: Three-Networks Multi-Hop Topology — Without Defender (`scenario_three_nets_nodefender.yaml`)

| Configuration | Episodes | Win % (±CI95) | Detect % (±CI95) | Return (±CI95) | Steps (±CI95) | FirstPass % | PostReflexion % | Override % | Entropy $H_t$ (bits) | Peak VRAM (MB) | Latency (ms/step) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_RandomAttacker` | 2,000 | 3.30 ± 0.78 | 0.00 ± 0.00 | -105.50 ± 1.08 | 99.13 ± 0.26 | 100.00% | 100.00% | 0.00% | 1.303 | — | — |
| `Baseline_PureHeuristic` | 150 | 70.00 ± 7.36 | 0.00 ± 0.00 | +17.93 ± 13.62 | 49.07 ± 5.77 | 100.00% | 100.00% | 0.00% | 0.585 | — | — |
| `Ablation_PureHeuristic_WithStealth` | 150 | 93.33 ± 4.01 | 0.00 ± 0.00 | +61.11 ± 7.64 | 31.55 ± 3.61 | 100.00% | 100.00% | 0.00% | 1.595 | — | — |
| `Ablation_SLM_7B_NF4_RawReAct` | 150 | 0.00 ± 0.00 | 0.00 ± 0.00 | -110.00 ± 0.00 | 100.00 ± 0.00 | 100.00% | 100.00% | 0.00% | 0.000 | 5,542.57 | 1,021.77 |
| `Ablation_SLM_7B_NF4_BeliefGraph_NoStealth` | 150 | 60.00 ± 7.87 | 0.00 ± 0.00 | -11.83 ± 13.34 | 67.83 ± 5.48 | 45.53% | 63.56% | 36.44% | 0.767 | 5,983.20 | 2,421.38 |
| `Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth` | 150 | 86.67 ± 5.46 | 0.00 ± 0.00 | +31.30 ± 9.35 | 54.03 ± 4.07 | 37.91% | 63.23% | 36.77% | 1.655 | 3,300.47 | 1,368.86 |
| **`Proposed_SLM_7B_NF4_BeliefGraph_WithStealth`** | **150** | **94.67 ± 3.61** | **0.00 ± 0.00** | **+52.05 ± 6.83** | **42.09 ± 3.65** | **77.40%** | **85.76%** | **14.24%** | **1.764** | **6,067.20** | **1,836.86** |

### 2.4 Regime 2B: Three-Networks Multi-Hop Topology — With Defender / Monitor On (`scenario_three_nets_defender.yaml`)

| Configuration | Episodes | Win % (±CI95) | Detect % (±CI95) | Return (±CI95) | Steps (±CI95) | FirstPass % | PostReflexion % | Override % | Entropy $H_t$ (bits) | Peak VRAM (MB) | Latency (ms/step) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_RandomAttacker` | 2,000 | 0.30 ± 0.24 | 99.30 ± 0.37 | -32.29 ± 0.76 | 22.62 ± 0.73 | 100.00% | 100.00% | 0.00% | 1.209 | — | — |
| `Baseline_PureHeuristic` | 150 | 38.00 ± 7.79 | 59.33 ± 7.89 | +7.64 ± 9.59 | 24.16 ± 3.41 | 100.00% | 100.00% | 0.00% | 0.775 | — | — |
| `Ablation_PureHeuristic_WithStealth` | 150 | 65.33 ± 7.64 | 34.00 ± 7.61 | +42.35 ± 8.33 | 19.52 ± 1.96 | 100.00% | 100.00% | 0.00% | 1.687 | — | — |
| `Ablation_SLM_7B_NF4_RawReAct` | 150 | 0.00 ± 0.00 | 98.67 ± 1.84 | -32.71 ± 3.06 | 22.71 ± 3.06 | 100.00% | 100.00% | 0.00% | 0.000 | 5,542.57 | 1,013.37 |
| `Ablation_SLM_7B_NF4_BeliefGraph_NoStealth` | 150 | 19.33 ± 6.34 | 78.67 ± 6.58 | -15.77 ± 7.63 | 27.03 ± 3.31 | 50.04% | 72.40% | 27.60% | 0.968 | 5,951.59 | 2,146.18 |
| `Proposed_SLM_1.5B_BF16_BeliefGraph_WithStealth` | 150 | 42.67 ± 7.94 | 57.33 ± 7.94 | +8.48 ± 8.12 | 28.45 ± 3.28 | 35.97% | 62.35% | 37.65% | 1.634 | 3,300.39 | 1,293.91 |
| **`Proposed_SLM_7B_NF4_BeliefGraph_WithStealth`** | **150** | **59.33 ± 7.89** | **40.00 ± 7.87** | **+28.01 ± 8.02** | **27.25 ± 2.76** | **80.92%** | **89.63%** | **10.37%** | **1.774** | **6,048.96** | **1,678.30** |

---

## 3. Paired Statistical Significance Tests ($n = 150$, Seeds `42..191`)

Exact two-sided **McNemar's tests** (win rate), **Wilcoxon signed-rank tests**, and **Cliff's $\delta$** effect sizes comparing `Proposed_SLM_7B_NF4_BeliefGraph_WithStealth` against paired baselines and ablations:

| Regime | Comparator | McNemar $p$ (Win) | Discordant $(b, c)$ | Wilcoxon $p$ (Return) | Cliff's $\delta$ (Return) | Wilcoxon $p$ (Steps) | Cliff's $\delta$ (Steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime 1A (Full, NoDef)** | vs. `BeliefGraph_NoStealth` | **`< 1e-6`** | `(50, 2)` | **`< 1e-6`** | **`+0.4260`** | **`< 1e-6`** | **`-0.4260`** |
| **Regime 1A (Full, NoDef)** | vs. `PureHeuristic` | `0.289062` | `(6, 2)` | `0.132773` | `-0.0186` | `0.145167` | `+0.0186` |
| **Regime 1A (Full, NoDef)** | vs. `PureHeuristic_WithStealth` | `0.500000` | `(0, 2)` | `0.000001` | `-0.3258` | `0.000001` | `+0.3258` |
| **Regime 1A (Full, NoDef)** | vs. `RawReAct` | **`< 1e-6`** | `(148, 0)` | **`< 1e-6`** | **`+0.9867`** | **`< 1e-6`** | **`-0.9867`** |
| **Regime 1A (Full, NoDef)** | vs. `Proposed_1.5B_BF16` | `0.179688` | `(7, 2)` | `< 1e-6` | `+0.3288` | `< 1e-6` | `-0.3289` |
| **Regime 1B (Full, Monitor On)** | vs. `BeliefGraph_NoStealth` | **`< 1e-6`** | `(60, 5)` | **`< 1e-6`** | **`+0.4390`** | `0.806417` | `+0.0093` |
| **Regime 1B (Full, Monitor On)** | vs. `PureHeuristic` | `0.001089` | `(35, 12)` | `0.148859` | `+0.0712` | `0.015983` | `+0.1677` |
| **Regime 1B (Full, Monitor On)** | vs. `PureHeuristic_WithStealth` | `0.042774` | `(9, 21)` | `< 1e-6` | `-0.3096` | `< 1e-6` | `+0.2487` |
| **Regime 1B (Full, Monitor On)** | vs. `RawReAct` | **`< 1e-6`** | `(95, 0)` | **`< 1e-6`** | **`+0.7492`** | `0.903191` | `+0.0439` |
| **Regime 1B (Full, Monitor On)** | vs. `Proposed_1.5B_BF16` | `0.000941` | `(32, 10)` | `0.000001` | `+0.2725` | `0.006517` | `-0.1102` |
| **Regime 2A (ThreeNets, NoDef)** | vs. `BeliefGraph_NoStealth` | **`< 1e-6`** | `(56, 4)` | **`< 1e-6`** | **`+0.3705`** | **`< 1e-6`** | **`-0.3682`** |
| **Regime 2A (ThreeNets, NoDef)** | vs. `PureHeuristic` | **`< 1e-6`** | `(44, 7)` | `0.284268` | `-0.0384` | `0.511419` | `+0.0404` |
| **Regime 2A (ThreeNets, NoDef)** | vs. `PureHeuristic_WithStealth` | `0.803619` | `(9, 7)` | `0.000005` | `-0.3042` | `0.000003` | `+0.3046` |
| **Regime 2A (ThreeNets, NoDef)** | vs. `RawReAct` | **`< 1e-6`** | `(142, 0)` | **`< 1e-6`** | **`+0.9467`** | **`< 1e-6`** | **`-0.9400`** |
| **Regime 2A (ThreeNets, NoDef)** | vs. `Proposed_1.5B_BF16` | `0.028959` | `(19, 7)` | `0.000119` | `+0.3015` | `0.000135` | `-0.3006` |
| **Regime 2B (ThreeNets, Monitor On)** | vs. `BeliefGraph_NoStealth` | **`< 1e-6`** | `(64, 4)` | **`< 1e-6`** | **`+0.4549`** | `0.259771` | `+0.0600` |
| **Regime 2B (ThreeNets, Monitor On)** | vs. `PureHeuristic` | `0.000030` | `(45, 13)` | `0.080383` | `+0.1105` | `0.010027` | `+0.1884` |
| **Regime 2B (ThreeNets, Monitor On)** | vs. `PureHeuristic_WithStealth` | `0.175465` | `(13, 22)` | `0.000001` | `-0.2593` | `< 1e-6` | `+0.2736` |
| **Regime 2B (ThreeNets, Monitor On)** | vs. `RawReAct` | **`< 1e-6`** | `(89, 0)` | **`< 1e-6`** | **`+0.6526`** | `0.000017` | `+0.2262` |
| **Regime 2B (ThreeNets, Monitor On)** | vs. `Proposed_1.5B_BF16` | `0.000170` | `(34, 9)` | `0.000027` | `+0.2193` | `0.325450` | `+0.0077` |

---

## 4. Comparison with Rigaki et al. (ESWA 2026, Tables 4 & 5)

Exact published baselines and 95% confidence intervals extracted directly from Tables 4 & 5 of Rigaki, Catania, & García (*Expert Systems With Applications*, Vol. 299, 2026, Article `129987`, `DOI: 10.1016/j.eswa.2025.129987`) compared against our 150-episode / 2,000-episode randomized evaluations (`max_steps = 100`). For our runs, returns are shown as `(Native / ESWA-50)`, where **ESWA-50** calibrates rewards to the ESWA 2026 penalty scale (`-50` detection penalty and `0` timeout penalty beyond `-1/step`, vs. `-10`/`-10` in local YAML).

| Agent / Method | Peak VRAM / Compute | Full (NoDef) Win % (Return) | Full (Monitor On) Win % (Return) | Three-Nets (NoDef) Win % (Return) | Three-Nets (Monitor On) Win % (Return) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline (ESWA 2026 Ref) | CPU | 31.16 ± 2.03 (-57.49) | 2.72 ± 0.71 (-66.56) | 4.86 ± 0.94 (-93.28) | 0.13 ± 0.16 (-70.73) |
| Random Baseline (Ours, 2,000 eps) | CPU | 4.65 ± 0.92 (-103.55 / -94.01) | 0.25 ± 0.22 (-32.24 / -71.99) | 3.30 ± 0.78 (-105.50 / -95.83) | 0.30 ± 0.24 (-32.29 / -71.97) |
| Q-Learning (ESWA 2026 Ref, fixed IPs) | CPU | 67.71 ± 4.09 (25.49) | 71.57 ± 3.95 (42.94) | 0.00 ± 0.00 (-100.00) | 0.00 ± 0.00 (-70.45) |
| GPT-3.5-turbo (ESWA 2026 Ref) | Cloud API | 30.00 ± 16.40 (-51.67) | 16.67 ± 13.33 (-58.00) | 10.00 ± 10.73 (-87.50) | 6.67 ± 8.93 (-63.63) |
| Untuned Zephyr-7B-$\beta$ (ESWA 2026 Ref) | ~14 GB FP16 | 23.65 ± 4.80 (-51.77) | 3.33 ± 2.03 (-66.68) | 1.75 ± 1.97 (-98.80) | 0.62 ± 1.21 (-70.42) |
| Fine-Tuned Hackphyr-7B (ESWA 2026 Ref) | ~14 GB FP16 | 89.10 ± 4.74 (61.03) | 44.00 ± 6.88 (3.56) | 50.34 ± 8.00 (-14.26) | 23.33 ± 6.77 (-37.67) |
| GPT-4 (ESWA 2026 Ref) | Cloud API | 100.00 ± 0.00 (77.13) | 53.33 ± 17.85 (8.80) | 82.35 ± 13.64 (18.78) | 36.36 ± 17.21 (-21.69) |
| **Proposed 1.5B-BF16 (Zero-Shot, Ours)** | **3.22 GB** | **95.33 ± 3.39 (+50.87 / +51.34)** | **48.67 ± 8.03 (+16.01 / -4.53)** | **86.67 ± 5.46 (+31.30 / +32.63)** | **42.67 ± 7.94 (+8.48 / -14.45)** |
| **Proposed 7B-NF4 (Zero-Shot, Ours)** | **5.91–5.92 GB** | **98.67 ± 1.84 (+65.16 / +65.29)** | **63.33 ± 7.74 (+36.11 / +21.45)** | **94.67 ± 3.61 (+52.05 / +52.58)** | **59.33 ± 7.89 (+28.01 / +12.08)** |

### Unpaired Two-Proportion $z$-Tests vs. External Baselines
- **`Proposed 7B-NF4` vs. `Hackphyr-7B`**: Regime 1A ($z=+3.48, p=4.98\times 10^{-4}$), Regime 1B ($z=+3.58, p=3.40\times 10^{-4}$), Regime 2A ($z=+8.60, p=8.08\times 10^{-18}$), Regime 2B ($z=+6.33, p=2.43\times 10^{-10}$).
- **`Proposed 7B-NF4` vs. `GPT-4`**: Regime 1A ($z=-0.64, p=0.525$), Regime 1B ($z=+1.03, p=0.304$), Regime 2A ($z=+2.36, p=0.0185$), Regime 2B ($z=+2.31, p=0.0208$).
- **`Proposed 1.5B-BF16` vs. `Hackphyr-7B`**: Regime 1A ($z=+2.05, p=0.0408$), Regime 1B ($z=+0.87, p=0.386$), Regime 2A ($z=+6.77, p=1.26\times 10^{-11}$), Regime 2B ($z=+3.56, p=3.68\times 10^{-4}$).

**Deterministic Scaffold vs. SLM Reasoning Separation (`Ablation_PureHeuristic_WithStealth`):** Running pure deterministic $\arg\max_{a \in \mathcal{A}_t^{\text{valid}}} U(a)$ (`100%` heuristic, zero LLM calls) achieves `100.00%` (1A), `71.33%` (1B), `93.33%` (2A), and `65.33%` (2B) win rates—confirming that the Topological Belief Graph + Multi-Scale Minimax Utility scaffold drives the majority of stealth-evasion robustness (`+23.33 pp` in 1B and `+27.33 pp` in 2B over non-stealth `PureHeuristic`). `Proposed_SLM_7B_NF4` maintains statistical win-rate parity with deterministic $\arg\max U(a)$ in 1A (`98.67%` vs. `100.00%`, McNemar $p=0.500$), 2A (`94.67%` vs. `93.33%`, $p=0.804$), and 2B (`59.33%` vs. `65.33%`, $p=0.175$), with an `8.00 pp` gap in 1B (`63.33%` vs. `71.33%`, $p=0.0428$), while maintaining `85.76%–89.63%` genuine SLM action selection (`PostReflexion %`) and higher Shannon diversity (`1.749–1.774` bits vs. `1.531–1.687` bits). Conversely, the `1.5B-BF16` SLM trails deterministic $\arg\max U(a)$ under monitoring (`48.67%` and `42.67%`), quantifying the reasoning-capacity threshold below `7B`.

---

## 5. Publication Figures (300 DPI)

1. **System Architecture**: ![System Architecture](figures/fig1_system_architecture.png)
2. **Belief Graph State Machine & Minimax Utility**: ![Mechanism](figures/fig2_belief_and_hazard_mechanism.png)
3. **Win Rate vs. Monitor Alert Rate Ablation & Autonomy Breakdown**: ![Ablation and Autonomy](figures/fig3_ablation_and_autonomy.png)
4. **Markov Action Transition Dynamics (`Three-Networks + Monitor On`)**: ![Markov Transitions](figures/fig4_markov_transitions_comparison.png)
5. **Shannon Action Entropy Trajectory & Hardware-Aware Pareto Frontier**: ![Entropy and Pareto](figures/fig5_entropy_and_pareto.png)
