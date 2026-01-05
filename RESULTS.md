# Results

This document captures a short, reproducible evidence run that demonstrates:
1) Stagnation/plateau detection in the core under a normal run.
2) Omega Forge discovering multiple distinct structural transitions (CFG novelty).
3) Unified critic approval/rejection behavior under guardrails.
4) End-to-end wiring: Orchestrator → Omega (on stagnation) → Critic → Orchestrator (register/reject), with logs.

## Reproduction Commands

> The single entrypoint below reproduces all evidence runs and overwrites `logs/`.

```bash
python scripts/run_results.py
```

### Optional advanced commands

```bash
# Run the full orchestrator loop (not required for evidence reproduction)
python NON_RSI_AGI_CORE_v5.py --rounds 40 --agents 8 --seed 0

# Run Omega Forge’s full two-stage pipeline
python omega_forge_two_stage_feedback.py full --stage1_gens 200 --stage2_gens 200 --seed 42
```

## Key Metrics

### 1) Stagnation signal (core baseline)
- **Stagnation rounds:** 8/30 rounds flagged by the rolling-window detector.
- **First stagnation:** Round 13.

Evidence source: `logs/core_baseline.txt`.

### 2) Omega evidence (CFG novelty)
- **Evidence lines:** 6
- **Unique CFG canonical hashes:** 6 (distinct structural transitions)
- **SCC counts:** all `scc_n=1`
- **Loop counts:** {1: 1, 2: 3, 4: 2}

Evidence source: `logs/omega_evidence.jsonl`.

### 3) Critic evaluation (guardrails)
- **Verdict counts:** approve=2, reject=1
- **Reject reasons (guardrails):** holdout metrics missing → `holdout_ok=false`, `holdout_cost_ok=false`

Evidence source: `logs/critic_eval.jsonl`.

### 4) End-to-end integration
- **Round 0:** stagnation forced → gap_spec emitted → Omega candidate evaluated → critic rejected (L0)
- **Round 1:** no stagnation → only L1/L2 proposals evaluated (both rejected)

Evidence source: `logs/blackboard.jsonl`.

## CHANGELOG (evidence run hardening)
- Renamed critic module to `unified_rsi_extended.py` and added a legacy loader fallback for the old spaced filename.
- Added `scripts/run_results.py` to reproduce all evidence logs from a single command.

## What this proves
- The core can hit a measurable stagnation signal under a fixed-seed run.
- Omega Forge produces multiple distinct CFG signatures with crash-safe evidence logging.
- The unified critic enforces guardrails (rejects on missing holdout metrics) and can approve qualifying packets.
- A minimal end-to-end loop from orchestrator to Omega, critic, and back executes with traceable logs.

## What this does **not** prove
- It does **not** demonstrate general AGI capability or open-ended autonomous self-improvement.
- It does **not** show large-scale performance gains or long-horizon training.
- It does **not** establish robustness beyond the guardrails and short deterministic runs shown above.
