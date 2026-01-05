# Results

This document captures a short, reproducible evidence run that demonstrates:
1) Stagnation/plateau detection in the core under a normal run.
2) Omega Forge discovering multiple distinct structural transitions (CFG novelty).
3) Unified critic approval/rejection behavior under guardrails.
4) End-to-end wiring: Orchestrator → Omega (on stagnation) → Critic → Orchestrator (register/reject), with logs.

## Reproduction Commands

> All commands are deterministic with fixed seeds and produce the logs in `logs/`.

### A) Baseline core run (stagnation evidence)
```bash
python - <<'PY'
import json
import random
from pathlib import Path
import NON_RSI_AGI_CORE_v5 as core

seed = 11
rounds = 30
log_path = Path('logs/core_baseline.txt')

random.seed(seed)
env = core.ResearchEnvironment(seed=seed)
tools = core.ToolRegistry()
orch_cfg = core.OrchestratorConfig(
    agents=6,
    base_budget=20,
    selection_top_k=3,
)
orch = core.Orchestrator(orch_cfg, env, tools)

tools.register("write_note", core.tool_write_note_factory(orch.mem))
tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
tools.register("evaluate_candidate", core.tool_evaluate_candidate)
tools.register("tool_build_report", core.tool_tool_build_report)

lines = []
lines.append(f"seed={seed} rounds={rounds}")
for r in range(rounds):
    out = orch.run_round(r)
    orch._record_round_rewards(out["results"])
    mean_reward = orch._recent_rewards[-1]
    stagnation = orch._detect_stagnation(window=5, threshold=0.01)
    top = sorted(out["results"], key=lambda x: x["reward"], reverse=True)[:3]
    lines.append(
        f"[Round {r:02d}] tasks={','.join(out['tasks']):<35} "
        f"mean_reward={mean_reward:.4f} stagnation={stagnation} "
        f"top_rewards={[round(x['reward'],4) for x in top]}"
    )

log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(log_path.read_text(encoding="utf-8"))
PY
```

### B) Omega evidence run (CFG novelty)
> The module includes an `cmd_evidence_run()` helper even though the CLI entrypoint is reserved for the `full` pipeline.
```bash
python - <<'PY'
from types import SimpleNamespace
import omega_forge_two_stage_feedback as omega

args = SimpleNamespace(out='logs/omega_evidence.jsonl', target=6, max_generations=2000, seed=42, report_every=50)
omega.cmd_evidence_run(args)
PY
```

### C) Critic evaluation (approve/reject under guardrails)
```bash
python - <<'PY'
import json
import random
from pathlib import Path
import NON_RSI_AGI_CORE_v5 as core
import omega_forge_two_stage_feedback as omega

critic = core.load_unified_critic_module()

log_path = Path('logs/critic_eval.jsonl')
records = []

random.seed(21)
env = core.ResearchEnvironment(seed=21)
tools = core.ToolRegistry()
orch_cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
orch = core.Orchestrator(orch_cfg, env, tools)

engine = omega.Stage1Engine(seed=123)
engine.init_population()
for _ in range(5):
    engine.step()
    if engine.candidates:
        break

omega_candidates = engine.candidates[:1]

for idx, cand in enumerate(omega_candidates):
    packet = {
        "proposal": {
            "proposal_id": f"omega_{idx}",
            "level": "L0",
            "payload": {"candidate": cand, "gap_spec": {"seed": 123}},
            "evidence": {"metrics": cand.get("metrics", {})},
        },
        "evaluation_rules": dict(orch.evaluation_rules),
        "invariants": dict(orch.invariants),
    }
    verdict = critic.critic_evaluate_candidate_packet(packet, invariants=orch.invariants)
    failed = [k for k, ok in verdict.items() if k.endswith('_ok') and ok is False]
    records.append({
        "source": "omega_stage1",
        "proposal_id": packet["proposal"]["proposal_id"],
        "gid": cand.get("gid"),
        "verdict": verdict.get("verdict"),
        "guardrails_ok": verdict.get("guardrails_ok"),
        "failed_checks": failed,
        "verdict_detail": verdict,
    })

reject_candidate = {
    "gid": "missing_holdout",
    "metrics": {"train_pass_rate": 0.33},
}
packet = {
    "proposal": {
        "proposal_id": "reject_missing_holdout",
        "level": "L0",
        "payload": {"candidate": reject_candidate, "gap_spec": {"seed": 123}},
        "evidence": {"metrics": reject_candidate["metrics"]},
    },
    "evaluation_rules": dict(orch.evaluation_rules),
    "invariants": dict(orch.invariants),
}
verdict = critic.critic_evaluate_candidate_packet(packet, invariants=orch.invariants)
failed = [k for k, ok in verdict.items() if k.endswith('_ok') and ok is False]
records.append({
    "source": "synthetic_guardrail",
    "proposal_id": packet["proposal"]["proposal_id"],
    "gid": reject_candidate.get("gid"),
    "verdict": verdict.get("verdict"),
    "guardrails_ok": verdict.get("guardrails_ok"),
    "failed_checks": failed,
    "verdict_detail": verdict,
})

approved_candidate = {
    "gid": "synthetic_approve",
    "metrics": {
        "train_pass_rate": 0.37,
        "holdout_pass_rate": 0.36,
        "adversarial_pass_rate": 0.30,
        "distribution_shift": {"holdout_pass_rate": 0.31},
        "discovery_cost": {"holdout": 0.5},
    },
}
packet = {
    "proposal": {
        "proposal_id": "synthetic_pass",
        "level": "L0",
        "payload": {"candidate": approved_candidate, "gap_spec": {"seed": 123}},
        "evidence": {"metrics": approved_candidate["metrics"]},
    },
    "evaluation_rules": dict(orch.evaluation_rules),
    "invariants": dict(orch.invariants),
}
verdict = critic.critic_evaluate_candidate_packet(packet, invariants=orch.invariants)
failed = [k for k, ok in verdict.items() if k.endswith('_ok') and ok is False]
records.append({
    "source": "synthetic",
    "proposal_id": packet["proposal"]["proposal_id"],
    "gid": approved_candidate.get("gid"),
    "verdict": verdict.get("verdict"),
    "guardrails_ok": verdict.get("guardrails_ok"),
    "failed_checks": failed,
    "verdict_detail": verdict,
})

with log_path.open('w', encoding='utf-8') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(log_path.read_text(encoding='utf-8'))
PY
```

### D) End-to-end integration (Orchestrator → Omega → Critic → Orchestrator)
```bash
python - <<'PY'
import json
import random
from pathlib import Path
import NON_RSI_AGI_CORE_v5 as core

seed = 33
rounds = 2
log_path = Path('logs/blackboard.jsonl')

random.seed(seed)
env = core.ResearchEnvironment(seed=seed)
tools = core.ToolRegistry()
orch_cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
orch = core.Orchestrator(orch_cfg, env, tools)

tools.register("write_note", core.tool_write_note_factory(orch.mem))
tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
tools.register("evaluate_candidate", core.tool_evaluate_candidate)
tools.register("tool_build_report", core.tool_tool_build_report)

with log_path.open('w', encoding='utf-8') as f:
    for r in range(rounds):
        stagnation_override = True if r == 0 else None
        out = orch.run_recursive_cycle(r, stagnation_override=stagnation_override, force_meta_proposal=True)
        f.write(json.dumps({
            "event": "round_complete",
            "round": r,
            "stagnation": out.get("stagnation"),
            "gap_spec": out.get("gap_spec"),
            "critic_results": out.get("critic_results"),
        }, ensure_ascii=False) + "\n")
        for item in out.get("critic_results", []):
            f.write(json.dumps({
                "event": "critic_decision",
                "round": r,
                "proposal_id": item.get("proposal_id"),
                "level": item.get("level"),
                "verdict": item.get("verdict"),
                "adopted": item.get("adopted"),
                "result": "REGISTERED" if item.get("adopted") else "REJECTED",
            }, ensure_ascii=False) + "\n")

print(log_path.read_text(encoding='utf-8'))
PY
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
- **Loop counts:** {2: 3, 3: 1, 4: 1, 5: 1}

Evidence source: `logs/omega_evidence.jsonl`.

### 3) Critic evaluation (guardrails)
- **Verdict counts:** approve=2, reject=1
- **Reject reasons (guardrails):** holdout metrics missing → `holdout_ok=false`, `holdout_cost_ok=false`

Evidence source: `logs/critic_eval.jsonl`.

### 4) End-to-end integration
- **Round 0:** stagnation forced → gap_spec emitted → Omega candidate evaluated → critic rejected (L0)
- **Round 1:** no stagnation → only L1/L2 proposals evaluated (both rejected)

Evidence source: `logs/blackboard.jsonl`.

## What this proves
- The core can hit a measurable stagnation signal under a fixed-seed run.
- Omega Forge produces multiple distinct CFG signatures with crash-safe evidence logging.
- The unified critic enforces guardrails (rejects on missing holdout metrics) and can approve qualifying packets.
- A minimal end-to-end loop from orchestrator to Omega, critic, and back executes with traceable logs.


