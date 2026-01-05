# Cognitive-Core-Engine-Test

Multi-module architecture integrating fixed cognitive core with invention and governance layers.

## Architecture

Three-module system with strict separation of concerns:

1. **NON_RSI_AGI_CORE_v5.py** - Fixed orchestrator (main loop owner)
   - Hyperdimensional Computing (HDC) with 10,000-bit vectors
   - Majority-rule bundling for neuro-symbolic memory
   - World model: feature-based Q-value estimation with experience replay
   - Planner: beam search over world model (depth=3, width=6)
   - Skill DSL: data-level interpreted programs (call/if/foreach)
   - Multi-agent orchestrator with project graph

2. **omega_forge_two_stage_feedback.py** - Invention plugin (invoked on stagnation)
   - Structural-transition discovery via CFG analysis
   - Virtual machine: 8 registers, 64 memory cells, 21 opcodes
   - Detector: multi-stage control flow novelty (CFG edit distance, SCC analysis)
   - Curriculum: warmup phase with relaxed constraints
   - Crash-safe JSONL evidence logging

3. **unified_rsi_extended.py** - Governance/evaluation gate (critic-only adoption)
   - Pre-filtering and stress-checking of candidates
   - Expandable grammar for invention representation
   - Blackboard JSONL logging for multi-loop coordination

## Integration

Call chain: `Orchestrator -> Omega (on stagnation) -> Unified (critic) -> Orchestrator (register/reject)`

- **Contract A (GapSpec)**: Orchestrator -> Omega capability gap specification
- **Contract B (CandidatePacket)**: Omega -> Unified -> Orchestrator artifact + evidence + verdict

No file merging. No self-adoption by invention module.

## Technical Details

**HDC Memory**: Associative retrieval using bundled hypervectors with:
- Token encoding cache (~8000 items)
- Similarity threshold: 0.48 (random baseline: 0.5)
- Recency weighting, reward boosting

**Structural Discovery**: CFG-based novelty detection with:
- Edit distance K (warmup: 3, strict: 6)
- Active subsequence length L (warmup: 8, strict: 10)
- Minimum coverage: 0.55
- Reproducibility: N=4 trials, max CFG variants: 2

**World Model**: TD-learning with:
- Non-linear feature combinations
- Experience replay buffer (200 samples)
- Gamma: 0.9, LR: 0.08

## Usage

```bash
# Run fixed core
python NON_RSI_AGI_CORE_v5.py --rounds 40 --agents 8

# Run invention engine
python omega_forge_two_stage_feedback.py evidence_run --target 6 --max_generations 2000

# Analyze results
python analyze_results.py
```

## Status

Research/engineering hybrid. Governance-gated architecture with rollback.
