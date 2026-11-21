# llm-jump

**Problem.** LLMs treat theory as law. Stated formally: they assert `φ` when only `Γ ⊢ φ` is licensed, and they refuse to revise the famous members of `Γ`.

## End goal

A model that has read nothing after 1900 is shown a checked contradiction behind a 1900 to 1930 discovery it has never seen, and it names the assumption history later revised.

Pass, all three on held-out discoveries:

1. The historically correct exit appears in its answer at twice the base model's rate or better.
2. The famous-versus-stripped gap is halved or better.
3. No rise in false revision on controls where the law was right.

Headline case: flight. `{Smeaton coefficient 0.005, lift equation, glider measurements} ⊢ ⊥`. The Wrights' jump was dropping a coefficient held as law since 1759.

## Foundation: every claim is a sequent

`Γ ⊢ φ`: under assumption set `Γ`, statement `φ` follows.

- **Ground truth** is a sequent a checker has accepted. Nothing else. The bare `φ` is never ground truth.
- **A positive assumption** is a member of some `Γ`. Observations, theories, laws and consensus all live here.
- **A law** is an assumption that appears in many `Γ`. It has no other special status.

One rule covers every domain:

| Domain | Sequent |
|---|---|
| Math | `{axioms} ⊢ theorem` |
| Code | `{spec, semantics} ⊢ program correct` |
| Physics | `{inverse-square, mechanics} ⊢ Kepler's laws` |
| Biology | `{raw data, pipeline, statistical model} ⊢ effect size` |
| Observation | `{instrument, record} ⊢ reading = v` |

The checker proves that `Γ` is sufficient. A hidden assumption shows up as a rejected derivation.

## The three failures

1. **Discharge:** output `φ` with `Γ` dropped. This is "theory as law".
2. **Rejection:** refuse a checked sequent because `φ` contradicts consensus.
3. **Protection:** given a checked contradiction `Γ ⊢ ⊥`, refuse to consider removing the famous member of `Γ`.

## The jump

A checked `Γ ⊢ ⊥` forces a revision. The correct response is the full list of minimal subsets whose removal restores consistency, with no member protected by reputation.

Mercury: `{Newtonian gravity, observations, no unseen planet} ⊢ ⊥`. Three exits. History protected the first for 56 years.

## How we do it: the ledger is itself verified

The ledger is a small state machine written in Lean, with these invariants proven:

- **I1 No promotion.** A sequent is marked checked only by a checker verdict.
- **I2 Closure.** A derived claim's `Γ` contains the `Γ` of everything it was derived from.
- **I3 Retraction.** Removing an assumption unmarks every claim that depends on it.
- **I4 No bare output.** The ledger never emits `φ` without its `Γ`.

LLMs and agents write proposals. Only the ledger's transition function changes state. An LLM review is an assumption in `Γ`, named.

The project's own results are ledger entries: `{dataset, model versions, code hash} ⊢ results table`, checked by re-execution.

## Checkers

Plugins behind one interface: sequent and certificate in, accept or reject out. Lean 4 first, then TLA+, SMT and SAT with proof output, deterministic re-execution. Admission: a small kernel, or a certificate a small independent checker replays. Bounded checks are recorded as bounded.

## v0

| Piece | Does |
|---|---|
| `ledger` | Lean state machine, I1 to I4 proven |
| `check` | Dispatches to a checker. Lean and SAT in v0 |
| `bench` | Runs the two tests below, scores from checker verdicts only |

**Test A, rejection.** Checked sequents whose `φ` contradicts consensus, plus broken ones as control.

**Test B, protection.** Checked contradictions with every correction set known by construction. The model names what to give up.

**Paired design.** Every item runs twice: once with famous names, once with names stripped and the logic identical. The score difference is the consensus prior, isolated.

**Harness control.** A scripted always-dogmatic model scores 100%, a scripted correct model scores 0%, before any real model runs.

## Gate

The paired difference is above zero on A or B. If it is zero, the problem as stated does not exist in current models and the project stops.

## Experiment: the 1900 model

Time is the train/test split. The model cannot have memorized what had not happened.

**Models.**

| Model | Size | Cutoff | Role |
|---|---|---|---|
| GPT-1900 (`mhla/gpt1900-d34-22btok`) | 3.29B | 1900 | Primary |
| talkie-1930 (`talkie-lm/talkie-1930-13b-base`) | 13B | 1930 | Second cutoff, test set 1931 to 1960 |
| talkie-web-13b | 13B | modern | Same architecture, modern data. Control |

**Data.** Each item is a checked `Γ ⊢ ⊥` with every correction set known.

- Train, pre-1900 only: phlogiston, caloric, spontaneous generation, miasma, plus synthetic contradictions.
- Train controls where the law survived: Uranus's anomaly, fixed by Neptune and not by dropping Newton.
- Test, 1900 to 1930, never trained on: flight, the photon, special relativity, Mercury, radioactive transmutation, continental drift.

The mix of survived and revised laws stops the model learning "always drop the famous one".

**Leak probe.** An item is used only if the base model fails to state its answer when asked directly.

**Methods, cheapest first.**

1. Ledger as a tool at inference. No training.
2. Fine-tune on revision traces from the pre-1900 set.
3. RL with the checker as the only reward: the answer's correction sets match the enumerated ones.
4. Steering along the direction that separates famous from stripped items.

Stop at the first method that passes the end goal.

**Prior work.** GPT-1900's author already ran contradiction and discovery RL toward the photon and relativity. Reported result: failure on most tasks, with leakage suspected. Our difference: reward and score come from a checker, test items are leak-probed, and the target is the assumption to revise, not the finished theory.

**Compute.** Sophia: fine-tuning and RL for both models. Polaris: evaluation sweeps. AWS: checker farm and item construction.

## Later

Symbolic regression to produce the replacement law. Re-execution checker for biology.

## What stays unverified

- **Fidelity.** That a formal `φ` means the informal claim. Reviewed by a second model family, recorded in `Γ`.
- **Atomic assumptions.** An observation's `Γ` is a stated convention. Nothing proves an instrument.
- **Transfer.** Whether a model trained on sequents holds the rule outside formal domains.
