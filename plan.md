# llm-jump

**Problem.** LLMs treat theory as law. Stated formally: they assert `φ` when only `Γ ⊢ φ` is licensed, and they refuse to revise the famous members of `Γ`.

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

## After the gate, one at a time

1. Ledger as a tool: does access remove the three failures?
2. Fine-tune on ledger sequents so the model emits `Γ` with every `φ`. Sophia.
3. Steering: attenuate the direction that separates famous-name from stripped items.
4. Symbolic regression: orbital data → checkable equation. Mercury as Test B on real history.
5. Re-execution checker. Biology.

## What stays unverified

- **Fidelity.** That a formal `φ` means the informal claim. Reviewed by a second model family, recorded in `Γ`.
- **Atomic assumptions.** An observation's `Γ` is a stated convention. Nothing proves an instrument.
- **Transfer.** Whether a model trained on sequents holds the rule outside formal domains.
