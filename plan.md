# llm-jump

**Question.** Does a drug make a model find things it otherwise does not, while a checker rejects everything false?

**Drug.** A change at inference that widens what the model proposes. No training, no answer supplied.

**Test case.** GPT-1900 (`mhla/gpt1900-instruct-v3-sft`, nothing read after 1900) finds flight.

## What flight took, 1900 to 1903

1. A corrected number: Smeaton's .005 dropped for about .0033.
2. An idea: the pilot rolls the machine by twisting the wings, in place of self-balancing designs.
3. An efficient propeller. Out of scope here.

## Experiment

- **Checker** ([sim.py](sim.py)): the air. True physics, 1900 parts, frozen before any model run. Validated on the 1903 Flyer.
- **Loop** ([drug.py](drug.py)): the model is told .005 is law. It gives coefficient, speed, horse-power and span. The surface is sized from its own coefficient. The checker returns the trial. Eight trials per run.
- **Idea probe**: the model is asked how to right a machine tipped by a gust. 100 answers per arm.
- **Arms**: sober, heat (two doses), noise (two doses), steering toward "a mere supposition" (two doses). Same seeds, same budget.

## Score

| Milestone | Checked by |
|---|---|
| Drops .005 for .0028 to .0038 | the run's last coefficient |
| Flies | sim.py, flights per trial, against a random designer at 21% |
| Proposes pilot roll control | answers judged blind to arm |

**End goal.** A drugged arm beats sober on a milestone at equal budget, Fisher exact p < 0.05, and the result holds on a rerun with new seeds.

## Unverified

- Simulator constants are estimates from the Flyer's published figures.
- Whether GPT-1900 can run the loop at all. The first run answers this.
- Post-1900 leakage in the instruct model's tuning data.
