# giving an LLM drugs to see if it can jump

We take a language model that has read nothing after 1900, tell it Smeaton's coefficient is law, get it high in three different ways, and ask it to invent the airplane. A flight simulator decides who flew. Design: [plan.md](plan.md).

## Result (2026-11-21)

Nobody invented the airplane.

| Arm | Buildable designs | Flights per trial | Runs that flew | Runs that dropped .005 | Roll-control ideas |
|---|---|---|---|---|---|
| Sober | 80% | 6% | 3 of 30 | 0 of 30 | 0 of 100 |
| Heat 1.2 | 17% | 2% | 4 of 30 | 0 of 30 | 0 of 100 |
| Heat 1.8 | 2% | 0% | 0 of 30 | 2 of 30 | 0 of 100 |
| Noise .03 | 84% | 0% | 1 of 30 | 0 of 30 | 0 of 100 |
| Noise .1 | 86% | 4% | 2 of 30 | 0 of 30 | 0 of 100 |
| Steer .15 | 80% | 7% | 3 of 30 | 0 of 30 | 0 of 100 |
| Steer .4 | 78% | 9% | 3 of 30 | 0 of 30 | 0 of 100 |
| A designer rolling dice | 100% | 21% | 95% | n/a | n/a |

Model: GPT-1900 instruct (`mhla/gpt1900-instruct-v3-sft`, 3.29B). Eight trials per run, 240 per arm.

## Reading

- **No drug beats sober.** The end goal fails on every milestone.
- **Dice beat the model.** A designer who draws speed, power and span at random flies three times as often as GPT-1900 in any state of mind.
- **The dogma survived everything.** Told after every trial that its lift measured 66 per cent of the computed, the model wrote .005 again in 222 to 228 of 240 trials in every coherent arm. The two runs that "dropped" it were on the highest heat, where the model was emitting random digits.
- **Heat is the drug that works as advertised.** The model becomes very creative and stops producing anything that can be built.
- **Noise and steering change nothing you can measure.** Steering toward "a mere supposition" left the supposition exactly where it was.
- **The roll-control idea never appeared.** 700 answers, zero that have the pilot change the angle of the two wings in opposite senses. Proposed instead: reins, a paraffin bath, a mallet, and "the most gentle application of his breath". Four loose near-misses: one sober, one heat, two steering.

**What this shows.** The model is not held back by dogma. It is held back by not being able to design a machine. A drug widens what a model says. It does not add what the model cannot do. The test needs a model that can do the task sober.

## Limits

- Simulator constants are estimates fitted to the 1903 Flyer's published figures.
- Doses were picked once, two per drug, and never tuned.
- Figures are drawn from number tokens only, and every run starts from one seed machine. Without both, the model wrote letters beginning "My dear Sir".
- Idea answers were judged by one reader, an LLM, blind to arm.
- Trials inside a run are not independent, so the per-trial rates overstate the sample.
- Post-1900 leakage in the model's tuning data is not ruled out. It did not help.

## Reproduce

```
python sim.py     # checks the simulator against the 1903 Flyer
qsub run.pbs      # all arms; needs sft/ (the model) and base/nanochat/ (its code)
```

Every design and every answer: [results/flight.json](results/flight.json). An earlier, off-scope experiment (a fine-tune that taught the model to say "false") is in git history at `755d279`.
