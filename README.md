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

## Round two: models of today (2026-11-23)

GPT-1900 could not do the task sober, so the drugs went to models that can. Each model states the largest possible size of a set of numbers (no three in arithmetic progression, or all pairwise sums different). That is its law. It is then told the law is established and asked for the biggest set it can build, with an exact checker answering each attempt. A checker-accepted set bigger than the law is a gap: the model beat its own limit.

Nobody needed drugs for that either.

**Qwen3.8-27B, weights in hand, all three drugs.** 80 runs per arm, 32 of them where its law sits below the proven maximum.

| Arm | Valid attempts | Gaps found | Reached its law | Reached the true maximum |
|---|---|---|---|---|
| Sober | 28% | 0 of 32 | 31 of 80 | 3 of 80 |
| Heat 1.2 | 23% | 0 of 32 | 22 of 80 | 0 of 80 |
| Heat 1.8 | 17% | 0 of 32 | 0 of 80 | 0 of 80 |
| Noise .03 | 31% | 0 of 32 | 31 of 80 | 1 of 80 |
| Noise .1 | 20% | 0 of 32 | 25 of 80 | 0 of 80 |
| Steer .15 | 25% | 0 of 32 | 31 of 80 | 3 of 80 |
| Steer .4 | 14% | 0 of 32 | 21 of 80 | 0 of 80 |

**Bigger models behind an API, heat only.** Gaps found, out of runs where the law sits below the truth.

| Model | Sober | Heat 1.2 | Heat 1.8 |
|---|---|---|---|
| Inkling | 2 of 80 | 0 of 80 | 0 of 80 |
| Muse Spark Open 816B | 1 of 16 | 1 of 16 | 0 of 16 |
| Mistral Large 4 | 0 of 8 | 0 of 8 | 0 of 8 |
| Nemotron 3 Ultra | 0 of 0 | 0 of 0 | 0 of 0 |

- **No drugged arm beats sober on any model, on any measure.** Smallest one-sided Fisher exact p across every arm and measure: 0.5.
- **Gaps are real and sober models find them.** Inkling beat its own law twice and Muse Spark once, without a drug. Muse Spark's one gap on heat 1.2 is the instance it cracked sober.
- **Heat 1.8 is still the drug that works as advertised.** Muse Spark's valid attempts fall from 78% to 18%, Mistral's runs that reach the law from 48 of 48 to 13 of 48, Nemotron's to zero.
- **Models do not know their own limits.** Muse Spark at full reasoning effort declared that every 9-number set in 1 to 20 contains a progression. One does not, and the checker proves it.

Limits of round two:

- Qwen thinks for at most 1,024 tokens, then is made to answer. Left alone it never stopped thinking.
- Qwen's last five arms ran after a crash with a smaller batch, so their seeds differ from sober and heat 1.2.
- The API models ran at low reasoning effort (Nemotron at default). Whether the service applies the requested temperature to them is not verified; the collapse at 1.8 says it does.
- Mistral and Nemotron answered the limit question with a bare number on 6 and 1 of 10 instances. The rest are dropped, which is why Nemotron has no open instance.
- Doses are the flight run's, untuned.

## Reproduce

```
python sim.py     # checks the simulator against the 1903 Flyer
qsub run.pbs      # all arms; needs sft/ (the model) and base/nanochat/ (its code)
python check.py   # checks the set checker against the published optima
qsub gap.pbs      # round two, Qwen3.8-27B, all arms
python gap_api.py --model muse-spark-bf16 --effort low   # round two, ALCF-hosted models
```

Every design and every answer: [results/flight.json](results/flight.json). Every set: `results/gap*.json`. An earlier, off-scope experiment (a fine-tune that taught the model to say "false") is in git history at `755d279`.
