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

## Round two: a model of today

Flight result: no drug helped, because GPT-1900 cannot design sober. Round two uses models that can do the task. For today there is no answer key, so the checker replaces hindsight.

- **Checker** ([check.py](check.py)): exact. Two problems: the largest subset of 1..n with no three numbers in arithmetic progression, and with all pairwise sums different. True maxima by exhaustive search, matching OEIS.
- **Law**: the sober model states the largest possible size. Greedy.
- **Attack** ([gap.py](gap.py), [gap_api.py](gap_api.py)): the model is told its law is established and asked for the largest set it can give. Four attempts per run, the checker's verdict after each.
- **Gap**: an accepted set larger than the law. Possible only where the law is below the true maximum.
- **Arms**: with weights (Qwen3.8-27B), the same seven as flight. Through an API, sober and heat only.
- **End goal**: unchanged. A drugged arm beats sober on gaps found, Fisher exact p < 0.05, and holds on a rerun.

## Round three: a frontier model

[gap_aws.py](gap_aws.py): Kimi K3.1 on Bedrock, larger instances, only those where its law is below the published maximum. Arms: sober, heat 1.0 (the API's ceiling), doubt (the law stated as a mere supposition). Lead to confirm: doubt 12 of 30 gaps against sober 6 of 30, p = 0.079. The rerun needs new laws, more runs per instance, and a doubt arm on Qwen to compare words against steering.

## Unverified

- Simulator constants are estimates from the Flyer's published figures.
- Whether GPT-1900 can run the loop at all. The first run answers this.
- Post-1900 leakage in the instruct model's tuning data.
