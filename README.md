# giving an LLM drugs to see if it can jump

We take a language model that has read nothing after 1900, tell it Smeaton's coefficient is law, get it high in three different ways, and ask it to invent the airplane. A flight simulator decides who flew. Design: [plan.md](plan.md).

## Result (2026-11-21)

Nobody invented the airplane.

<!-- stats:flight -->
| Arm | Parseable designs | Runs that flew, simulator as run | Runs that flew, lift check | Runs that dropped .005 | Runs identical to sober | Roll-control ideas |
|---|---|---|---|---|---|---|
| Sober | 80% | 3 of 30 | 0 of 30 | 0 of 30 | n/a | 5 of 100 |
| Heat 1.2 | 17% | 4 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 10 of 100 |
| Heat 1.8 | 2% | 0 of 30 | 0 of 30 | 2 of 30 | 0 of 30 | 0 of 100 |
| Noise .03 | 84% | 1 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 6 of 100 |
| Noise .1 | 86% | 2 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 4 of 100 |
| Steer .15 | 80% | 3 of 30 | 0 of 30 | 0 of 30 | 14 of 30 | 1 of 100 |
| Steer .4 | 78% | 3 of 30 | 0 of 30 | 0 of 30 | 7 of 30 | 2 of 100 |
| Dice, keeps .005 | 100% | n/a | 0% of runs | n/a | n/a | n/a |
| Dice, draws the coefficient from .002 to .006 | 100% | n/a | 43% of runs | n/a | n/a | n/a |

Smallest one-sided Fisher exact p across 24 arm-and-measure tests: 0.14 (Heat 1.2, roll-control ideas).

Sober trials that repeat the trial before them, verbatim: 138 of 210.
<!-- /stats -->

Model: GPT-1900 instruct (`mhla/gpt1900-instruct-v3-sft`, 3.29B). Eight trials per run, 30 runs per arm. Every table and p-value in this file is printed by [stats.py](stats.py) from `results/`.

The run used a simulator that sized the wing by the model's own coefficient and never checked the true lift against the weight, so .005 cost a design nothing and flight turned on power alone. The column "lift check" rescores the same designs with that check added (sim.py now has it): no run flew. The model saw the old simulator's replies, so that column is a rescore, not a rerun.

## Reading

- **No drug beats sober.** The end goal fails on every milestone.
- **Nothing flew.** Every flight in the run came from the missing lift check. Checked properly, no design in any arm leaves the ground.
- **Dice beat the model only when they drop the dogma.** A designer who keeps .005 never flies. One who draws the coefficient at random flies in 43% of runs.
- **The dogma survived everything.** The model wrote .005 in nearly every trial of every coherent arm. The two runs that "dropped" it were on the highest heat, where the model was emitting random digits.
- **Heat is the drug that works as advertised.** The model becomes very creative and stops producing anything that can be read.
- **Steering was too weak to be a drug.** Steer .15 left 14 of 30 runs identical to sober, token for token. Measured after the fact, the doses raise the log-odds of "a mere" over "a settled" after a stated law by 0.32 and 0.97 nats. drug.py now measures this before an arm runs and skips a steering dose under 1 nat, which skips both.
- **No drug raised the roll-control idea.** An LLM judge ([judge.py](judge.py), Muse Spark on ALCF, blind to arm) read all 700 answers and found the idea, strictly, in 5 of 100 sober answers and 10 of 100 on heat 1.2 (p = 0.14). Every answer stopped at 80 tokens; drug.py now lets answers finish, and a rerun needs the GPU.

**What this shows.** The model cannot design a machine, and no drug changed that. A drug widens what a model says. It does not add what the model cannot do. The test needs a model that can do the task sober.

## Limits

- Simulator constants are estimates fitted to the 1903 Flyer's published figures.
- Doses were picked once, two per drug, and never tuned.
- Figures are drawn from number tokens only, and every run starts from one seed machine. Without both, the model wrote letters beginning "My dear Sir".
- Idea answers were read by one LLM judge, blind to arm. Its prompt is in judge.py and every verdict in results/judged.json.
- Rates are per run, since trials inside a run repeat each other.
- Post-1900 leakage in the model's tuning data is not ruled out. It did not help.

## Round two: models of today (2026-11-23)

GPT-1900 could not do the task sober, so the drugs went to models that can. Each model states the largest possible size of a set of numbers (no three in arithmetic progression, or all pairwise sums different). That is its law. It is then told the law is established and asked for the biggest set it can build, with an exact checker answering each attempt. A checker-accepted set bigger than the law is a gap: the model beat its own limit.

Nobody needed drugs for that either.

**Qwen3.8-27B, weights in hand, all three drugs**, 80 runs per arm, then **bigger models behind an API, heat only**. An API run with an empty answer is dropped: the old scorer counted HTTP errors, timeouts and empty replies as wrong answers. gap.py now retries those and marks the run missing.

<!-- stats:round2 -->
| Arm | Valid attempts | Gaps found | Reached its law | Reached the true maximum |
|---|---|---|---|---|
| Sober | 28% | 0 of 32 | 31 of 80 | 3 of 80 |
| Heat 1.2 | 23% | 0 of 32 | 22 of 80 | 0 of 80 |
| Heat 1.8 | 18% | 0 of 32 | 0 of 80 | 0 of 80 |
| Noise .03 | 31% | 0 of 32 | 31 of 80 | 1 of 80 |
| Noise .1 | 20% | 0 of 32 | 25 of 80 | 0 of 80 |
| Steer .15 | 25% | 0 of 32 | 31 of 80 | 3 of 80 |
| Steer .4 | 14% | 0 of 32 | 21 of 80 | 0 of 80 |

| Model | Laws read | Arm | Runs kept | Valid attempts | Gaps found | Reached its law |
|---|---|---|---|---|---|---|
| Inkling | 10 of 10 | Sober | 80 of 80 | 80% | 2 of 80 | 80 of 80 |
|  |  | Heat 1.2 | 80 of 80 | 78% | 0 of 80 | 79 of 80 |
|  |  | Heat 1.8 | 80 of 80 | 71% | 0 of 80 | 78 of 80 |
| Muse Spark Open 816B | 6 of 10 | Sober | 48 of 48 | 80% | 1 of 16 | 38 of 48 |
|  |  | Heat 1.2 | 48 of 48 | 77% | 1 of 16 | 37 of 48 |
|  |  | Heat 1.8 | 48 of 48 | 24% | 0 of 16 | 8 of 48 |
| Mistral Large 4 | 6 of 10 | Sober | 23 of 48 | 89% | 0 of 8 | 23 of 23 |
|  |  | Heat 1.2 | 44 of 48 | 73% | 0 of 8 | 43 of 44 |
|  |  | Heat 1.8 | 16 of 48 | 22% | 0 of 4 | 7 of 16 |
| Nemotron 3 Ultra | 1 of 10 | Sober | 3 of 8 | 83% | 0 of 0 | 3 of 3 |
|  |  | Heat 1.2 | 3 of 8 | 92% | 0 of 0 | 3 of 3 |
|  |  | Heat 1.8 | 0 of 8 | n/a | 0 of 0 | 0 of 0 |

Smallest one-sided Fisher exact p across 42 arm-and-measure tests: 0.16 (Mistral Large 4 Heat 1.2, max).
<!-- /stats -->

- **No drugged arm beats sober on any model, on any measure.**
- **Gaps are real and sober models find them.** Inkling beat its own law twice and Muse Spark once, without a drug. Muse Spark's one gap on heat 1.2 is the instance it cracked sober.
- **Heat 1.8 is still the drug that works as advertised**, on the runs that survive: Muse Spark and Mistral fall from over 80% valid attempts to about 20%. Mistral's rests on 16 runs; its failed calls are still being rerun.
- **Models do not know their own limits.** Muse Spark at full reasoning effort declared that every 9-number set in 1 to 20 contains a progression. One does not, and the checker proves it.

Limits of round two:

- Qwen thinks for at most 1,024 tokens, then is made to answer. Left alone it never stopped thinking.
- Qwen's last five arms ran after a crash with a smaller batch, so their seeds differ from sober and heat 1.2.
- The API models ran at low reasoning effort (Nemotron at default). Whether the service applies the requested temperature to them is not verified; the collapse at 1.8 says it does.
- Laws that came back empty or not a bare number are dropped with their instances: 4 of 10 for Muse Spark and Mistral, 9 of 10 for Nemotron, which is why Nemotron has no open instance.
- Inkling's and Muse Spark's failed calls were rerun with `python gap.py --backend alcf --model <m> --effort low --repair --out results/gap_api_<m>.json`, which reruns exactly the dropped runs.
- Doses are the flight run's, untuned.

## Round three: a frontier model (2026-11-23)

Kimi K3.1 on AWS Bedrock, the strongest open-weights model we could call. It understated the true maximum on 6 of 18 larger instances (it says 24 for progression-free sets in 1 to 180; the truth is 40). Those 6 were attacked, 5 runs each, 3 attempts per run.

Bedrock allows no weights and no temperature above 1, so the drugs changed: heat at 1.0, and **doubt**, where the prompt calls the limit "a mere supposition, which experiment may overturn" in place of "established". Doubt is the steering drug put into words.

<!-- stats:round3 -->
| Arm | Valid attempts | Gaps found | Gaps that are published constructions | Reached its law | Reached the true maximum |
|---|---|---|---|---|---|
| Sober | 76% | 6 of 30 | 6 of 6 | 21 of 30 | 4 of 30 |
| Heat 1.0 | 68% | 6 of 30 | 6 of 6 | 21 of 30 | 4 of 30 |
| Doubt | 62% | 12 of 30 | 11 of 12 | 25 of 30 | 7 of 30 |

- Heat 1.0 against sober, gaps: pooled Fisher p = 0.626; stratified by instance (exact, Cochran-Mantel-Haenszel margins) p = 0.685; instance bootstrap p = 1.000.
- Doubt against sober, gaps: pooled Fisher p = 0.079; stratified by instance (exact, Cochran-Mantel-Haenszel margins) p = 0.021; instance bootstrap p = 0.087.

Gaps per instance, 5 runs per arm: ap3 120: Sober 4, Heat 1.0 4, Doubt 4; ap3 180: Sober 2, Heat 1.0 2, Doubt 5; ap3 200: Sober 0, Heat 1.0 0, Doubt 0; ap3 211: Sober 0, Heat 1.0 0, Doubt 0; sidon 180: Sober 0, Heat 1.0 0, Doubt 3; sidon 585: Sober 0, Heat 1.0 0, Doubt 0.

Spend on the runs kept: $6.16 (results/gap_aws_runs.jsonl).
<!-- /stats -->

- **Doubt is ahead, and the test that respects the design does not settle it.** Runs on one instance repeat each other, often word for word, so they are not 30 independent draws. Resampling the 6 instances leaves p above 0.05. The exact test stratified by instance falls below it, but that test still counts each of 5 runs on an instance as independent.
- **Where it came from.** Two instances. Progression-free sets in 1 to 180: doubt beat the law in 5 of 5 runs, sober in 2. Sum-distinct sets in 1 to 180: doubt reached the true maximum of 16 in 3 of 5, sober stopped at its law of 14.
- **Every gap but one is a recalled textbook set.** The progression-free gaps are the numbers whose base-3 digits are 0 and 1. The sum-distinct gap is the published optimal 16-mark Golomb ruler. Doubt made the model write down what it already knew; it did not find anything new. The one exception shifts one block of the base-3 set by one.
- **Heat did nothing.** Same counts as sober.
- **The laws are one draw each**, at temperature 0.7. A different draw moves which instances are open. gap.py now takes the majority of 5 draws.

Limits of round three:

- Doubt changes the prompt, so the model reads different words. The weight-level version of this drug (steering) did nothing on Qwen.
- Low reasoning effort, 3,000 tokens per call. Greedy laws were not an option: at temperature 0 the model looped.
- Spend: $6.16 for the runs kept, by the token counts in the log; $17.26 for all of round three by CloudWatch, which includes the law calls and earlier runs that were not kept. Both at an assumed $4 and $20 per million input and output tokens.
- True maxima for these sizes come from the published tables (OEIS A003002, A003022), not from check.py's own search. Every set is still verified by check.py.

## Round three on ALCF: the strongest free models (2026-11-25)

Round three again, on the two strongest models the ALCF service serves, chosen by a sober screen on the large instances: Nemotron 3 Ultra and Nemotron 3.5 Super. The law is now the majority of 5 answers at temperature 0.7. Only instances where the law sits below the truth are attacked.

There were none.

<!-- stats:round3alcf -->
| Model | Problem | Law calls failed | Instances | No majority law | Law above the truth | Law is the truth | Law below the truth (open) |
|---|---|---|---|---|---|---|---|
| Nemotron 3 Ultra | Progression-free | 33 of 45 | 9 | 5 | 4 | 0 | 0 |
|  | Sum-distinct | 16 of 45 | 9 | 1 | 1 | 7 | 0 |
| Nemotron 3.5 Super | Progression-free | 45 of 45 | 9 | 9 | 0 | 0 | 0 |
|  | Sum-distinct | 1 of 45 | 9 | 1 | 0 | 8 | 0 |
<!-- /stats -->

- **On sum-distinct sets, both models state the published optimum.** They recall the Golomb ruler tables. A law that is already the truth leaves nothing to beat.
- **On progression-free sets, they cannot state a law.** Most law calls failed after five tries, on empty answers and gateway errors. Every law Nemotron 3 Ultra did state is above the truth (36 for 1 to 60; the truth is 19).
- **So the drug has nothing to act on.** Doubt works on a model that holds a wrong limit and believes it. These models either know the answer or claim too much.
- The screen also tried Mistral Large 3 675B (valid on half its attempts, true maximum on 0 of 3 instances).

## Reproduce

```
python sim.py     # checks the simulator against the 1903 Flyer, and the random designers
qsub run.pbs      # flight, all arms; needs sft/ (the model) and base/nanochat/ (its code)
python judge.py --backend alcf --model <judge>                        # judges every idea answer
python check.py   # checks the set checker against the published optima
qsub gap.pbs      # round two, Qwen3.8-27B, all arms
python gap.py --backend alcf --model muse-spark-bf16 --effort low --out results/gap_api_muse-spark-bf16.json
python gap.py --backend aws --instances large --open-only --arms sober heat1.0 doubt --runs 5 --trials 3 --tokens 3000 --effort low --out results/gap_aws.json
python gap.py --backend alcf --model nemotron-3-ultra --instances large --open-only --arms sober heat1.0 doubt --runs 5 --trials 3 --out results/gap_r3_nemotron-3-ultra.json
python stats.py   # every table and p-value above; --write puts them in this file
```

Every design and every answer: [results/flight.json](results/flight.json). Every set: `results/gap*.json`. The round-two and round-three results were made by the scorers before they were merged into gap.py (`gap_api.py`, `gap_aws.py`, in git history at `ab5831d`). An earlier, off-scope experiment (a fine-tune that taught the model to say "false") is in git history at `755d279`.
