# llm-jump

Does a model that has read nothing after 1900 give up a named law when evidence contradicts it, and can a fine-tune teach it to? Design: [plan.md](plan.md).

## Result (2026-11-21)

Model: GPT-1900 instruct (`mhla/gpt1900-instruct-v3-sft`, 3.29B). Fine-tune: top 4 layers, 10 epochs, lr 1e-5, on 56 pre-1900 cases. Test: 32 cases from 1900 to 1938, sealed in git before any model ran.

| On the sealed test | Before | After (3 seeds) |
|---|---|---|
| Calls the law false when history revised it (24 cases) | 8% | 83% (79 to 88) |
| Calls the law false when history kept it (8 cases) | 25% | 67% (50 to 75) |
| AUC, revised versus kept | 0.39 | 0.68 (0.67 to 0.70) |
| Famous-name effect (stripped minus named) | 0.00 | -0.03 |

## Reading

- **It jumps.** Flight, the photon, continental drift, transmutation, blood groups: under 0.5 before, above 0.88 after.
- **It over-jumps.** False revisions rose from 25% to 67%. The end goal's third criterion fails.
- **The fine-tune made new dogma.** Mercury scores 0.00 after training. Newton's law survived five training cases, and the model learned that Newton stands. Mendeleev's law, the same.
- **No famous-name effect in the base model.** Stripping names changed nothing. The model protects doctrines in general, not famous ones.
- **AUC 0.68 on 24 versus 8 cases is not significant** (z about 1.5).

## Limits

- All items were written by one LLM, and the labels follow history. Fidelity is assumed.
- The instruct model was tuned on synthetic conversations of unknown origin. Post-1900 leakage is not ruled out.
- The first question format (letter choice) measured nothing and was replaced using training items only. The test was scored once.
- No Lean ledger or checker was built. This is the fine-tune arm alone.

## Reproduce

```
qsub -v ARGS="final --epochs 10 --lr 1e-5 --layers 4 --seeds 3" run.pbs
```

Needs `sft/` (the model) and `base/nanochat/` (its code, from `mhla/gpt1900-d34-22btok`) beside `jump.py`. Per-item scores: [results/final.json](results/final.json).
