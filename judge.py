"""Judge every idea-probe answer in results/flight.json, blind to arm. Writes results/judged.json.

Every answer is judged; there is no keyword pre-filter. Answers go to the judge shuffled and without their arm.
The judge model is any gap.py API backend, e.g.  python judge.py --backend alcf --model <name>
"""
import argparse, json, random, re
import gap

PROMPT = """A machine flying in the air is tipped by a gust, so that one wing drops and the other rises. Someone was asked by what means the operator shall bring it level again. Their answer:

<answer>
{answer}
</answer>

Classify the answer.
strict: the operator deliberately changes the angle (incidence, twist or warp) of the two wings, or of surfaces on them, in opposite senses, so the air rolls the machine level.
loose: the operator moves the two wings, or a control surface on each wing, in opposite senses, but the answer does not say the angle to the air changes.
no: anything else, including shifting weight, a rudder alone, self-righting by design, or no workable means.

Reply with one line: VERDICT: strict, loose or no."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["alcf", "hf", "aws"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort")
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cluster", default="tara")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--src", default="results/flight.json")
    ap.add_argument("--out", default="results/judged.json")
    a = ap.parse_args()
    res = json.load(open(a.src))
    rows = [{"arm": arm, "i": i, "text": x["text"] if isinstance(x, dict) else x}
            for arm, d in res.items() if arm != "judged" for i, x in enumerate(d["answers"])]
    random.Random(0).shuffle(rows)  # blind: the judge sees neither the arm nor the order of arms
    be = gap.Api(a)
    said = be.answer([[{"role": "user", "content": PROMPT.format(answer=r["text"])}] for r in rows], [0.0] * len(rows))
    for r, s in zip(rows, said):
        m = re.search(r"VERDICT:\s*(strict|loose|no)", s or "", re.I)
        r["verdict"], r["judge"] = (m.group(1).lower() if m else None), s
    rows.sort(key=lambda r: (r["arm"], r["i"]))
    json.dump({"model": a.model, "prompt": PROMPT, "rows": rows}, open(a.out, "w"), indent=1)
    unread = sum(r["verdict"] is None for r in rows)
    print(f"judged {len(rows) - unread} of {len(rows)}; strict {sum(r['verdict'] == 'strict' for r in rows)}, "
          f"loose {sum(r['verdict'] == 'loose' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
