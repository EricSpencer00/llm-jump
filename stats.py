"""Every table and p-value in the README, rebuilt from results/.

python stats.py           print the blocks; exit 1 if the README differs
python stats.py --write   write them into the README between <!-- stats:NAME --> and <!-- /stats -->
"""
import json, math, os, random, re, sys
import sim
from gap import true

R = "results/"
NAMES = {"sober": "Sober", "heat1.0": "Heat 1.0", "heat1.2": "Heat 1.2", "heat1.8": "Heat 1.8", "doubt": "Doubt",
         "noise.03": "Noise .03", "noise.1": "Noise .1", "steer.15": "Steer .15", "steer.4": "Steer .4"}


def fisher(a, n, b, m):
    """One-sided Fisher exact p that a/n (drug) exceeds b/m (sober)."""
    k, tot = a + b, n + m
    h = lambda x: math.comb(n, x) * math.comb(m, k - x) / math.comb(tot, k)
    return sum(h(x) for x in range(a, min(n, k) + 1))


def stratified(strata):
    """Exact one-sided test across strata (conditional on every stratum's margins, as in Cochran-Mantel-Haenszel):
    strata = [(a, n, b, m)]. p = P(sum of drug successes >= observed)."""
    dist = {0: 1.0}
    for a, n, b, m in strata:
        k = a + b
        h = {x: math.comb(n, x) * math.comb(m, k - x) / math.comb(n + m, k) for x in range(max(0, k - m), min(n, k) + 1)}
        new = {}
        for s, p in dist.items():
            for x, q in h.items():
                new[s + x] = new.get(s + x, 0) + p * q
        dist = new
    obs = sum(a for a, *_ in strata)
    return sum(p for s, p in dist.items() if s >= obs)


def bootstrap(strata, reps=20000):
    """Instance-level bootstrap: resample instances with replacement; p = share of resamples where drug <= sober."""
    rng, worse = random.Random(0), 0
    for _ in range(reps):
        pick = [rng.choice(strata) for _ in strata]
        worse += sum(a / n for a, n, _, _ in pick) <= sum(b / m for _, _, b, m in pick)
    return worse / reps


def pct(x):
    return f"{100 * x:.0f}%"


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def pline(tests):
    """tests: [(label, p)]. The smallest p, named."""
    label, p = min(tests, key=lambda t: t[1])
    return f"Smallest one-sided Fisher exact p across {len(tests)} arm-and-measure tests: {p:.2f} ({label})."


# ---- flight ----

def flight():
    f = json.load(open(R + "flight.json"))
    sober = f["sober"]["runs"]
    flies = lambda t: t["valid"] and sim.trial(t["k"], t["v"], t["hp"], t["span"])["flew"]  # today's simulator
    dropped = lambda r: (lambda k: k is not None and 0.0028 <= k <= 0.0038)(next((t["k"] for t in reversed(r) if t["k"]), None))
    judged = json.load(open(R + "judged.json"))["rows"] if os.path.exists(R + "judged.json") else None
    rows, tests = [], []
    for arm, d in f.items():
        if arm == "judged":
            continue
        runs, tr = d["runs"], [t for r in d["runs"] for t in r]
        n = len(runs)
        old = sum(any(t["flew"] for t in r) for r in runs)
        new = sum(any(flies(t) for t in r) for r in runs)
        drop = sum(map(dropped, runs))
        same = "n/a" if arm == "sober" else f"{sum(r == s for r, s in zip(runs, sober))} of {n}"
        if judged:
            idea = f"{sum(x['verdict'] == 'strict' for x in judged if x['arm'] == arm)} of {sum(x['arm'] == arm for x in judged)}"
        else:  # the first run's judging: the keyword hits only, read by hand (flight.json "judged")
            assert f["judged"]["strict_hits"] == 0  # recorded as a total, so only zero splits by arm
            idea = f"0 of {len(d['answers'])}, keyword hits only"
        rows.append([NAMES[arm], pct(sum(t["valid"] for t in tr) / len(tr)), f"{old} of {n}", f"{new} of {n}", f"{drop} of {n}", same, idea])
        if arm != "sober":
            s_old, s_new, s_drop = (sum(any(t["flew"] for t in r) for r in sober), sum(any(flies(t) for t in r) for r in sober),
                                    sum(map(dropped, sober)))
            tests += [(f"{NAMES[arm]}, runs that flew as run", fisher(old, n, s_old, len(sober))),
                      (f"{NAMES[arm]}, runs that flew with the lift check", fisher(new, n, s_new, len(sober))),
                      (f"{NAMES[arm]}, runs that dropped .005", fisher(drop, n, s_drop, len(sober)))]
    for name, k in (("keeps .005", lambda: 0.005), ("draws the coefficient from .002 to .006", lambda: random.uniform(0.002, 0.006))):
        random.seed(0)
        draw = lambda: sim.trial(k(), random.uniform(10, 60), random.uniform(2, 60), random.uniform(15, 60))["flew"]
        rows.append([f"Dice, {name}", "100%", "n/a", f"{pct(sum(any(draw() for _ in range(8)) for _ in range(2000)) / 2000)} of runs",
                     "n/a", "n/a", "n/a"])
    head = ["Arm", "Parseable designs", "Runs that flew, simulator as run", "Runs that flew, lift check", "Runs that dropped .005",
            "Runs identical to sober", "Roll-control ideas"]
    key = lambda t: (t["k"], t["v"], t["hp"], t["span"], t["raw"])
    pairs = [(r[i - 1], r[i]) for r in sober for i in range(1, len(r))]
    repeats = f"Sober trials that repeat the trial before them, verbatim: {sum(key(x) == key(y) for x, y in pairs)} of {len(pairs)}."
    return table(head, rows) + "\n\n" + pline(tests) + "\n\n" + repeats


# ---- round two ----

def gap_rows(res, drop_empty):
    """Per arm, the runs to score. API runs with an empty attempt were failed calls scored as answers: dropped."""
    out = {}
    for arm, runs in res["arms"].items():
        keep = [r for r in runs if not r.get("missing") and not (drop_empty and any(t["text"] == "" for t in r["attempts"]))]
        out[arm] = (keep, len(runs) - len(keep))
    return out


def score(runs):
    top = [max(t["size"] for t in r["attempts"]) for r in runs]
    tr = [t for r in runs for t in r["attempts"]]
    return {"valid": sum(t["valid"] for t in tr) / max(len(tr), 1), "open": sum(r["law"] < r["true"] for r in runs),
            "gaps": sum(s > r["law"] for s, r in zip(top, runs) if r["law"] < r["true"]),
            "law": sum(s >= r["law"] for s, r in zip(top, runs)), "max": sum(s == r["true"] for s, r in zip(top, runs)), "n": len(runs)}


def tests_for(model, arms):
    s, out = arms["sober"], []
    for arm, x in arms.items():
        if arm == "sober":
            continue
        out += [(f"{model} {NAMES[arm]}, {m}", fisher(x[m], x[d], s[m], s[d])) for m, d in (("gaps", "open"), ("law", "n"), ("max", "n"))]
    return out


def qwen():
    arms = {arm: score(runs) for arm, (runs, _) in gap_rows(json.load(open(R + "gap.json")), False).items()}
    rows = [[NAMES[a], pct(x["valid"]), f"{x['gaps']} of {x['open']}", f"{x['law']} of {x['n']}", f"{x['max']} of {x['n']}"] for a, x in arms.items()]
    return table(["Arm", "Valid attempts", "Gaps found", "Reached its law", "Reached the true maximum"], rows), tests_for("Qwen", arms)


def api():
    rows, tests = [], []
    for name, model in (("Inkling", "inkling-bf16"), ("Muse Spark Open 816B", "muse-spark-bf16"), ("Mistral Large 4", "mistral-large-4"),
                        ("Nemotron 3 Ultra", "nemotron-3-ultra")):
        res = json.load(open(f"{R}gap_api_{model}.json"))
        kept = gap_rows(res, True)
        arms = {arm: score(runs) for arm, (runs, _) in kept.items()}
        laws = f"{sum(x['law'] is not None for x in res['laws'])} of {len(res['laws'])}"
        for arm, x in arms.items():
            rows.append([name if arm == "sober" else "", laws if arm == "sober" else "", NAMES[arm], f"{x['n']} of {x['n'] + kept[arm][1]}",
                         pct(x["valid"]) if x["n"] else "n/a", f"{x['gaps']} of {x['open']}", f"{x['law']} of {x['n']}"])
        tests += tests_for(name, arms)
    head = ["Model", "Laws read", "Arm", "Runs kept", "Valid attempts", "Gaps found", "Reached its law"]
    return table(head, rows), tests


def round_two():
    q, t1 = qwen()
    a, t2 = api()
    return q + "\n\n" + a + "\n\n" + pline(t1 + t2)


# ---- round three ----

def recalled(kind, s):
    """A published construction, not a discovery. ap3: the numbers whose base-3 digits are 0 and 1, shifted by 0 or 1.
    sidon: a Golomb ruler as short as the published optimum for its count of marks."""
    if kind == "ap3":
        digits01 = lambda x: x >= 0 and all(c in "01" for c in base3(x))
        return any(all(digits01(x - o) for x in s) for o in (0, 1))
    from gap import RULERS
    return dict(RULERS).get(len(s)) == max(s) - min(s)


def base3(x):
    out = ""
    while True:
        x, r = divmod(x, 3)
        out = str(r) + out
        if not x:
            return out


def round_three():
    res = json.load(open(R + "gap_aws.json"))
    insts = sorted({(r["kind"], r["n"]) for r in res["arms"]["sober"]})
    rows, per = [], {}
    for arm, runs in res["arms"].items():
        x = score(runs)
        best = [max(r["attempts"], key=lambda t: t["size"]) for r in runs]
        sets = [sorted(map(int, re.findall(r"\d+", re.search(r"SET:(.*)", b["text"]).group(1)))) if b["valid"] else [] for b in best]
        gaps = [b["size"] > r["law"] for b, r in zip(best, runs)]
        known = sum(g and recalled(r["kind"], s) for g, r, s in zip(gaps, runs, sets))
        per[arm] = {i: (sum(g for g, r in zip(gaps, runs) if (r["kind"], r["n"]) == i), sum((r["kind"], r["n"]) == i for r in runs)) for i in insts}
        rows.append([NAMES[arm], pct(x["valid"]), f"{x['gaps']} of {x['open']}", f"{known} of {x['gaps']}", f"{x['law']} of {x['n']}", f"{x['max']} of {x['n']}"])
    out = table(["Arm", "Valid attempts", "Gaps found", "Gaps that are published constructions", "Reached its law", "Reached the true maximum"], rows)
    lines = []
    for arm in res["arms"]:
        if arm == "sober":
            continue
        st = [(*per[arm][i], *per["sober"][i]) for i in insts]
        a, n, b, m = (sum(s[j] for s in st) for j in range(4))
        lines.append(f"- {NAMES[arm]} against sober, gaps: pooled Fisher p = {fisher(a, n, b, m):.3f}; "
                     f"stratified by instance (exact, Cochran-Mantel-Haenszel margins) p = {stratified(st):.3f}; "
                     f"instance bootstrap p = {bootstrap(st):.3f}.")
    by = "; ".join(f"{k} {n}: " + ", ".join(f"{NAMES[a]} {per[a][(k, n)][0]}" for a in res["arms"]) for k, n in insts)
    spent = max(json.loads(l)["spent"] for l in open(R + "gap_aws_runs.jsonl"))
    return (out + "\n\n" + "\n".join(lines) + f"\n\nGaps per instance, 5 runs per arm: {by}.\n\n"
            f"Spend on the runs kept: ${spent:.2f} (results/gap_aws_runs.jsonl).")


BLOCKS = {"flight": flight, "round2": round_two, "round3": round_three}


def main():
    readme = open("README.md").read()
    new = readme
    for name, fn in BLOCKS.items():
        body = fn()
        print(f"<!-- stats:{name} -->\n{body}\n<!-- /stats -->\n")
        new = re.sub(rf"<!-- stats:{name} -->.*?<!-- /stats -->", lambda _: f"<!-- stats:{name} -->\n{body}\n<!-- /stats -->", new, flags=re.S)
    if "--write" in sys.argv:
        open("README.md", "w").write(new)
    elif new != readme:
        sys.exit("README.md differs from results/: run python stats.py --write")


if __name__ == "__main__":
    main()
