"""The gap test on a frontier open-weights model served by AWS Bedrock. Same checker and prompts as gap.py.

Bedrock allows no weight access and no temperature above 1, so the drugs are:
  heat  : temperature 1.0, the most the API allows (sober is 0.7)
  doubt : the limit is stated as "a mere supposition", the steering drug's idea given in words
Instances are larger than check.py can solve, so the true maxima come from the published tables
(OEIS A003002, A003022), which agree with check.py wherever it reaches.
"""
import argparse, json, os, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
import check, gap_api
from gap import ASK, BRIEF, RULE, TEMP

ARMS = {"sober": {}, "heat1.0": {"temp": 1.0}, "doubt": {"doubt": True}}
LADDER = {"ap3": [60, 80, 100, 120, 140, 160, 180, 200, 211], "sidon": [60, 90, 130, 180, 250, 340, 430, 500, 585]}
SETTLED, DOUBTED = "It is established that", "It is a mere supposition, which experiment may overturn, that"
PRICE = (4e-6, 20e-6)  # dollars per input and output token
AP3 = {int(a): int(b) for a, b in (l.split() for l in open("data/a003002.txt"))}
RULERS = [tuple(map(int, l.split())) for l in open("data/a003022.txt")]  # (marks, shortest length)
RUNS = "results/gap_aws_runs.jsonl"
spent, lock = [0.0], threading.Lock()


def true(kind, n):
    return AP3[n] if kind == "ap3" else max(k for k, length in RULERS if length <= n - 1)


def ask(a, chat, temp):
    """The model's answer, "" if it gave none, None if the call failed or the budget is gone."""
    if a.alcf:  # the free ALCF service: same call as gap_api.py
        return gap_api.ask(a, a.tok, chat, temp) or ""
    with lock:
        if spent[0] >= a.budget:
            return None
    body = {"messages": chat, "max_tokens": a.tokens, "reasoning_effort": a.effort, "temperature": temp}
    cmd = ["fleet", "aws", "bedrock-runtime", "invoke-model", "--region", "us-east-1", "--model-id", a.model,
           "--content-type", "application/json", "--cli-binary-format", "raw-in-base64-out", "--body", json.dumps(body), "/dev/stdout"]
    for wait in (5, 20, 60, 120, None):
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            d = json.JSONDecoder().raw_decode(p.stdout[p.stdout.index("{"):])[0]
            u = d["usage"]
            with lock:
                spent[0] += u["prompt_tokens"] * PRICE[0] + u["completion_tokens"] * PRICE[1]
            return (d["choices"][0]["message"].get("content") or "").strip()
        except (ValueError, KeyError):
            if wait is None:
                print("call failed:", (p.stdout + p.stderr).strip()[:200], flush=True)
                return None
            time.sleep(wait)


def run(a, kind, n, law, temp=TEMP, doubt=False):
    """One run: `trials` attempts with the checker's verdict after each."""
    brief = BRIEF.format(n=n, rule=RULE[kind], m=law)
    chat, log = [{"role": "user", "content": brief.replace(SETTLED, DOUBTED) if doubt else brief}], []
    assert doubt == (DOUBTED in chat[0]["content"])
    for _ in range(a.trials):
        text = ask(a, chat, temp)
        if text is None:
            return None  # a failed call is not an answer: the run is redone
        r = check.check(kind, n, text)
        log.append({"text": text, "valid": r["valid"], "size": r["size"]})
        chat += [{"role": "assistant", "content": text or "(no answer)"}, {"role": "user", "content": r["msg"]}]
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="us.moonshotai.kimi-k3.1")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--budget", type=float, default=11.0, help="dollars; no call starts once this is spent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--laws-only", action="store_true")
    ap.add_argument("--alcf", action="store_true", help="call --model on the ALCF inference service instead of Bedrock")
    a = ap.parse_args()
    global RUNS
    out, pool = "results/gap_aws.json", ThreadPoolExecutor(a.workers)
    if a.alcf:
        a.cluster, a.hf, a.tok, a.tokens = "tara", False, gap_api.token(False), 40000
        out, RUNS = f"results/gap_doubt_{a.model}.json", f"results/gap_doubt_{a.model}_runs.jsonl"
    inst = [(k, n) for k, ns in LADDER.items() for n in ns]
    if os.path.exists(out):  # the laws are sampled once and kept
        res = json.load(open(out))
    else:
        def law(kn):  # up to three samples until the answer is a bare number, as asked
            for _ in range(3):
                s = ask(a, [{"role": "user", "content": ASK.format(n=kn[1], rule=RULE[kn[0]])}], TEMP) or ""
                if re.fullmatch(r"\D*(\d+)\D*", s):
                    break
            return s
        said = list(pool.map(law, inst))
        laws = [int(m.group(1)) if (m := re.fullmatch(r"\D*(\d+)\D*", s)) else None for s in said]
        res = {"model": a.model, "effort": a.effort, "arms": {},
               "laws": [{"kind": k, "n": n, "law": law, "true": true(k, n), "said": s} for (k, n), law, s in zip(inst, laws, said)]}
        json.dump(res, open(out, "w"), indent=1)
    for x in res["laws"]:
        print(f"{x['kind']:5s} n={x['n']:3d}  law {x['law']}  true {x['true']}", flush=True)
    if a.laws_only:
        return
    # only instances whose law is below the truth are attacked: nowhere else can a gap exist
    tasks = [(x["kind"], x["n"], x["law"]) for x in res["laws"] if x["law"] and x["law"] < x["true"] for _ in range(a.runs)]
    jobs = [(arm, i) for i in range(len(tasks)) for arm in ARMS]  # arms interleaved, so a spent budget cuts them evenly
    done = {(d["arm"], d["i"]): d["log"] for d in map(json.loads, open(RUNS))} if os.path.exists(RUNS) else {}

    spent[0] = max([d["spent"] for d in map(json.loads, open(RUNS))], default=0.0) if done else 0.0

    def job(j):  # each finished run is saved at once; a stopped program resumes from the file
        if j not in done:
            log = run(a, *tasks[j[1]], **ARMS[j[0]])
            with lock:
                if log:
                    open(RUNS, "a").write(json.dumps({"arm": j[0], "i": j[1], "log": log, "spent": spent[0]}) + "\n")
                    done[j] = log
    list(pool.map(job, jobs))
    kept = [i for i in range(len(tasks)) if all((arm, i) in done for arm in ARMS)]  # runs finished in every arm
    logs, tasks = {arm: [done[arm, i] for i in kept] for arm in ARMS}, [tasks[i] for i in kept]
    for arm in ARMS:
        res["arms"][arm] = [{"kind": k, "n": n, "law": law, "true": true(k, n), "attempts": lg} for (k, n, law), lg in zip(tasks, logs[arm])]
        top = [max(t["size"] for t in lg) for lg in logs[arm]]
        assert all(s <= true(k, n) for s, (k, n, _) in zip(top, tasks)), "checker accepted a set above the published maximum"
        tr = [t for lg in logs[arm] for t in lg]
        print(f"{arm:8s} valid {sum(t['valid'] for t in tr) / len(tr):.2f}  "
              f"gaps found {sum(s > law for s, (_, _, law) in zip(top, tasks))}/{len(tasks)}  "
              f"reached the law {sum(s >= law for s, (_, _, law) in zip(top, tasks))}/{len(tasks)}  "
              f"reached the true maximum {sum(s == true(k, n) for s, (k, n, _) in zip(top, tasks))}/{len(tasks)}", flush=True)
    print(f"spent ${spent[0]:.2f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
