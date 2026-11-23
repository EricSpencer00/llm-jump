"""The gap test on models served over an API: the ALCF inference service, or with --hf the Hugging Face router.
Same task, checker and prompts as gap.py.

An API gives no access to the weights, so the only drug here is heat.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import check
from gap import ASK, BRIEF, INSTANCES, RULE, TEMP

ARMS = {"sober": TEMP, "heat1.2": 1.2, "heat1.8": 1.8}
URL = "https://inference-api.alcf.anl.gov/resource_server/{cluster}/api/v1/chat/completions"
HF = "https://router.huggingface.co/v1/chat/completions"
HELPER = os.path.expanduser("~/.local/bin/inference_auth_token.py")


def token(hf):
    if hf:
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    return subprocess.run([sys.executable, HELPER, "get_access_token"], capture_output=True, text=True, check=True).stdout.strip()


def ask(a, tok, chat, temp):
    """The model's answer, or "" if it gave none."""
    body = json.dumps({"model": a.model, "messages": chat, "temperature": temp, "max_tokens": a.tokens,
                       **({"reasoning_effort": a.effort} if a.effort else {})}).encode()
    for wait in (5, 15, 45, 120, 300, None):
        try:
            req = urllib.request.Request(HF if a.hf else URL.format(cluster=a.cluster), body, {"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
            return (json.load(urllib.request.urlopen(req, timeout=3000))["choices"][0]["message"].get("content") or "").strip()
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            if wait is None or getattr(e, "code", 500) in (400, 401, 402, 403):
                print("request failed:", repr(e)[:200], flush=True)
                return ""
            time.sleep(wait)


def run(a, tok, kind, n, law, temp):
    """One run: `trials` attempts with the checker's verdict after each."""
    chat, log = [{"role": "user", "content": BRIEF.format(n=n, rule=RULE[kind], m=law)}], []
    for _ in range(a.trials):
        text = ask(a, tok, chat, temp)
        r = check.check(kind, n, text)
        log.append({"text": text, "valid": r["valid"], "size": r["size"]})
        chat += [{"role": "assistant", "content": text or "(no answer)"}, {"role": "user", "content": r["msg"]}]
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cluster", default="tara")
    ap.add_argument("--hf", action="store_true", help="use the Hugging Face router")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--effort", help="reasoning effort, for models that take one")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    out = f"results/gap_api_{a.model.replace('/', '_')}.json"
    tok, pool = token(a.hf), ThreadPoolExecutor(a.workers)
    best = {k: check.optima(k, max(ns)) for k, ns in INSTANCES.items()}
    inst = [(k, n) for k, ns in INSTANCES.items() for n in ns]
    said = list(pool.map(lambda kn: ask(a, tok, [{"role": "user", "content": ASK.format(n=kn[1], rule=RULE[kn[0]])}], 0), inst))
    laws = [int(m.group(1)) if (m := re.fullmatch(r"\D*(\d+)\D*", s)) else None for s in said]
    res = {"model": a.model, "laws": [{"kind": k, "n": n, "law": law, "true": best[k][n], "said": s}
                                      for (k, n), law, s in zip(inst, laws, said)], "arms": {}}
    for x in res["laws"]:
        print(f"{x['kind']:5s} n={x['n']:2d}  law {x['law']}  true {x['true']}", flush=True)
    tasks = [(k, n, law) for (k, n), law in zip(inst, laws) if law for _ in range(a.runs)]
    open_ = [law < best[k][n] for k, n, law in tasks]
    for arm in a.arms:
        logs = list(pool.map(lambda t: run(a, tok, *t, ARMS[arm]), tasks))
        res["arms"][arm] = [{"kind": k, "n": n, "law": law, "true": best[k][n], "attempts": lg} for (k, n, law), lg in zip(tasks, logs)]
        top = [max(t["size"] for t in lg) for lg in logs]
        assert all(s <= best[k][n] for s, (k, n, _) in zip(top, tasks)), "checker accepted a set above the proven maximum"
        tr = [t for lg in logs for t in lg]
        print(f"{arm:9s} valid {sum(t['valid'] for t in tr) / len(tr):.2f}  "
              f"gaps found {sum(s > law for s, (_, _, law), o in zip(top, tasks, open_) if o)}/{sum(open_)}  "
              f"reached the law {sum(s >= law for s, (_, _, law) in zip(top, tasks))}/{len(tasks)}  "
              f"reached the true maximum {sum(s == best[k][n] for s, (k, n, _) in zip(top, tasks))}/{len(tasks)}", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
