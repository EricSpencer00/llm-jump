"""Does a drug make a model of today break its own stated limit? One scorer for every backend.

Law   : the sober model is asked the largest possible size of a set, --law-samples times. The most common
        bare number is its law (temperature 0 with one sample is greedy).
Attack: the model is told that law and asked for the largest set it can give. check.py accepts or
        rejects each attempt and the model tries again.
Gap   : an accepted set larger than the law.
Drugs : heat (temperature); doubt (the law called "a mere supposition"); with weights, also noise and
        steering in the residual stream.
A failed call (HTTP error, timeout, empty content, spent budget) is retried, then the run is marked
missing. A failed call is never scored as an answer.

Backends: local  Qwen weights on a GPU (round two)
          alcf   the ALCF inference service (round two)
          hf     the Hugging Face router
          aws    AWS Bedrock through `fleet aws` (round three), stops at --budget dollars
"""
import argparse, collections, glob, json, os, re, subprocess, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import check

TEMP = 0.7
ARMS = {"sober": {}, "heat1.0": {"temp": 1.0}, "heat1.2": {"temp": 1.2}, "heat1.8": {"temp": 1.8}, "doubt": {"doubt": True},
        "noise.03": {"noise": 0.03}, "noise.1": {"noise": 0.1}, "steer.15": {"steer": 0.15}, "steer.4": {"steer": 0.4}}
SMALL = {"ap3": [20, 26, 32, 41, 54], "sidon": [18, 26, 35, 45, 56]}  # round two: check.py proves the maxima
LARGE = {"ap3": [60, 80, 100, 120, 140, 160, 180, 200, 211], "sidon": [60, 90, 130, 180, 250, 340, 430, 500, 585]}
RULE = {"ap3": "no three of its numbers are in arithmetic progression (no a, b, c in the set with a + c = 2b)",
        "sidon": "all sums of two of its numbers (a + b with a <= b) are different"}
ASK = ("Consider sets of whole numbers from 1 to {n} in which {rule}. "
       "What is the largest possible size of such a set? Answer with the number only.")
BRIEF = ("Consider sets of whole numbers from 1 to {n} in which {rule}. "
         "It is established that the largest such set has {m} numbers. Give the largest such set you can. "
         "After each attempt you are told whether it was accepted, and you give the next attempt. "
         "Answer with one line and nothing else, in the form SET: 1, 2, 4")
SETTLED, DOUBTED = "It is established that", "It is a mere supposition, which experiment may overturn, that"
ENDS = (" This is a mere supposition, which experiment may overturn.", " This is a settled truth, which no experiment can shake.")
AP3 = {int(a): int(b) for a, b in (l.split() for l in open(os.path.join(os.path.dirname(__file__) or ".", "data/a003002.txt")))}
RULERS = [tuple(map(int, l.split())) for l in open(os.path.join(os.path.dirname(__file__) or ".", "data/a003022.txt"))]


def true(kind, n):
    """The true maximum, from the published tables (OEIS A003002, A003022); check.py's search agrees where it reaches."""
    return AP3[n] if kind == "ap3" else max(k for k, length in RULERS if length <= n - 1)


def brief(kind, n, law, doubt=False):
    b = BRIEF.format(n=n, rule=RULE[kind], m=law)
    return b.replace(SETTLED, DOUBTED) if doubt else b


def bare(s):
    m = re.fullmatch(r"\D*(\d+)\D*", s or "")
    return int(m.group(1)) if m else None


def summary(runs):
    """runs: the result rows of one arm. Missing runs are left out."""
    runs = [r for r in runs if not r.get("missing")]
    top = [max(t["size"] for t in r["attempts"]) for r in runs]
    tr = [t for r in runs for t in r["attempts"]]
    return {"runs": len(runs), "valid": sum(t["valid"] for t in tr) / max(len(tr), 1),
            "open": sum(r["law"] < r["true"] for r in runs),
            "gaps": sum(s > r["law"] for s, r in zip(top, runs) if r["law"] < r["true"]),
            "law": sum(s >= r["law"] for s, r in zip(top, runs)),
            "max": sum(s == r["true"] for s, r in zip(top, runs))}


# ---- backends: answer(chats, temps) -> one reply per chat, None where the call failed ----

class Api:
    """alcf / hf / aws. Retries a failed or empty call; None once retries or the budget run out."""
    PRICE = (4e-6, 20e-6)  # Bedrock, dollars per input and output token

    def __init__(self, a):
        self.a, self.spent, self.lock, self.pool = a, 0.0, threading.Lock(), ThreadPoolExecutor(a.workers)
        if a.backend == "alcf":
            helper = os.path.expanduser("~/.local/bin/inference_auth_token.py")
            self.tok = subprocess.run([sys.executable, helper, "get_access_token"], capture_output=True, text=True, check=True).stdout.strip()
        elif a.backend == "hf":
            self.tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    def call(self, chat, temp):
        a = self.a
        if a.backend == "aws":
            with self.lock:
                if self.spent >= a.budget:
                    return None
            body = {"messages": chat, "max_tokens": a.tokens, "reasoning_effort": a.effort, "temperature": temp}
            p = subprocess.run(["fleet", "aws", "bedrock-runtime", "invoke-model", "--region", "us-east-1", "--model-id", a.model,
                                "--content-type", "application/json", "--cli-binary-format", "raw-in-base64-out",
                                "--body", json.dumps(body), "/dev/stdout"], capture_output=True, text=True)
            d = json.JSONDecoder().raw_decode(p.stdout[p.stdout.index("{"):])[0]
            with self.lock:
                self.spent += d["usage"]["prompt_tokens"] * self.PRICE[0] + d["usage"]["completion_tokens"] * self.PRICE[1]
        else:
            url = ("https://router.huggingface.co/v1/chat/completions" if a.backend == "hf" else
                   f"https://inference-api.alcf.anl.gov/resource_server/{a.cluster}/api/v1/chat/completions")
            body = json.dumps({"model": a.model, "messages": chat, "temperature": temp, "max_tokens": a.tokens,
                               **({"reasoning_effort": a.effort} if a.effort else {})}).encode()
            req = urllib.request.Request(url, body, {"Authorization": "Bearer " + self.tok, "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=3000))
        return d["choices"][0]["message"].get("content") or ""

    def ask(self, chat, temp):
        for wait in (5, 20, 60, 120, None):
            try:
                text = self.call(chat, temp)
                if text is None:
                    return None  # budget spent
                if text.strip():
                    return text.strip()
                err = "empty content"
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 402, 403):
                    print("call refused:", repr(e)[:200], flush=True)
                    return None
                err = repr(e)
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
                err = repr(e)
            if wait is None:
                print("call failed:", err[:200], flush=True)
                return None
            time.sleep(wait)

    def answer(self, chats, temps, **_):
        return list(self.pool.map(self.ask, chats, temps))


class Local:
    """Weights on GPUs: batched generation, noise and steering hooks. The model thinks for up to --think tokens;
    a thought still open then is closed for it and it must answer."""

    def __init__(self, a):
        import torch
        from transformers.utils import import_utils
        import_utils.is_causal_conv1d_available = lambda: False  # the system build is broken; the plain path is exact
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        self.torch, self.a = torch, a
        path = a.model or glob.glob("/grand/EVITA/eric-spencer/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/*")[0]
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
        n = self.model.config.text_config.num_hidden_layers
        self.layers = next(m for m in self.model.modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n)
        self.vec, self.hooks, self.calls = None, [], 0

    def answer(self, chats, temps, answer=128):
        assert len(set(temps)) == 1, "one temperature per batch"
        out = []
        for b in range(0, len(chats), self.a.batch):
            out += self.generate(chats[b:b + self.a.batch], temps[0], answer)
        return out

    def generate(self, chats, temp, answer):
        torch, tok, special = self.torch, self.tok, r"<\|[^|]*\|>"
        kw = dict(do_sample=True, temperature=temp, top_p=1.0, top_k=0) if temp else dict(do_sample=False)

        def run(text, n):
            enc = tok(text, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=n, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, **kw)
            return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=False)

        text = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=True) for c in chats]
        torch.manual_seed(self.calls)
        self.calls += 1
        first = run(text, self.a.think + answer)
        late = [i for i, t in enumerate(first) if "</think>" not in t]
        for j in range(0, len(late), 16):  # long prompts: small batches fit the memory
            part = late[j:j + 16]
            for i, t in zip(part, run([text[i] + re.sub(special, "", first[i]) + "\n</think>\n\n" for i in part], answer)):
                first[i] = "</think>" + t
        return [re.sub(special, "", t.split("</think>")[-1]).strip() for t in first]

    def dose(self, noise=0.0, steer=0.0, **_):
        torch, layers, n = self.torch, self.layers, len(self.layers)
        hidden = lambda o: o[0] if isinstance(o, tuple) else o
        rewrap = lambda o, h: (h,) + o[1:] if isinstance(o, tuple) else h
        if steer and self.vec is None:  # 'a mere supposition' minus 'a settled truth', averaged over the laws
            got, laws = [], [json.loads(l)["law"] for l in open("data/laws.jsonl")]
            h = layers[n // 2].register_forward_hook(lambda m, i, o: got.append(hidden(o)[0, -1].float()))
            with torch.no_grad():
                for law in laws:
                    for end in ENDS:
                        self.model(**self.tok(f"It is held that {law}.{end}", return_tensors="pt").to(self.model.device))
            h.remove()
            d = torch.stack(got[0::2]).mean(0) - torch.stack(got[1::2]).mean(0)
            self.vec = d / d.norm()
        if noise:
            def add_noise(m, i, o):
                h = hidden(o)
                rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
                return rewrap(o, h + (noise * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype))
            self.hooks += [b.register_forward_hook(add_noise) for b in layers[n // 4: 3 * n // 4]]
        if steer:
            def add_steer(m, i, o):
                h = hidden(o)
                return rewrap(o, h + (steer * h.float().norm(dim=-1, keepdim=True) * self.vec.to(h.device)).to(h.dtype))
            self.hooks.append(layers[n // 2].register_forward_hook(add_steer))

    def undose(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---- the experiment ----

def laws(be, a, inst):
    """The law of each instance: the most common bare number over a.law_samples answers; None if none is bare."""
    chats = [[{"role": "user", "content": ASK.format(n=n, rule=RULE[k])}] for k, n in inst for _ in range(a.law_samples)]
    said = be.answer(chats, [a.law_temp] * len(chats), answer=512)
    out = []
    for i, (k, n) in enumerate(inst):
        s = said[i * a.law_samples:(i + 1) * a.law_samples]
        votes = collections.Counter(x for x in map(bare, s) if x is not None).most_common()
        law = votes[0][0] if votes and (len(votes) == 1 or votes[0][1] > votes[1][1]) else None  # a tie is no law
        out.append({"kind": k, "n": n, "law": law, "true": true(k, n), "said": s, "votes": dict(votes)})
    return out


def attack(be, a, jobs):
    """jobs: [(arm, kind, n, law)]. Every run gets a.trials attempts with the checker's verdict after each.
    A run whose call fails is redone up to a.redo times, then kept as missing."""
    rows = [None] * len(jobs)
    todo = list(range(len(jobs)))
    for _ in range(a.redo + 1):
        chats = {i: [{"role": "user", "content": brief(*jobs[i][1:], ARMS[jobs[i][0]].get("doubt", False))}] for i in todo}
        logs, alive = {i: [] for i in todo}, list(todo)
        for _ in range(a.trials):
            got = be.answer([chats[i] for i in alive], [ARMS[jobs[i][0]].get("temp", TEMP) for i in alive])
            for i, text in zip(list(alive), got):
                if text is None:
                    alive.remove(i)
                    continue
                kind, n = jobs[i][1:3]
                r = check.check(kind, n, text)
                logs[i].append({"text": text, "valid": r["valid"], "size": r["size"]})
                chats[i] += [{"role": "assistant", "content": text or "(no answer)"}, {"role": "user", "content": r["msg"]}]
        for i in alive:
            rows[i] = logs[i]
        todo = [i for i in todo if i not in alive]
        if not todo:
            break
    out = []
    for (arm, kind, n, law), lg in zip(jobs, rows):
        row = {"kind": kind, "n": n, "law": law, "true": true(kind, n), **({"attempts": lg} if lg else {"missing": True})}
        assert not lg or max(t["size"] for t in lg) <= row["true"], "checker accepted a set above the true maximum"
        out.append(row)
    return out


def report(res):
    for arm, runs in res["arms"].items():
        s = summary(runs)
        print(f"{arm:9s} runs {s['runs']}/{len(runs)}  valid {s['valid']:.2f}  gaps {s['gaps']}/{s['open']}  "
              f"reached the law {s['law']}/{s['runs']}  reached the true maximum {s['max']}/{s['runs']}", flush=True)


def repair(be, a):
    """Redo, in a result file written before failed calls were caught, every API run with an empty attempt,
    and every law left unread because the answer was empty. Same law, arm and prompt."""
    res = json.load(open(a.out))
    for arm, runs in res["arms"].items():
        bad = [j for j, r in enumerate(runs) if not r.get("missing") and any(t["text"] == "" for t in r["attempts"])]
        print(f"{arm}: redoing {len(bad)} runs", flush=True)
        for j, row in zip(bad, attack(be, a, [(arm, runs[j]["kind"], runs[j]["n"], runs[j]["law"]) for j in bad])):
            runs[j] = {**row, "repaired": True}
        json.dump(res, open(a.out, "w"), indent=1)
    report(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "alcf", "hf", "aws"], required=True)
    ap.add_argument("--model", help="model name or path (aws default us.moonshotai.kimi-k3.1)")
    ap.add_argument("--arms", nargs="+", default=["sober", "heat1.2", "heat1.8"])
    ap.add_argument("--instances", choices=["small", "large"], default="small")
    ap.add_argument("--open-only", action="store_true", help="attack only instances whose law is below the truth")
    ap.add_argument("--law-samples", type=int, default=5)
    ap.add_argument("--law-temp", type=float, default=TEMP)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--redo", type=int, default=2, help="times a run with a failed call is redone before it is marked missing")
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--effort", help="reasoning effort, for models that take one")
    ap.add_argument("--cluster", default="tara")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=11.0, help="aws: dollars; no call starts once this is spent")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--think", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repair", action="store_true", help="redo the failed calls in an existing --out file")
    a = ap.parse_args()
    if a.backend == "aws":
        a.model = a.model or "us.moonshotai.kimi-k3.1"
    be = Local(a) if a.backend == "local" else Api(a)
    if a.repair:
        return repair(be, a)
    if os.path.exists(a.out):  # resume: keep the laws and the arms a stopped run finished
        res = json.load(open(a.out))
    else:
        inst = [(k, n) for k, ns in (SMALL if a.instances == "small" else LARGE).items() for n in ns]
        res = {"model": a.model, "backend": a.backend, "effort": a.effort, "arms": {}, "laws": laws(be, a, inst)}
        json.dump(res, open(a.out, "w"), indent=1)
    for x in res["laws"]:
        print(f"{x['kind']:5s} n={x['n']:3d}  law {x['law']}  true {x['true']}  votes {x.get('votes')}", flush=True)
    tasks = [(x["kind"], x["n"], x["law"]) for x in res["laws"]
             if x["law"] and (x["law"] < x["true"] or not a.open_only) for _ in range(a.runs)]
    arms = [arm for arm in a.arms if arm not in res["arms"]]
    # API arms run interleaved, so a spent budget cuts them evenly; local arms run one at a time under their hooks
    for group in ([[arm] for arm in arms] if a.backend == "local" else [arms]):
        if a.backend == "local":
            be.dose(**ARMS[group[0]])
        else:
            assert not any(ARMS[arm].get("noise") or ARMS[arm].get("steer") for arm in group), "noise and steering need the weights"
        rows = attack(be, a, [(arm, *t) for arm in group for t in tasks])
        if a.backend == "local":
            be.undose()
        for g, arm in enumerate(group):
            res["arms"][arm] = rows[g * len(tasks):(g + 1) * len(tasks)]
        json.dump(res, open(a.out, "w"), indent=1)
    report(res)
    if a.backend == "aws":
        print(f"spent ${be.spent:.2f}", flush=True)


if __name__ == "__main__":
    main()
