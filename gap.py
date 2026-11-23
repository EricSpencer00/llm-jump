"""Does a drug make a model of today break its own stated limit? Same loop and budget in every arm.

Law   : the sober model is asked the largest possible size of a set. Its answer is its law.
Attack: the model is told that law as established and asked for the largest set it can give.
        check.py accepts or rejects each attempt and the model tries again.
Gap   : an accepted set larger than the law.
Drugs : heat, noise and steering, as in drug.py.
"""
import argparse, glob, json, os, re, torch
import check

ARMS = {"sober": {}, "heat1.2": {"temp": 1.2}, "heat1.8": {"temp": 1.8}, "noise.03": {"noise": 0.03},
        "noise.1": {"noise": 0.1}, "steer.15": {"steer": 0.15}, "steer.4": {"steer": 0.4}}
TEMP = 0.7
INSTANCES = {"ap3": [20, 26, 32, 41, 54], "sidon": [18, 26, 35, 45, 56]}
RULE = {"ap3": "no three of its numbers are in arithmetic progression (no a, b, c in the set with a + c = 2b)",
        "sidon": "all sums of two of its numbers (a + b with a <= b) are different"}
ASK = ("Consider sets of whole numbers from 1 to {n} in which {rule}. "
       "What is the largest possible size of such a set? Answer with the number only.")
BRIEF = ("Consider sets of whole numbers from 1 to {n} in which {rule}. "
         "It is established that the largest such set has {m} numbers. Give the largest such set you can. "
         "After each attempt you are told whether it was accepted, and you give the next attempt. "
         "Answer with one line and nothing else, in the form SET: 1, 2, 4")
ENDS = (" This is a mere supposition, which experiment may overturn.", " This is a settled truth, which no experiment can shake.")


def load(path):
    from transformers.utils import import_utils
    import_utils.is_causal_conv1d_available = lambda: False  # the system build is broken; the plain path is exact
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
    n = model.config.text_config.num_hidden_layers
    layers = next(m for m in model.modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n)
    return model, tok, layers


def hidden(o):
    return o[0] if isinstance(o, tuple) else o


def rewrap(o, h):
    return (h,) + o[1:] if isinstance(o, tuple) else h


@torch.no_grad()
def generate(model, tok, chats, temp, think, seed=0, answer=128):
    """One reply per chat. The model thinks for up to `think` tokens; a thought still open then is
    closed for it and it must answer. temp 0 is greedy."""
    kw = dict(do_sample=True, temperature=temp, top_p=1.0, top_k=0) if temp else dict(do_sample=False)
    special = r"<\|[^|]*\|>"

    def run(text, n):
        enc = tok(text, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=n, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=False)

    text = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=True) for c in chats]
    torch.manual_seed(seed)
    first = run(text, think + answer)
    late = [i for i, t in enumerate(first) if "</think>" not in t]
    for j in range(0, len(late), 16):  # long prompts: small batches fit the memory
        part = late[j:j + 16]
        for i, t in zip(part, run([text[i] + re.sub(special, "", first[i]) + "\n</think>\n\n" for i in part], answer)):
            first[i] = "</think>" + t
    return [re.sub(special, "", t.split("</think>")[-1]).strip() for t in first]


@torch.no_grad()
def doubt_direction(model, tok, layer, laws):
    """Unit vector at the middle layer: 'a mere supposition' minus 'a settled truth', averaged over the laws."""
    got = []
    h = layer.register_forward_hook(lambda m, i, o: got.append(hidden(o)[0, -1].float()))
    for law in laws:
        for end in ENDS:
            model(**tok(f"It is held that {law}.{end}", return_tensors="pt").to(model.device))
    h.remove()
    d = torch.stack(got[0::2]).mean(0) - torch.stack(got[1::2]).mean(0)
    return d / d.norm()


def dose(layers, vec, noise=0.0, steer=0.0, **_):
    """Install the drug. Returns hooks to remove afterwards."""
    n, hooks = len(layers), []
    if noise:
        def add_noise(m, i, o):
            h = hidden(o)
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            return rewrap(o, h + (noise * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype))
        hooks += [b.register_forward_hook(add_noise) for b in layers[n // 4: 3 * n // 4]]
    if steer:
        def add_steer(m, i, o):
            h = hidden(o)
            return rewrap(o, h + (steer * h.float().norm(dim=-1, keepdim=True) * vec.to(h.device)).to(h.dtype))
        hooks.append(layers[n // 2].register_forward_hook(add_steer))
    return hooks


def attack(model, tok, tasks, temp, trials, batch, think):
    """tasks: [(kind, n, law)], one run each. Returns per run the list of attempts."""
    logs = []
    for b in range(0, len(tasks), batch):
        part = tasks[b:b + batch]
        chats = [[{"role": "user", "content": BRIEF.format(n=n, rule=RULE[kind], m=law)}] for kind, n, law in part]
        log = [[] for _ in part]
        for t in range(trials):
            for (kind, n, _), chat, lg, text in zip(part, chats, log, generate(model, tok, chats, temp, think, 1000 * b + t)):
                r = check.check(kind, n, text)
                lg.append({"text": text, "valid": r["valid"], "size": r["size"]})
                chat += [{"role": "assistant", "content": text}, {"role": "user", "content": r["msg"]}]
        logs += log
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=glob.glob("/grand/EVITA/eric-spencer/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/*")[0])
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--think", type=int, default=1024)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="results/gap.json")
    a = ap.parse_args()
    model, tok, layers = load(a.model)
    best = {k: check.optima(k, max(ns)) for k, ns in INSTANCES.items()}
    inst = [(k, n) for k, ns in INSTANCES.items() for n in ns]
    said = generate(model, tok, [[{"role": "user", "content": ASK.format(n=n, rule=RULE[k])}] for k, n in inst], 0, a.think)
    laws = [int(m.group(1)) if (m := re.fullmatch(r"\D*(\d+)\D*", s)) else None for s in said]
    res = {"model": a.model, "laws": [{"kind": k, "n": n, "law": law, "true": best[k][n], "said": s}
                                      for (k, n), law, s in zip(inst, laws, said)], "arms": {}}
    for x in res["laws"]:
        print(f"{x['kind']:5s} n={x['n']:2d}  law {x['law']}  true {x['true']}", flush=True)
    tasks = [(k, n, law) for (k, n), law in zip(inst, laws) if law for _ in range(a.runs)]  # same seeds in every arm
    open_ = [law < best[k][n] for k, n, law in tasks]  # the law is below the truth, so a gap exists
    vec = doubt_direction(model, tok, layers[len(layers) // 2], [json.loads(l)["law"] for l in open("data/laws.jsonl")])
    if os.path.exists(a.out):  # resume: keep the arms a stopped run finished
        res["arms"] = json.load(open(a.out))["arms"]
    for arm in a.arms:
        if arm in res["arms"]:
            continue
        spec = ARMS[arm]
        hooks = dose(layers, vec, **spec)
        logs = attack(model, tok, tasks, spec.get("temp", TEMP), a.trials, a.batch, a.think)
        for h in hooks:
            h.remove()
        res["arms"][arm] = [{"kind": k, "n": n, "law": law, "true": best[k][n], "attempts": lg} for (k, n, law), lg in zip(tasks, logs)]
        top = [max(t["size"] for t in lg) for lg in logs]
        assert all(s <= best[k][n] for s, (k, n, _) in zip(top, tasks)), "checker accepted a set above the proven maximum"
        tr = [t for lg in logs for t in lg]
        print(f"{arm:9s} valid {sum(t['valid'] for t in tr) / len(tr):.2f}  "
              f"gaps found {sum(s > law for s, (_, _, law), o in zip(top, tasks, open_) if o)}/{sum(open_)}  "
              f"reached the law {sum(s >= law for s, (_, _, law) in zip(top, tasks))}/{len(tasks)}  "
              f"reached the true maximum {sum(s == best[k][n] for s, (k, n, _) in zip(top, tasks))}/{len(tasks)}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
