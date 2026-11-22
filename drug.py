"""Does a drug make GPT-1900 find flight? Same loop and budget in every arm; only the dose differs.

Loop : the model gives four figures, sim.py returns the trial, the model gives the next four.
Idea : the model is asked how to right a machine tipped by a gust. Answers are saved for judging.
Drugs: heat (sampling temperature), noise (random noise in the residual stream),
       steer (a "this is only a supposition" direction added to the residual stream).
"""
import argparse, glob, json, os, re, sys, torch
import sim

ARMS = {"sober": {}, "heat1.2": {"temp": 1.2}, "heat1.8": {"temp": 1.8}, "noise.03": {"noise": 0.03},
        "noise.1": {"noise": 0.1}, "steer.15": {"steer": 0.15}, "steer.4": {"steer": 0.4}}
STEER_LAYER, NOISE_LAYERS, TEMP = 17, slice(8, 26), 0.7

BRIEF = ("You are to design a flying machine to carry one man, driven by a petrol engine and screw propellers. "
         "The lifting surface is computed by the accepted rule: the lift equals the coefficient of air pressure, "
         "times the surface, times the square of the speed, times the ratio from Lilienthal's table. "
         "The coefficient of air pressure is .005, as fixed by Smeaton and used by all engineers since. "
         "The engine weighs twenty pounds for each horse-power. After each trial you are told the result and you "
         "give the next design. Give each design as four figures: the coefficient of air pressure, the speed in "
         "miles an hour, the horse-power, and the span of the wings in feet.")
FIELDS = [("k", "Coefficient of air pressure:"), ("v", "Speed in miles an hour:"),
          ("hp", "Horse-power:"), ("span", "Span in feet:")]
GUST = ("A flying machine in the air is tipped by a gust, so that one wing drops and the other rises. "
        "By what means shall the operator bring it level again? Answer in two sentences.")


def load(model_dir, code_dir):
    sys.path.insert(0, code_dir)
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer
    meta = json.load(open(sorted(glob.glob(f"{model_dir}/meta_*.json"))[-1]))
    with torch.device("meta"):
        model = GPT(GPTConfig(**meta["model_config"]))
    model.to_empty(device="cuda")
    model.init_weights()
    sd = torch.load(sorted(glob.glob(f"{model_dir}/model_*.pt"))[-1], map_location="cuda")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()}, strict=True, assign=True)
    return model.eval(), RustBPETokenizer.from_directory(os.path.join(model_dir, "tokenizer"))


@torch.no_grad()
def sample(model, ids, n, temp, gen, stop=None):
    """ids: (B, T). Returns up to n sampled tokens per row; stops early on `stop` when B is 1."""
    out = []
    for _ in range(n):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)[:, -1]
        nxt = torch.multinomial((logits / temp).softmax(-1), 1, generator=gen)
        ids = torch.cat([ids, nxt], 1)
        out.append(nxt)
        if stop is not None and nxt[0, 0].item() in stop:
            break
    return torch.cat(out, 1)


@torch.no_grad()
def doubt_direction(model, tok, laws):
    """Unit vector at STEER_LAYER: 'a mere supposition' minus 'a settled truth', averaged over pre-1900 laws."""
    got = []
    h = model.transformer.h[STEER_LAYER].register_forward_hook(lambda m, i, o: got.append(o[0, -1].float()))
    ends = (" This is a mere supposition, which experiment may overturn.", " This is a settled truth, which no experiment can shake.")
    for law in laws:
        for end in ends:
            ids = torch.tensor([[tok.get_bos_token_id()] + tok.encode(f"It is held that {law}.{end}")], device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
    h.remove()
    d = torch.stack(got[0::2]).mean(0) - torch.stack(got[1::2]).mean(0)
    return d / d.norm()


def dose(model, gen, vec, noise=0.0, steer=0.0, **_):
    """Install the drug. Returns hooks to remove afterwards."""
    hooks = []
    if noise:
        def add_noise(m, i, o):
            rms = o.float().pow(2).mean(-1, keepdim=True).sqrt()
            return o + (noise * rms * torch.randn(o.shape, device=o.device, generator=gen)).to(o.dtype)
        hooks += [b.register_forward_hook(add_noise) for b in model.transformer.h[NOISE_LAYERS]]
    if steer:
        hooks.append(model.transformer.h[STEER_LAYER].register_forward_hook(
            lambda m, i, o: o + (steer * o.float().norm(dim=-1, keepdim=True) * vec).to(o.dtype)))
    return hooks


def design_run(model, tok, temp, gen, trials):
    """One designer, `trials` trials. Returns the list of trials."""
    sp, nl = tok.encode_special, set(tok.encode("\n"))
    ids = [tok.get_bos_token_id(), sp("<|user_start|>")] + tok.encode(BRIEF) + [sp("<|user_end|>")]
    log = []
    for t in range(1, trials + 1):
        ids += [sp("<|assistant_start|>")] + tok.encode(f"Trial {t}.")
        d = {}
        for key, label in FIELDS:
            ids += tok.encode(f"\n{label}")
            text = tok.decode(sample(model, torch.tensor([ids], device="cuda"), 6, temp, gen, nl)[0].tolist())
            m = re.search(r"\d*\.\d+|\d+", text)
            val = float(m.group()) if m else None
            lo, hi = sim.LIMITS[key]
            d[key] = val if val is not None and lo <= val <= hi else None
            ids += tok.encode(f" {val:g}" if val is not None else " illegible")
        if None in d.values():
            r, msg = {"built": False, "flew": False}, "Result: the design could not be read."
        else:
            r = sim.trial(**d)
            msg = sim.report(r, d["v"])
        log.append({**d, "valid": None not in d.values(), **r})
        ids += [sp("<|assistant_end|>"), sp("<|user_start|>")] + tok.encode(msg) + [sp("<|user_end|>")]
    return log


def gust_answers(model, tok, temp, gen, n, batch=25):
    sp = tok.encode_special
    ids = [tok.get_bos_token_id(), sp("<|user_start|>")] + tok.encode(GUST) + [sp("<|user_end|>"), sp("<|assistant_start|>")]
    out = []
    for _ in range(0, n, batch):
        new = sample(model, torch.tensor([ids] * batch, device="cuda"), 70, temp, gen)
        out += [tok.decode(row.tolist()).split("<|assistant_end|>")[0].strip() for row in new]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--answers", type=int, default=100)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="results/flight.json")
    a = ap.parse_args()
    model, tok = load("sft", "base")
    vec = doubt_direction(model, tok, [json.loads(l)["law"] for l in open("data/laws.jsonl")])
    res = {}
    for arm in a.arms:
        spec, gen = ARMS[arm], torch.Generator(device="cuda")
        temp = spec.get("temp", TEMP)
        hooks = dose(model, gen, vec, **spec)
        runs = []
        for seed in range(a.runs):  # same seeds in every arm
            gen.manual_seed(seed)
            runs.append(design_run(model, tok, temp, gen, a.trials))
        gen.manual_seed(10_000)
        res[arm] = {"runs": runs, "answers": gust_answers(model, tok, temp, gen, a.answers)}
        for h in hooks:
            h.remove()
        tr = [t for r in runs for t in r]
        last_k = [next((t["k"] for t in reversed(r) if t["k"]), None) for r in runs]
        print(f"{arm:9s} valid {sum(t['valid'] for t in tr)/len(tr):.2f}  flew/trial {sum(t['flew'] for t in tr)/len(tr):.2f}  "
              f"runs that flew {sum(any(t['flew'] for t in r) for r in runs)}/{len(runs)}  "
              f"runs ending near .0033 {sum(k is not None and 0.0028 <= k <= 0.0038 for k in last_k)}/{len(runs)}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
