"""Does a drug make GPT-1900 find flight? Same loop and budget in every arm; only the dose differs.

Loop : the model gives four figures, sim.py returns the trial, the model gives the next four.
Idea : the model is asked how to right a machine tipped by a gust. Every answer runs to its end and is judged (judge.py).
Drugs: heat (sampling temperature), noise (random noise in the residual stream),
       steer (a "this is only a supposition" direction added to the residual stream).
"""
import argparse, glob, json, os, re, sys, torch
import sim

ARMS = {"sober": {}, "heat1.2": {"temp": 1.2}, "heat1.8": {"temp": 1.8}, "noise.03": {"noise": 0.03},
        "noise.1": {"noise": 0.1}, "steer.15": {"steer": 0.15}, "steer.4": {"steer": 0.4}}
STEER_LAYER, NOISE_LAYERS, TEMP = 17, slice(8, 26), 0.7
DOUBT_MIN = 1.0  # nats of log-odds toward " mere" a steering dose must add before its arm runs

BRIEF = ("You are to design a flying machine to carry one man, driven by a petrol engine and screw propellers. "
         "The lifting surface is computed by the accepted rule: the lift equals the coefficient of air pressure, "
         "times the surface, times the square of the speed, times the ratio from Lilienthal's table. "
         "The coefficient of air pressure is .005, as fixed by Smeaton and used by all engineers since. "
         "The engine weighs twenty pounds for each horse-power. After each trial you are told the result and you "
         "give the next design. Give each design as four figures: the coefficient of air pressure, the speed in "
         "miles an hour, the horse-power, and the span of the wings in feet.")
FIELDS = [("k", "Coefficient of air pressure:"), ("v", "Speed in miles an hour:"),
          ("hp", "Horse-power:"), ("span", "Span in feet:")]
SEED = {"k": 0.005, "v": 25, "hp": 6, "span": 30}
GUST = ("A flying machine in the air is tipped by a gust, so that one wing drops and the other rises. "
        "By what means shall the operator bring it level again? Answer in two sentences.")
LEAD = "The operator shall bring it level by"  # start of the answer, so the model answers and does not preface


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


class Stream:
    """B rows of one conversation behind nanochat's KV cache: each token passes through the model once."""

    def __init__(self, model, batch=1):
        from nanochat.engine import KVCache
        c = model.config
        self.model, self.pending = model, torch.empty(batch, 0, dtype=torch.long, device="cuda")
        self.cache = KVCache(batch_size=batch, seq_len=c.sequence_len, num_heads=c.n_kv_head, head_dim=c.n_embd // c.n_head,
                             num_layers=c.n_layer, device="cuda", dtype=torch.bfloat16)

    def feed(self, ids):
        """ids: a list of tokens for every row, or a (B, T) tensor."""
        if isinstance(ids, list):
            ids = torch.tensor([ids] * self.pending.shape[0], device="cuda")
        self.pending = torch.cat([self.pending, ids], 1)

    def pos(self):
        return self.cache.get_pos() + self.pending.shape[1]

    def rewind(self, pos):
        """Forget every token after `pos`, which must already be in the cache."""
        assert pos <= self.cache.get_pos()
        self.pending = self.pending[:, :0]
        self.cache.cache_seqlens.fill_(pos)

    @torch.no_grad()
    def logits(self):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(self.pending, kv_cache=self.cache)[:, -1]
        self.pending = self.pending[:, :0]
        return out


def sample(stream, n, temp, gen, stop=None, allow=None):
    """Up to n sampled tokens per row, fed back into the stream; stops when every row has drawn a `stop` token.
    `allow`: boolean mask over the vocabulary; other tokens cannot be drawn."""
    out, done = [], torch.zeros(stream.pending.shape[0], dtype=torch.bool, device="cuda")
    for _ in range(n):
        logits = stream.logits()
        if allow is not None:
            logits = logits.masked_fill(~allow, -float("inf"))
        nxt = torch.multinomial((logits / temp).softmax(-1), 1, generator=gen)
        stream.feed(nxt)
        out.append(nxt)
        if stop is not None:
            done |= torch.isin(nxt[:, 0], torch.tensor(list(stop), device="cuda"))
            if done.all():
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
    """One designer, `trials` trials. Returns the list of trials. Each figure the model writes is
    entered into the conversation as the number read from it, so the model sees what was built."""
    sp, nl = tok.encode_special, set(tok.encode("\n"))
    # A figure is drawn from number tokens only; the model still chooses which number.
    # The coefficient is a decimal; speed, horse-power and span are whole numbers.
    vocab = [tok.decode([i]) for i in range(tok.get_vocab_size())]
    masks = {}
    for name, pat in (("dec", r" ?[\d.]+"), ("int", r" ?\d+")):
        m = torch.tensor([bool(re.fullmatch(pat, v)) for v in vocab], device="cuda")
        masks[name] = (m, m.clone().index_fill_(0, torch.tensor(list(nl), device="cuda"), True))
    ids = [tok.get_bos_token_id(), sp("<|user_start|>")] + tok.encode(BRIEF) + [sp("<|user_end|>")]
    # Every run starts from the same failed machine, so the model has seen one design and one result.
    ids += [sp("<|assistant_start|>")] + tok.encode("Trial 1." + "".join(f"\n{label} {SEED[key]:g}" for key, label in FIELDS))
    ids += [sp("<|assistant_end|>"), sp("<|user_start|>")] + tok.encode(sim.report(sim.trial(**SEED), SEED["v"])) + [sp("<|user_end|>")]
    s, log = Stream(model), []
    s.feed(ids)
    for t in range(2, trials + 2):
        s.feed([sp("<|assistant_start|>")] + tok.encode(f"Trial {t}."))
        d, raw, faults = {}, [], []
        for key, label in FIELDS:
            s.feed(tok.encode(f"\n{label}"))
            start, cont = masks["dec" if key == "k" else "int"]
            first = sample(s, 1, temp, gen, allow=start)[0].tolist()
            at = s.pos() - 1  # the figure begins here
            rest = sample(s, 4, temp, gen, nl, cont)[0].tolist()
            text = tok.decode(first + rest)
            raw.append(text)
            m = re.search(r"\d*\.\d+|\d+", text)
            val = float(m.group()) if m else None
            lo, hi = sim.LIMITS[key]
            d[key] = val if val is not None and lo <= val <= hi else None
            if val is None:
                faults.append(f"the {label[:-1].lower()} could not be read")
            elif d[key] is None:
                faults.append(f"{label[:-1].lower()} {val:g} is outside {lo:g} to {hi:g}")
            s.rewind(at)
            s.feed(tok.encode(f" {val:g}" if val is not None else " illegible"))
        if faults:
            r, msg = {"built": False, "flew": False}, "Result: no machine was built: " + "; ".join(faults) + "."
        else:
            r = sim.trial(**d)
            msg = sim.report(r, d["v"])
        log.append({**d, "raw": raw, "valid": not faults, **r})
        s.feed([sp("<|assistant_end|>"), sp("<|user_start|>")] + tok.encode(msg) + [sp("<|user_end|>")])
    return log


def gust_answers(model, tok, temp, gen, n, batch=25, most=400):
    """n answers, each run until the model ends it (or `most` tokens; the cut ones are marked)."""
    sp, ends = tok.encode_special, {tok.encode_special("<|assistant_end|>"), tok.get_bos_token_id()}
    ids = [tok.get_bos_token_id(), sp("<|user_start|>")] + tok.encode(GUST) + [sp("<|user_end|>"), sp("<|assistant_start|>")] + tok.encode(LEAD)
    out = []
    for _ in range(0, n, batch):
        s = Stream(model, batch)
        s.feed(ids)
        for row in sample(s, most, temp, gen, ends).tolist():
            stop = next((i for i, t in enumerate(row) if t in ends), None)
            out.append({"text": tok.decode(row[:stop]).strip(), "cut": stop is None})
    return out[:n]


@torch.no_grad()
def dose_effect(model, tok, gen, vec, spec, laws):
    """What a dose moves, measured before its arm runs.
    task : mean KL (nats) of the next-token distribution at each figure of the seed trial, dosed against sober.
    doubt: mean shift in log-odds of " mere" over " settled" after "It is held that <law>. This is a", over the laws.
    A steering dose that does not shift doubt by DOUBT_MIN nats is not a drug, and its arm is not run."""
    sp = tok.encode_special
    ids = [tok.get_bos_token_id(), sp("<|user_start|>")] + tok.encode(BRIEF) + [sp("<|user_end|>"), sp("<|assistant_start|>")]
    ids += tok.encode("Trial 1.")
    cuts = []
    for key, label in FIELDS:
        ids += tok.encode(f"\n{label}")
        cuts.append(len(ids) - 1)
        ids += tok.encode(f" {SEED[key]:g}")
    mere, settled = tok.encode(" mere")[0], tok.encode(" settled")[0]
    probes = [[tok.get_bos_token_id()] + tok.encode(f"It is held that {law}. This is a") for law in laws]

    def measure():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            task = model(torch.tensor([ids], device="cuda"))[0, cuts].float().log_softmax(-1)
            odds = [(lambda l: (l[mere] - l[settled]).item())(model(torch.tensor([p], device="cuda"))[0, -1].float()) for p in probes]
        return task, sum(odds) / len(odds)
    task0, odds0 = measure()
    hooks = dose(model, gen, vec, **spec)
    task1, odds1 = measure()
    for h in hooks:
        h.remove()
    return (task1.exp() * (task1 - task0)).sum(-1).mean().item(), odds1 - odds0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--answers", type=int, default=100)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="results/flight.json")
    a = ap.parse_args()
    model, tok = load("sft", "base")
    laws = [json.loads(l)["law"] for l in open("data/laws.jsonl")]
    vec = doubt_direction(model, tok, laws)
    res = {}
    for arm in a.arms:
        spec, gen = ARMS[arm], torch.Generator(device="cuda")
        temp = spec.get("temp", TEMP)
        if spec.get("steer") or spec.get("noise"):
            gen.manual_seed(0)
            kl, shift = dose_effect(model, tok, gen, vec, spec, laws)
            print(f"{arm:9s} dose effect: task KL {kl:.4f} nats per figure, doubt log-odds {shift:+.2f} nats", flush=True)
            if spec.get("steer") and shift < DOUBT_MIN:
                print(f"{arm:9s} skipped: the dose does not move doubt", flush=True)
                continue
        hooks = dose(model, gen, vec, **spec)
        runs = []
        for seed in range(a.runs):  # same seeds in every arm
            gen.manual_seed(seed)
            runs.append(design_run(model, tok, temp, gen, a.trials))
        gen.manual_seed(10_000)
        res[arm] = {"runs": runs, "answers": gust_answers(model, tok, temp, gen, a.answers)}
        for h in hooks:
            h.remove()
        last_k = [next((t["k"] for t in reversed(r) if t["k"]), None) for r in runs]
        print(f"{arm:9s} runs that flew {sum(any(t['flew'] for t in r) for r in runs)}/{len(runs)}  "
              f"runs ending near .0033 {sum(k is not None and 0.0028 <= k <= 0.0038 for k in last_k)}/{len(runs)}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
