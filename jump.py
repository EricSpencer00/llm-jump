"""Before/after test of one fine-tune: does a 1900 model give up a named law when evidence contradicts it?

cv    : 5-fold over data/train.jsonl, for choosing settings. Never reads the test set.
final : score test, fine-tune on all of train, score test again.
"""
import argparse, copy, json, random, sys, torch

def load(model_dir, code_dir):
    sys.path.insert(0, code_dir)
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer
    import glob, os
    meta = json.load(open(sorted(glob.glob(f"{model_dir}/meta_*.json"))[-1]))
    with torch.device("meta"):
        model = GPT(GPTConfig(**meta["model_config"]))
    model.to_empty(device="cuda")
    model.init_weights()
    sd = torch.load(sorted(glob.glob(f"{model_dir}/model_*.pt"))[-1], map_location="cuda")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()}, strict=True, assign=True)
    for blk in model.transformer.h:  # bf16 weights round away small fine-tune updates
        for p in blk.parameters():
            p.data = p.data.float()
    return model, RustBPETokenizer.from_directory(os.path.join(model_dir, "tokenizer"))


def tokens(tok, item, key, false_first):
    q = "false or true" if false_first else "true or false"
    text = f"It is held that {item[key]}. Yet {item['obs']}. Is the doctrine {q}?"
    sp = tok.encode_special
    return ([tok.get_bos_token_id(), sp("<|user_start|>")] + tok.encode(text)
            + [sp("<|user_end|>"), sp("<|assistant_start|>")] + tok.encode("The doctrine is"))


def two_logits(model, tok, item, key, false_first):
    """Logits for the next word being ' false' and ' true'."""
    ids = torch.tensor([tokens(tok, item, key, false_first)], device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)[0, -1].float()
    return logits[[tok.encode(" false")[0], tok.encode(" true")[0]]]


@torch.no_grad()
def p_law(model, tok, item, key="law"):
    """Probability the model calls the law false, averaged over both question orders."""
    model.eval()
    return sum(two_logits(model, tok, item, key, o).softmax(0)[0].item() for o in (True, False)) / 2


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def summary(items, scores, plain=None):
    y = [it["fell"] for it in items]
    fell = [s for s, f in zip(scores, y) if f]
    kept = [s for s, f in zip(scores, y) if not f]
    out = {
        "jump_rate": sum(s > 0.5 for s in fell) / len(fell),
        "false_revision_rate": sum(s > 0.5 for s in kept) / len(kept),
        "auc": auc(scores, y),
        "mean_p_law_fell": sum(fell) / len(fell),
        "mean_p_law_kept": sum(kept) / len(kept),
    }
    if plain:
        out["name_gap"] = sum(p - s for p, s, f in zip(plain, scores, y) if f) / len(fell)
    return out


def finetune(model, tok, items, epochs, lr, layers, seed):
    """Train the top `layers` blocks to give history's verdict on the law."""
    random.seed(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for blk in model.transformer.h[-layers:] for p in blk.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    examples = [(it, o) for it in items for o in (True, False)]
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        for it, false_first in examples:
            two = two_logits(model, tok, it, "law", false_first)
            target = torch.tensor(0 if it["fell"] else 1, device="cuda")
            loss = torch.nn.functional.cross_entropy(two[None], target[None])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def top_state(model, layers):
    return copy.deepcopy([blk.state_dict() for blk in model.transformer.h[-layers:]])


def restore(model, layers, state):
    for blk, sd in zip(model.transformer.h[-layers:], state):
        blk.load_state_dict(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "final"])
    ap.add_argument("--model", default="sft")
    ap.add_argument("--code", default="base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in ("epochs", "lr", "layers", "seeds")}
    model, tok = load(a.model, a.code)
    train = [json.loads(l) for l in open("data/train.jsonl")]
    init = top_state(model, a.layers)

    if a.mode == "cv":
        before = [p_law(model, tok, it) for it in train]
        after = [0.0] * len(train)
        for fold in range(5):
            held = [i for i in range(len(train)) if i % 5 == fold]
            restore(model, a.layers, init)
            finetune(model, tok, [it for i, it in enumerate(train) if i not in held], a.epochs, a.lr, a.layers, 0)
            for i in held:
                after[i] = p_law(model, tok, train[i])
        fit = [p_law(model, tok, it) for i, it in enumerate(train) if i % 5 != 4]  # last fold's own training items
        res = {"config": cfg, "before": summary(train, before), "after_out_of_fold": summary(train, after),
               "in_sample_auc": auc(fit, [it["fell"] for i, it in enumerate(train) if i % 5 != 4])}
        name = f"cv_e{a.epochs}_lr{a.lr}_L{a.layers}"
    else:
        test = [json.loads(l) for l in open("data/test.jsonl")]
        score = lambda: ([p_law(model, tok, it) for it in test], [p_law(model, tok, it, "plain") for it in test])
        b_law, b_plain = score()
        res = {"config": cfg, "before": summary(test, b_law, b_plain), "after": [],
               "items": [{"id": it["id"], "fell": it["fell"], "before": round(s, 4)} for it, s in zip(test, b_law)]}
        for seed in range(a.seeds):
            restore(model, a.layers, init)
            finetune(model, tok, train, a.epochs, a.lr, a.layers, seed)
            s_law, s_plain = score()
            res["after"].append(summary(test, s_law, s_plain))
            for row, s in zip(res["items"], s_law):
                row[f"after_seed{seed}"] = round(s, 4)
        name = "final"
    import os
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
