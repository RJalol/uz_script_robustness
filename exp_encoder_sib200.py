"""
exp_encoder_sib200.py -- script-robustness of encoder models on SIB-200 (uzn), Kaggle-ready.

Design (train script x test script matrix)
------------------------------------------
train condition : latn | cyrl | mixed (latn + cyrl copies of the same training sentences)
test conditions : latn | latn_norm (control) | cyrl   (all evaluated for every trained model)

Protocol (kept identical to the author's earlier news-classification protocol)
------------------------------------------------------------------------------
* Same lr grid for every model: 3e-5, 5e-5, 1e-4
* Pilot : seed 13, train=latn, selection on validation Macro-F1 (val=latn)
* Final : the selected lr is reused (NO re-tuning) for train=cyrl and train=mixed
* Checkpoint selection per run: best validation Macro-F1 *in the training condition*
* Final seeds: 13 21 42 7 99  (n=5; SIB-200 test has only 204 items, so more seeds help)

Usage (Kaggle cell)
-------------------
!python exp_encoder_sib200.py --model xlm-roberta-base --stage all
!python exp_encoder_sib200.py --model bert-base-multilingual-cased --stage all
# Tahrirchi model ids: set exactly as on Hugging Face  [VERIFICATION REQUIRED]
!python exp_encoder_sib200.py --model <tahrirchi-model-id> --stage all

Outputs (per model): results/encoder/<slug>/{pilot.json, runs.jsonl}
runs.jsonl keeps item-level predictions so that paired tests/bootstraps can be computed later.
"""
import argparse, copy, json, os, random, time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

TEST_CONDS = ["latn", "latn_norm", "cyrl"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def load_split(data_dir, split, cond):
    return pd.read_csv(os.path.join(data_dir, f"sib200_{split}_{cond}.tsv"), sep="\t",
                       quoting=3, dtype={"index_id": int})  # quoting=3: QUOTE_NONE (texts contain quotes)


def get_xy(data_dir, split, cond, label2id):
    if cond == "mixed":
        df = pd.concat([load_split(data_dir, split, "latn"), load_split(data_dir, split, "cyrl")],
                       ignore_index=True)
    else:
        df = load_split(data_dir, split, cond)
    return df["text"].astype(str).tolist(), [label2id[c] for c in df["category"]]


def make_loader(tok, texts, labels, max_len, bs, shuffle):
    enc = tok(texts, truncation=True, max_length=max_len)
    items = [{**{k: enc[k][i] for k in enc}, "labels": labels[i]} for i in range(len(texts))]
    n_trunc = sum(len(x) >= max_len for x in enc["input_ids"])

    def collate(batch):
        # IMPORTANT: copy; popping from the shared dict would remove 'labels' for epoch 2+
        batch = [dict(b) for b in batch]
        lab = torch.tensor([b.pop("labels") for b in batch])
        out = tok.pad(batch, return_tensors="pt")
        out["labels"] = lab
        return out
    dl = DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate)
    dl.n_truncated = n_trunc  # sentences that hit max_len (must be compared across scripts)
    return dl


# ----------------------------------------------------------------------------- train / eval
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for b in loader:
        b = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            logits = model(**b).logits
        preds += logits.argmax(-1).cpu().tolist()
    return preds


def run_one(args, tok, label2id, train_cond, lr, seed, test_conds):
    set_seed(seed)
    n_lab = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_lab, trust_remote_code=args.trust_remote_code).to(DEV)
    xtr, ytr = get_xy(args.data_dir, "train", train_cond, label2id)
    xva, yva = get_xy(args.data_dir, "validation", train_cond, label2id)
    tr = make_loader(tok, xtr, ytr, args.max_len, args.bs, True)
    va = make_loader(tok, xva, yva, args.max_len, 64, False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))

    best_f1, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        for b in tr:
            b = {k: v.to(DEV) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        f1 = f1_score(yva, predict(model, va), average="macro",
                      labels=list(range(n_lab)), zero_division=0)
        if f1 > best_f1:
            best_f1, best_ep = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    out = {"val_macro_f1": best_f1, "best_epoch": best_ep + 1,
           # training collapse (model predicts ~one class): best validation Macro-F1 < 0.2.
           # Flagged, NOT retried, so the fairness protocol stays unchanged; report it.
           "collapsed": bool(best_f1 < 0.2), "tests": {}}
    for tc in test_conds:
        xte, yte = get_xy(args.data_dir, "test", tc, label2id)
        te = make_loader(tok, xte, yte, args.max_len, 64, False)
        p = predict(model, te)
        out["tests"][tc] = {
            "acc": accuracy_score(yte, p),
            "macro_f1": f1_score(yte, p, average="macro", labels=list(range(n_lab)), zero_division=0),
            "n_classes_never_predicted": int(n_lab - len(set(p))),
            "n_truncated": te.n_truncated,
            "preds": p, "labels": yte,
        }
    del model, opt
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results/encoder")
    ap.add_argument("--stage", choices=["pilot", "final", "all"], default="all")
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-5, 5e-5, 1e-4])
    ap.add_argument("--pilot_seed", type=int, default=13)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 7, 99])
    ap.add_argument("--train_conds", nargs="+", default=["latn", "cyrl", "mixed"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    # 256, not 128: Cyrillic text yields more subword tokens for some tokenizers, so a short
    # limit would truncate Cyrillic more than Latin (a hidden confound). n_truncated is logged.
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    slug = args.model.replace("/", "__")
    odir = os.path.join(args.out_dir, slug)
    os.makedirs(odir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    labels = sorted(load_split(args.data_dir, "train", "latn")["category"].unique())
    label2id = {c: i for i, c in enumerate(labels)}
    pilot_path = os.path.join(odir, "pilot.json")

    # ---- pilot: lr selection (train=latn, val=latn, one seed)
    if args.stage in ("pilot", "all"):
        res = {}
        for lr in args.lrs:
            r = run_one(args, tok, label2id, "latn", lr, args.pilot_seed, ["latn"])
            res[repr(lr)] = r["val_macro_f1"]
            print(f"[pilot] {args.model} lr={lr} val_macroF1={r['val_macro_f1']:.4f}", flush=True)
        best_lr = float(max(res, key=res.get))  # ties -> first (lowest) lr in the grid order
        json.dump({"grid": res, "selected_lr": best_lr, "pilot_seed": args.pilot_seed,
                   "note": "selected on validation Macro-F1, train=latn only"},
                  open(pilot_path, "w"), indent=2)

    # ---- final: reuse the selected lr for every training condition
    if args.stage in ("final", "all"):
        if not os.path.exists(pilot_path):
            raise SystemExit(f"{pilot_path} missing: run --stage pilot (or all) first")
        lr = json.load(open(pilot_path))["selected_lr"]
        runs_path = os.path.join(odir, "runs.jsonl")
        done = set()  # resume support: never duplicate a (train_cond, seed) run
        if os.path.exists(runs_path):
            for line in open(runs_path, encoding="utf-8"):
                d = json.loads(line)
                done.add((d["train_cond"], d["seed"], d["lr"]))
        with open(runs_path, "a", encoding="utf-8") as f:
            for tcond in args.train_conds:
                for seed in args.seeds:
                    if (tcond, seed, lr) in done:
                        print(f"[skip] {tcond} seed={seed} already done", flush=True)
                        continue
                    t0 = time.time()
                    r = run_one(args, tok, label2id, tcond, lr, seed, TEST_CONDS)
                    rec = {"model": args.model, "train_cond": tcond, "lr": lr, "seed": seed,
                           "label_names": labels, **r, "seconds": round(time.time() - t0, 1)}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"[final] {args.model} train={tcond} seed={seed} " +
                          " ".join(f"{k}:{v['macro_f1']:.3f}" for k, v in r["tests"].items()), flush=True)


if __name__ == "__main__":
    main()
