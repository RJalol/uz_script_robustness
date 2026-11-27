"""
exp_llm_belebele.py -- script robustness of decoder LLMs on Belebele (uzn_Latn) with
Latin / Latin-normalised / Cyrillic versions of the *whole* item
(passage, question, four options).

Scoring
-------
No free-form generation: the prompt ends with "Answer:" and we compare the next-token
logits of the four option letters A/B/C/D (standard, deterministic, no parsing errors).
Optional k-shot examples are drawn from OTHER Belebele items with a fixed seed and are
rendered in the SAME script as the test item.

Why this design
---------------
* Deterministic -> differences between conditions are not sampling noise.
* The instruction text is English and identical in all conditions, so only the Uzbek
  content changes script.
* Letter-position bias is recorded (pred_letter_counts) because it can interact with script.

Usage (Kaggle, 2xT4: use 4-bit for 7-9B models)
-----------------------------------------------
python prepare_data.py --out data --belebele
python exp_llm_belebele.py --model <hf-model-id> --shots 0 --load_4bit
python exp_llm_belebele.py --model <hf-model-id> --shots 5 --load_4bit

Model ids are an experimental choice -> [INFORMATION REQUIRED]. Decide the list before running
and keep it fixed (open-weight instruct models of 7-9B + at least one multilingual-strong model).
"""
import argparse, json, os, random, time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CONDS = ["latn", "latn_norm", "cyrl"]
LETTERS = ["A", "B", "C", "D"]


def load_cond(data_dir, cond):
    return [json.loads(l) for l in open(os.path.join(data_dir, f"belebele_{cond}.jsonl"), encoding="utf-8")]


def render(ex, with_answer):
    s = (f"P: {ex['flores_passage']}\nQ: {ex['question']}\n"
         f"A: {ex['mc_answer1']}\nB: {ex['mc_answer2']}\nC: {ex['mc_answer3']}\nD: {ex['mc_answer4']}\n"
         "Please choose the correct answer from the options above.\nAnswer:")
    if with_answer:
        s += " " + LETTERS[int(ex["correct_answer_num"]) - 1]
    return s


def letter_token_ids(tok):
    ids = []
    for L in LETTERS:
        cand = set()
        for form in (" " + L, L):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                cand.add(t[-1])
        ids.append(sorted(cand))
    return ids


@torch.inference_mode()
def score_item(model, tok, prompt, ids_per_letter, max_ctx):
    """Returns (letter log-probs, truncated_flag, n_tokens).
    If the prompt is longer than max_ctx the START is cut (demonstrations first), keeping BOS and the
    END (question + options + 'Answer:'). Cyrillic needs more tokens than Latin, so truncation can
    differ between scripts -> it is recorded per item and MUST be reported."""
    enc = tok(prompt, return_tensors="pt", truncation=False)
    n_tok = int(enc["input_ids"].shape[1])
    truncated = n_tok > max_ctx
    if truncated:
        keep_bos = tok.bos_token_id is not None and int(enc["input_ids"][0, 0]) == tok.bos_token_id
        tail = max_ctx - 1 if keep_bos else max_ctx
        ids = enc["input_ids"][:, -tail:]
        am = enc["attention_mask"][:, -tail:]
        if keep_bos:
            ids = torch.cat([enc["input_ids"][:, :1], ids], 1)
            am = torch.cat([enc["attention_mask"][:, :1], am], 1)
        enc = {"input_ids": ids, "attention_mask": am}
    enc = {k: v.to(model.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    logits = model(**enc).logits[0, -1].float()
    logp = torch.log_softmax(logits, -1)
    return [torch.logsumexp(logp[ids], 0).item() for ids in ids_per_letter], truncated, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results/llm")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_ctx", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N items")
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    ids = letter_token_ids(tok)
    assert all(len(x) > 0 for x in ids), "could not find single tokens for answer letters"

    data = {c: load_cond(args.data_dir, c) for c in CONDS}
    n = len(data["latn"]) if args.limit is None else args.limit

    # ---- tokenizer sanity check: does the tokenizer silently drop characters of some script?
    # (a model that never SEES the Uzbek text answers by letter bias, not by reading; that is a
    #  tokenizer limitation, not "script robustness", and must be reported separately)
    tokcheck = {}
    for cond in CONDS:
        texts = [ex[k] for ex in data[cond][:100]
                 for k in ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4")]
        kept = total = 0
        for t in texts:
            tids = tok(t, add_special_tokens=False)["input_ids"]
            a = "".join(t.split()).lower()
            b = "".join(tok.decode(tids, skip_special_tokens=True).split()).lower()
            total += len(a); kept += min(len(a), len(b))
        tokcheck[cond] = kept / max(total, 1)
    print("tokenizer character retention:", {k: round(v, 3) for k, v in tokcheck.items()}, flush=True)
    for cond, v in tokcheck.items():
        if v < 0.9:
            print(f"!!! WARNING: tokenizer of {args.model} drops {100*(1-v):.0f}% of characters in '{cond}'. "
                  f"Accuracy in this condition does NOT measure reading ability.", flush=True)
    rng = random.Random(args.seed)
    # few-shot indices fixed per item and shared across conditions (same demos in every script)
    shots = {i: rng.sample([j for j in range(len(data["latn"])) if j != i], args.shots)
             for i in range(n)} if args.shots else {i: [] for i in range(n)}

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model.replace('/', '__')}__{args.shots}shot" + ("__4bit" if args.load_4bit else "")
    out_path = os.path.join(args.out_dir, tag + ".jsonl")
    json.dump({"model": args.model, "shots": args.shots, "char_retention": tokcheck},
              open(os.path.join(args.out_dir, tag + ".tokcheck.json"), "w"))
    with open(out_path, "w", encoding="utf-8") as f:
        for cond in CONDS:
            correct, letters, t0, n_trunc = [], Counter(), time.time(), 0
            for i in range(n):
                demos = "\n\n".join(render(data[cond][j], True) for j in shots[i])
                prompt = (demos + "\n\n" if demos else "") + render(data[cond][i], False)
                lp, trunc, n_tok = score_item(model, tok, prompt, ids, args.max_ctx)
                pred = int(np.argmax(lp)) + 1
                gold = int(data[cond][i]["correct_answer_num"])
                correct.append(int(pred == gold)); letters[LETTERS[pred - 1]] += 1
                n_trunc += int(trunc)
                f.write(json.dumps({"model": args.model, "shots": args.shots, "cond": cond, "i": i,
                                    "pred": pred, "gold": gold, "logp": lp,
                                    "truncated": trunc, "n_tokens": n_tok}) + "\n")
            acc = float(np.mean(correct))
            print(f"[{tag}] {cond}: acc={acc:.4f} n={n} truncated={n_trunc} "
                  f"letters={dict(letters)} ({time.time()-t0:.0f}s)", flush=True)
            f.flush()


if __name__ == "__main__":
    main()
