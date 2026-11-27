"""
exp_diagnostics.py -- WHY does the script change hurt?  Tokenizer + (optional) LM-likelihood diagnostics.

For every tokenizer/model and every condition (latn, latn_norm, cyrl) on SIB-200 sentences
(SIB-200 sentences are FLORES-200 sentences) and, optionally, an extra sentence file:

  tokens_per_word   fertility
  tokens_per_char   tokens per non-space character
  unk_rate          share of tokens equal to unk_token (if the tokenizer has one)
  bits_per_word     (optional) bits per WORD of a causal LM -> comparable across tokenisations and scripts

python exp_diagnostics.py --models xlm-roberta-base bert-base-multilingual-cased --data_dir data
python exp_diagnostics.py --models <causal-lm-id> --lm --data_dir data
python exp_diagnostics.py --models ... --extra_file flores_uzn_sentences.txt     # one sentence per line

FLORES note: FLORES-200 provides uzn_Latn; a Cyrillic Uzbek variant of FLORES was NOT found during
the literature/data check (only Southern Uzbek in Arabic script exists in FLORES+ dev)
[VERIFICATION REQUIRED before submission]. Therefore the Cyrillic text is synthetic.
"""
import argparse, glob, math, os
import numpy as np
import pandas as pd


def read_sib(data_dir, cond):
    texts = []
    for split in ("train", "validation", "test"):
        df = pd.read_csv(os.path.join(data_dir, f"sib200_{split}_{cond}.tsv"), sep="\t", quoting=3)
        texts += df["text"].astype(str).tolist()
    return texts


def char_retention(tok, texts, n=300):
    """Share of non-space characters that survive encode -> decode (special tokens such as <unk>
    are dropped by the decoder). 1.0 = lossless; a value near 0 means the tokenizer silently
    throws the text away (observed for uzlm/alloma-8B-Instruct on Cyrillic in the smoke run).
    Whitespace and case are ignored on purpose, so it measures lost CHARACTERS only."""
    kept = total = 0
    for t in texts[:n]:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        dec = tok.decode(ids, skip_special_tokens=True)
        a = "".join(t.split()).lower()
        b = "".join(dec.split()).lower()
        total += len(a)
        kept += min(len(b), len(a))
    return kept / max(total, 1)


def tok_stats(tok, texts, max_lens=(128, 256)):
    n_tok = n_word = n_char = n_unk = 0
    over = {m: 0 for m in max_lens}
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        n_word += len(t.split())
        n_char += len(t.replace(" ", ""))
        for m in max_lens:  # +2 for [CLS]/[SEP]-like special tokens
            over[m] += int(len(ids) + 2 > m)
        if tok.unk_token_id is not None:
            n_unk += sum(1 for i in ids if i == tok.unk_token_id)
    r = {"tokens_per_word": n_tok / n_word,
         # per-character values are NOT strictly comparable across scripts: sh/ch/o'/g' are 2 Latin
         # characters but 1 Cyrillic letter. Words are identical across scripts -> use per word.
         "tokens_per_char": n_tok / n_char,
         "unk_rate": n_unk / max(n_tok, 1), "n_sent": len(texts), "n_words": n_word,
         "char_retention": char_retention(tok, texts)}
    for m in max_lens:
        r[f"share_truncated_at_{m}"] = over[m] / len(texts)
    return r


def bits_per_word(model_id, texts, limit=None):
    """Total negative log2-likelihood divided by the number of WORDS (identical across scripts),
    which is comparable across tokenisations and scripts; bits per character is not (see tok_stats)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()
    nll = words = 0.0
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for t in (texts if limit is None else texts[:limit]):
        ids = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        ids = torch.cat([torch.tensor([[bos]], device=ids.device), ids], 1)
        with torch.inference_mode():
            logits = model(ids).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        nll += -lp.sum().item()
        words += len(t.split())
    return nll / math.log(2) / words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--lm", action="store_true", help="also compute bits per word (causal LMs; run causal LMs in a separate call from encoders)")
    ap.add_argument("--extra_file", default=None)
    ap.add_argument("--out", default="results/tables/diagnostics.csv")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from translit_uz import lat2cyr, latin_normalised

    sets = {c: read_sib(a.data_dir, c) for c in ("latn", "latn_norm", "cyrl")}
    if a.extra_file:
        extra = [l.strip() for l in open(a.extra_file, encoding="utf-8") if l.strip()]
        sets.update({"extra_latn": extra, "extra_latn_norm": [latin_normalised(x) for x in extra],
                     "extra_cyrl": [lat2cyr(x) for x in extra]})
    rows = []
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        for c, texts in sets.items():
            r = {"model": m, "cond": c, **tok_stats(tok, texts)}
            if a.lm:
                r["bits_per_word"] = bits_per_word(m, texts)
            rows.append(r)
            print(r, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
