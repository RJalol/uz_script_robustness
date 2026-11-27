"""
prepare_data.py -- build the 3 script conditions for SIB-200 (uzn_Latn) and Belebele (uzn_Latn).

Conditions
----------
latn       : original text, untouched
latn_norm  : CONTROL -- same Latin text, apostrophe-like characters canonicalised to "'"
             (isolates typographic normalisation from the script change)
cyrl       : rule-based Latin->Cyrillic transliteration (translit_uz.lat2cyr)

Usage
-----
python prepare_data.py --out data            # SIB-200 only (GitHub raw files, no HF needed)
python prepare_data.py --out data --belebele # also Belebele (needs `datasets` + internet)

Writes:  data/sib200_{split}_{condition}.tsv  (index_id, category, text)
         data/belebele_{condition}.jsonl
         data/prep_report.json
"""
import argparse, json, os, urllib.request
from collections import Counter

from translit_uz import lat2cyr, latin_normalised, script_share, normalize_apostrophes

SIB_URL = "https://raw.githubusercontent.com/dadelani/sib-200/main/data/annotated/uzn_Latn/{split}.tsv"
SPLIT_FILES = {"train": "train", "validation": "dev", "test": "test"}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def transform_all(text: str) -> dict:
    return {"latn": text, "latn_norm": latin_normalised(text), "cyrl": lat2cyr(text)}


def prep_sib(out: str, report: dict):
    rep = {}
    for split, fname in SPLIT_FILES.items():
        raw = fetch(SIB_URL.format(split=fname))
        # NOTE: parse by hand. 78 SIB-200 lines contain a double quote, which the csv module
        # (default quoting) would treat as a quote character and corrupt.
        lines = [l for l in raw.replace("\r\n", "\n").split("\n") if l.strip()]
        header = lines[0].split("\t")
        assert header == ["index_id", "category", "text"], header
        rows = []
        for l in lines[1:]:
            parts = l.split("\t")
            assert len(parts) == 3, f"bad line: {l[:80]!r}"
            rows.append(dict(zip(header, parts)))
        for cond in ["latn", "latn_norm", "cyrl"]:
            path = os.path.join(out, f"sib200_{split}_{cond}.tsv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("\t".join(header) + "\n")
                for r in rows:
                    t = transform_all(r["text"])[cond]
                    assert "\t" not in t and "\n" not in t
                    f.write(f"{r['index_id']}\t{r['category']}\t{t}\n")
        # sanity statistics on this split
        cyr_text = " ".join(lat2cyr(r["text"]) for r in rows)
        rep[split] = {
            "n": len(rows),
            "labels": dict(Counter(r["category"] for r in rows)),
            "cyrl_script_share": script_share(cyr_text),
            "latn_script_share": script_share(" ".join(r["text"] for r in rows)),
        }
    report["sib200"] = rep


def prep_belebele(out: str, report: dict):
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", "uzn_Latn", split="test")
    fields_to_convert = ["flores_passage", "question", "mc_answer1", "mc_answer2",
                         "mc_answer3", "mc_answer4"]
    need = fields_to_convert + ["correct_answer_num"]
    missing = [c for c in need if c not in ds.column_names]
    assert not missing, f"Belebele columns changed, missing {missing}; found {ds.column_names}"
    assert len(ds) == 900, f"expected 900 Belebele items, got {len(ds)}"
    # `correct_answer_num` may be a ClassLabel (HF returns 0..3) or a plain string ("1".."4").
    # Normalise to the string "1".."4" that exp_llm_belebele.py expects.
    feat = ds.features["correct_answer_num"]

    def gold_str(v):
        if hasattr(feat, "int2str") and isinstance(v, int):
            v = feat.int2str(v)
        v = str(v)
        assert v in {"1", "2", "3", "4"}, f"unexpected correct_answer_num value: {v!r}"
        return v

    keep_meta = [c for c in ("link", "question_number") if c in ds.column_names]
    for cond in ["latn", "latn_norm", "cyrl"]:
        path = os.path.join(out, f"belebele_{cond}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in ds:
                # only JSON-safe fields (the `ds` timestamp column would break json.dumps)
                d = {k: ex[k] for k in keep_meta}
                for k in fields_to_convert:
                    d[k] = transform_all(ex[k])[cond]
                d["correct_answer_num"] = gold_str(ex["correct_answer_num"])
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    report["belebele"] = {"n": len(ds), "columns": ds.column_names,
                          "correct_answer_feature": type(feat).__name__}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--belebele", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}
    prep_sib(a.out, report)
    if a.belebele:
        prep_belebele(a.out, report)
    with open(os.path.join(a.out, "prep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
