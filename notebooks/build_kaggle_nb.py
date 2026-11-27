"""Builds a compact, self-contained Kaggle notebook: all scripts are embedded as ONE base64(tar.gz) blob.
usage: python build_kaggle_nb.py OUT.ipynb [config_override.py]   (override only for the local smoke test)"""
import base64, io, sys, tarfile
import nbformat as nbf

SRC = "/home/claude/script_robustness/"
OUT = sys.argv[1]
OVERRIDE = open(sys.argv[2]).read() if len(sys.argv) > 2 else None
FILES = ["translit_uz.py", "test_translit.py", "prepare_data.py", "exp_encoder_sib200.py", "analyze_encoder.py",
         "exp_llm_belebele.py", "analyze_llm.py", "exp_diagnostics.py"]

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as t:
    for f in FILES:
        t.add(SRC + f, arcname=f)
blob = base64.b64encode(buf.getvalue()).decode()
blob_lines = "\n".join(blob[i:i + 100] for i in range(0, len(blob), 100))

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Lotin va kirill yozuvi: o'zbek NLP modellarining chidamliligi

SIB-200 (kodlovchilar) va Belebele (LLM) bo'yicha lotin / lotin-normallashgan / kirill (qoidaga asoslangan o'girish) shartlari.
Notebook ketma-ket ishlaydi (Save & Run All mumkin). `SMOKE = True` tezkor sinov, `False` to'liq tajriba.
Natijalar `/kaggle/working/results` va `results.zip` ga yoziladi.
""")

code("""
import os, sys, subprocess, json, glob, shutil, time
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()
os.chdir(WORK); sys.path.insert(0, WORK)

SMOKE = False           # True: tezkor sinov (2 epoxa, 1 seed, 20 ta LLM savoli); to'liq tajribada False

ENCODER_MODELS = [
    "FacebookAI/xlm-roberta-base",
    "google-bert/bert-base-multilingual-cased",
    "tahrirchi/tahrirchi-bert-base",
    "tahrirchi/tahrirchi-bert-small",
]
LLM_MODELS = ["Qwen/Qwen3-8B"]     # uzlm/alloma-8B-Instruct tokenizator tekshiruvidan keyin alohida
LLM_SHOTS = [0]
LLM_4BIT = True
RUN_ENCODERS = True
RUN_DIAGNOSTICS = True
RUN_LLM = True

SEEDS = [13] if SMOKE else [13, 21, 42, 7, 99]
EPOCHS = 2 if SMOKE else 10
LRS = ["3e-5", "1e-4"] if SMOKE else ["3e-5", "5e-5", "1e-4"]
LLM_LIMIT = 20 if SMOKE else None
""" + ("\n" + OVERRIDE if OVERRIDE else "") + """

def run(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if any(w in line for w in ("Warning", "warn(", "FutureWarning")):
            continue
        print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with code {p.returncode}: {cmd}")

import torch
print("GPU:", torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("SMOKE =", SMOKE, "| seeds =", SEEDS, "| epochs =", EPOCHS, "| encoders =", len(ENCODER_MODELS), "| LLMs =", LLM_MODELS)
""")

code("""
run("pip install -q accelerate bitsandbytes scipy scikit-learn pytest datasets")
import transformers, datasets
print("transformers", transformers.__version__, "| datasets", datasets.__version__)
""")

md("## Skriptlar (%%writefile). Ularni o'zgartirish shart emas.")
for _f in FILES:
    code(f"%%writefile {_f}\n" + open(SRC + _f, encoding="utf-8").read().rstrip("\n"))
code("run('python -m py_compile ' + ' '.join(f for f in os.listdir(WORK) if f.endswith('.py'))); print('skriptlar kompilyatsiya qilindi')")

md("## O'girgich testi (29 ta test o'tishi kerak)")
code('run("python -m pytest -q test_translit.py")')

md("## Ma'lumot tayyorlash (lotin / lotin-normallashgan / kirill)")
code("""
run("python prepare_data.py --out data" + (" --belebele" if RUN_LLM and LLM_MODELS else ""))
""")

md("## Kodlovchilar (SIB-200): pilot, so'ng yakuniy; to'xtasa shu katakchani qayta ishga tushiring (resume)")
code("""
if RUN_ENCODERS:
    for m in ENCODER_MODELS:
        t0 = time.time()
        run(f"python exp_encoder_sib200.py --model {m} --data_dir data --out_dir results/encoder "
            f"--stage all --epochs {EPOCHS} --seeds {' '.join(map(str, SEEDS))} --lrs {' '.join(LRS)}")
        print(f"### {m}: {(time.time()-t0)/60:.1f} min", flush=True)
    run("python analyze_encoder.py results/encoder results/tables")
""")

md("## Tokenizator tashxisi")
code("""
if RUN_DIAGNOSTICS:
    run("python exp_diagnostics.py --models " + " ".join(ENCODER_MODELS) + " --data_dir data --out results/tables/diagnostics_encoders.csv")
""")

md("## LLM (Belebele, 0-shot, 4-bit)")
code("""
if RUN_LLM and LLM_MODELS:
    for m in LLM_MODELS:
        for k in LLM_SHOTS:
            t0 = time.time()
            cmd = f"python exp_llm_belebele.py --model {m} --data_dir data --out_dir results/llm --shots {k}"
            if LLM_4BIT: cmd += " --load_4bit"
            if LLM_LIMIT: cmd += f" --limit {LLM_LIMIT}"
            run(cmd)
            print(f"### {m} {k}-shot: {(time.time()-t0)/60:.1f} min", flush=True)
    run("python analyze_llm.py results/llm results/tables")
""")

md("## Natijalar")
code("""
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
for f in sorted(glob.glob("results/tables/*.csv")):
    if "all_runs" in f: continue
    print("\\n==", f); print(pd.read_csv(f).round(4).to_string(index=False)[:8000])
shutil.make_archive(os.path.join(WORK, "results"), "zip", WORK, "results")
print("\\nTAYYOR:", os.path.join(WORK, "results.zip"))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("written", OUT, len(cells), "cells,", len(blob), "blob chars")
