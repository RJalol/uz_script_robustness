# Script robustness (Latin vs Cyrillic, Uzbek) — full run results (Kaggle "full-run-1")

Run: Kaggle, GPU T4 x2, 1 h 57 min. Seeds 13,21,42,7,99; 10 epochs; max_len 256; lr chosen on pilot train=latn seed 13, reused elsewhere.
Qwen3-8B: 0-shot, 4-bit NF4, 900 Belebele items. Cyrillic = rule-based transliteration (not natural Cyrillic).
Source: log of Kaggle version "full-run-1" (tables printed by analyze_encoder.py / analyze_llm.py / exp_diagnostics.py). results.zip not yet downloaded locally [INFORMATION REQUIRED: archive raw files].

## Encoders, test Macro-F1 mean ± std (n=5)
| Model | tr latn→latn | latn→cyrl | tr cyrl→latn | cyrl→cyrl | tr mixed→latn | mixed→cyrl |
|---|---|---|---|---|---|---|
| XLM-R base | .7850±.0103 | .4802±.0329 | .6202±.1295 | .5493±.0648 | .6653±.3402 | .5142±.2573 |
| mBERT | .7506±.0112 | .5355±.0354 | .6273±.0400 | .6530±.0222 | .7355±.0222 | .6401±.0395 |
| TahrirchiBERT base | .8320±.0145 | .0924±.0519 | .2246±.0943 | .3584±.1826 | .8136±.0162 | .4978±.0343 |
| TahrirchiBERT small | .7877±.0178 | .0458±.0428 | .1379±.0436 | .3246±.0556 | .7768±.0330 | .3050±.0308 |
latn_norm equals latn exactly everywhere (D_typo = 0).

## D_script (cyrl − latn), train latn: mean [95% bootstrap CI], p_holm
XLM-R −0.3048 [−0.3719,−0.2419] 0.0001; mBERT −0.2151 [−0.2753,−0.1588] 0.0004; TBase −0.7396 [−0.7878,−0.6846] <0.0001; TSmall −0.7419 [−0.7888,−0.6870] <0.0001.
Train cyrl: XLM-R −0.0709 (p_holm .23), mBERT +0.0257 (.27), TBase +0.1338 (.109), TSmall +0.1867 (.006).
Train mixed: −0.1511, −0.0955, −0.3158, −0.4718 (p_holm .032, .032, .0001, <.0001).

## Collapsed runs (val Macro-F1 < 0.2): XLM-R mixed 1/5; TahrirchiBERT-base cyrl 1/5.

## Belebele, Qwen3-8B 0-shot: latn .6867, latn_norm .6856, cyrl .6500
D_script −0.0367 [−0.0633,−0.0100], McNemar p .0090 (59 only-cyrl correct vs 92 only-latn), n 900; D_typo −0.0011; D_pure −0.0356 (p_holm .0111). Truncated 0. Tokenizer char retention 1.0/1.0/1.0.
alloma-8B-Instruct not run (tokenizer erases Cyrillic — [VERIFICATION REQUIRED]).

## Tokenization (tokens/word latn → cyrl; share cyrl >128 tokens)
XLM-R 2.23→3.15 (0.4%); mBERT 2.83→3.43 (0.8%); TBase 1.81→3.65 (1.4%); TSmall 1.78→4.48 (5.8%).

## Caveats
Cyrillic-trained models not lr-tuned (val F1 .37–.67). SIB-200 already ASCII apostrophes → typography control uninformative. Transliterator not validated on natural Cyrillic [VERIFICATION REQUIRED].
