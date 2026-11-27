# Script robustness of Uzbek NLP models: Latin vs Cyrillic

Code for comparing Latin, apostrophe-normalised Latin and rule-based Cyrillic transliteration of SIB-200 (encoders) and Belebele (LLMs).
Cyrillic text here is transliterated from Latin, not natural Cyrillic Uzbek.

- Scripts: `translit_uz.py`, `prepare_data.py`, `exp_encoder_sib200.py`, `analyze_encoder.py`, `exp_llm_belebele.py`, `analyze_llm.py`, `exp_diagnostics.py`, `test_translit.py`
- Kaggle notebook: `notebooks/uz_script_robustness_kaggle.ipynb` (`SMOKE = False` for the full run; GPU T4 x2, Internet on)
- Results of the full run: `results/full_run_results.md` (summary only; raw files not archived yet)
- Usage notes (Uzbek): `README_uz.md`

Datasets and models keep their own licences [VERIFICATION REQUIRED]. No licence is set for this code yet [INFORMATION REQUIRED].
