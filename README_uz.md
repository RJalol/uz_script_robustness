# Lotin va kirill yozuvi: o'zbek NLP modellarining chidamliligi — ish yo'riqnomasi

Holat sanasi: 2026-10-18. Natijalar hali yo'q: barcha tajribalar Kaggle'da ishga tushirilishi kerak.

## 1. Tajriba qanday ishlaydi

Bir xil gaplar uch ko'rinishda beriladi:

| Shart | Mazmuni | Nima uchun kerak |
| --- | --- | --- |
| `latn` | Asl lotin matni | Asosiy ko'rsatkich |
| `latn_norm` | Lotin, barcha apostrof belgilari ASCII `'` ga keltirilgan | Nazorat sharti: apostrof normallashuvi ta'sirini ajratadi |
| `cyrl` | `translit_uz.lat2cyr` bilan kirillga o'girilgan | Yozuv o'zgarishi |

- Kodlovchi modellar (SIB-200, 7 sinf): trenirovka yozuvi `latn | cyrl | mixed`, test yozuvi `latn | latn_norm | cyrl`. Shunda 3x3 o'tish matritsasi hosil bo'ladi.
- LLM (Belebele, 900 savol): matn, savol va 4 variant to'liq o'giriladi; A/B/C/D harflarining keyingi-token ehtimoli solishtiriladi (generatsiya yo'q, shuning uchun tasodifiylik yo'q).
- Tashxis: tokenizator ko'rsatkichlari (so'zga token, UNK ulushi) va ixtiyoriy bits-per-char.
- Asosiy son: `Delta = ko'rsatkich(cyrl) - ko'rsatkich(latn)`; sof yozuv ta'siri `cyrl - latn_norm`.

## 2. Fayllar

| Fayl | Vazifasi |
| --- | --- |
| `translit_uz.py`, `test_translit.py` | Qoidaga asoslangan o'girgich va 29 ta test (hammasi o'tdi) |
| `prepare_data.py` | SIB-200 (GitHub) va Belebele (HF) dan uch shartli fayllar yaratadi |
| `exp_encoder_sib200.py` | Kodlovchilarni o'qitish: pilot (lr tanlash) + yakuniy (5 seed) |
| `analyze_encoder.py` | Jadvallar, Delta, bootstrap CI, Holm tuzatishi |
| `exp_llm_belebele.py`, `analyze_llm.py` | LLM baholash va McNemar/bootstrap tahlili |
| `exp_diagnostics.py` | Tokenizator va (ixtiyoriy) LM ehtimoli tashxisi |
| `validate_translit.py` | O'girgichni inson yozgan kirill matniga qarshi tekshirish (CER) |

## 3. Kaggle'da ishga tushirish tartibi

Internet yoqilgan, GPU T4 (yoki 2xT4) bo'lsin. Papkani Kaggle dataset sifatida yuklang yoki GitHub'dan klonlang.

```bash
pip install -q transformers datasets accelerate bitsandbytes scipy scikit-learn pytest
python -m pytest -q test_translit.py                     # 29 test o'tishi kerak
python prepare_data.py --out data --belebele             # uch shartli ma'lumotlar

# kodlovchilar (har model uchun alohida)
python exp_encoder_sib200.py --model xlm-roberta-base --stage all
python exp_encoder_sib200.py --model bert-base-multilingual-cased --stage all
python exp_encoder_sib200.py --model <TahrirchiBERT-base-id> --stage all     # id ni aniq yozing
python exp_encoder_sib200.py --model <TahrirchiBERT-small-id> --stage all
python analyze_encoder.py results/encoder results/tables

# tokenizator tashxisi
python exp_diagnostics.py --models xlm-roberta-base bert-base-multilingual-cased <Tahrirchi ids>

# LLM (modellar ro'yxatini avval belgilang; 7-9B uchun --load_4bit)
python exp_llm_belebele.py --model <hf-model-id> --shots 0 --load_4bit
python exp_llm_belebele.py --model <hf-model-id> --shots 5 --load_4bit
python analyze_llm.py results/llm results/tables
```

Vaqt sarfi o'lchanmagan: birinchi kodlovchi ishidan keyin `runs.jsonl` dagi `seconds` maydonidan baholang.

## 4. Siz hal qilishingiz kerak bo'lgan masalalar

1. **LLM ro'yxati** (ochiq modellar, kvantlash, API modellari bormi) — [INFORMATION REQUIRED].
2. **Tahrirchi model identifikatorlari** Hugging Face'dagi aniq nomi bilan — [VERIFICATION REQUIRED].
3. **FLORES kirill varianti:** qidiruvda shimoliy o'zbek kirill FLORES topilmadi (faqat janubiy o'zbek, arab yozuvida, FLORES+ dev). Yuborishdan oldin FLORES-200 / FLORES+ ni qo'lda tekshiring — [VERIFICATION REQUIRED].
4. **Tabiiy kirill tekshiruvi:** FeruzaSpeech (lotin va kirill transkriptlar) dan juft gaplar olib `validate_translit.py` ni ishlating. Ruxsat va litsenziya — [VERIFICATION REQUIRED]. Bu tekshiruvsiz "sun'iy kirill" deb yozishdan tashqari xulosa qilib bo'lmaydi.
5. **Litsenziyalar:** SIB-200 va Belebele hosila ma'lumotlarini e'lon qilishdan oldin.
6. **Maqsadli jurnal** tanlanmagan; formatlashdan oldin jurnal tahlilini bajaring.

## 5. Ma'lum xavflar

- Kirill matni sun'iy: rus o'zlashma so'zlaridagi ц, щ, ь kabi harflar lotindan tiklanmaydi (masalan `prinsipi` → `принсипи`, lug'atdagi imlo `принцип`).
- SIB-200 testi 204 gap: alohida modellar orasidagi farqlar shovqinli, shuning uchun 5 seed va bootstrap CI.
- Gipoteza (natija emas): tokenizatordagi kirill qamrovi farqni tushuntirishi mumkin. Tashxis natijalari buni tasdiqlamaguncha mexanizm sifatida yozmang.
- Oldingi yangiliklar tajribasidagi cheklovlar (kam seed, bir seedli pilot, eng past lr tanlanishi) bu yerga ham o'tadi.

## 6. Maqola

Inglizcha qoralama Docs hujjatida: sarlavha, annotatsiya, kirish, bog'liq ishlar, ma'lumotlar, protokol, bo'sh jadvallar, cheklovlar, deklaratsiyalar. Natijalar bo'limini faqat sonlar chiqqach yozing. O'zbek yoki rus tiliga tarjima natijalardan keyin qilinadi.
