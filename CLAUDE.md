# Sentimen NLP — AI Menggantikan Manusia

## Tujuan Proyek
Analisis sentimen terhadap tema **AI menggantikan manusia**, menggunakan data komentar TikTok yang telah di-scraping.

## File Data

| File | Status | Keterangan |
|------|--------|------------|
| `Neysa_Scrap 1.csv` | Raw | Attribute lengkap — sudah dibersihkan ke `Neysa_Scrap 1_clean.csv` |
| `Neysa_Scrap 1_clean.csv` | ✅ Siap | Hanya kolom `text`, 557 baris |
| `Rizky_Hasil scrap komentar.csv` | ✅ Siap | Hanya kolom `text` |
| `Rizky_Hasil scrap komentar full-attr.csv` | Arsip | Versi lengkap attribute data Rizky |
| `Rizky_Hasil Scrapper untuk 10 link.csv` | Arsip | Metadata video TikTok, tidak masuk pipeline |

## Alur Kerja Final

```
1. Merge data
   Neysa_Scrap 1_clean.csv + Rizky_Hasil scrap komentar.csv
         ↓
2. Preprocessing teks
   - Hapus URL, mention, hashtag
   - Hapus/normalisasi emoji
   - Strip whitespace & karakter aneh
         ↓
3. Language Detection (FastText)
   + Routing ke model sentimen per bahasa:
   ┌─────────────────────────────────────────────────┐
   │ 'en'  → cardiffnlp/twitter-xlm-roberta-base-   │
   │          sentiment                              │
   │ 'id'  → mdhugol/indonesia-bert-sentiment-      │
   │          classification                         │
   │ other → cardiffnlp/twitter-xlm-roberta-base-   │
   │          sentiment-multilingual                 │
   └─────────────────────────────────────────────────┘
         ↓
4. Auto-labeling → label: Positif / Negatif / Netral
         ↓
5. Training & Evaluasi
   - SVM (TF-IDF + LinearSVC)
   - Transformer (hasil routing step 3)
   - Bandingkan: Accuracy, F1-score, Confusion Matrix
         ↓
6. Visualisasi
   - Distribusi sentimen (pie/bar chart)
   - WordCloud per sentimen
   - Confusion matrix SVM vs Transformer
   - Perbandingan akurasi
```

## Stack

```
pandas · fasttext-langdetect · transformers · torch
scikit-learn · matplotlib · seaborn · wordcloud
```

## Model per Bahasa

| Bahasa | Model HuggingFace |
|--------|------------------|
| English (mayoritas) | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| Indonesian (signifikan) | `mdhugol/indonesia-bert-sentiment-classification` |
| Japanese, Hindi, lainnya | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` |

## Struktur File Output

```
output/
  ├── data_merged.csv          # gabungan Neysa + Rizky
  ├── data_preprocessed.csv    # setelah cleaning
  ├── data_labeled.csv         # + kolom label, language, confidence
  └── figures/
        ├── sentiment_distribution.png
        ├── wordcloud_positif.png
        ├── wordcloud_negatif.png
        ├── wordcloud_netral.png
        ├── confusion_matrix_svm.png
        └── model_comparison.png
```
