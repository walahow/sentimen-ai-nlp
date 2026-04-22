"""
Step 2: Language detection + routing ke transformer model per bahasa.
Auto-label setiap komentar → Positif / Negatif / Netral
Output: output/data_labeled.csv
"""

import pandas as pd
from langdetect import detect, LangDetectException
from transformers import pipeline
import torch

df = pd.read_csv("output/data_preprocessed.csv", encoding="utf-8")
print(f"Loaded {len(df)} baris")

# --- Language Detection ---
def detect_lang(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "en"

print("Mendeteksi bahasa...")
df["language"] = df["text_clean"].apply(detect_lang)
lang_counts = df["language"].value_counts()
print(f"Distribusi bahasa:\n{lang_counts.head(10)}\n")

# --- Load Models ---
device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")

print("Loading model English...")
pipe_en = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=device,
    truncation=True,
    max_length=128,
)

print("Loading model Indonesian...")
pipe_id = pipeline(
    "sentiment-analysis",
    model="mdhugol/indonesia-bert-sentiment-classification",
    device=device,
    truncation=True,
    max_length=128,
)

print("Loading model Multilingual (fallback)...")
pipe_multi = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    device=device,
    truncation=True,
    max_length=128,
)

# --- Label normalization ---
# setiap model punya label format berbeda, normalize ke Positif/Negatif/Netral
def normalize_label(raw_label: str) -> str:
    raw = raw_label.lower().strip()
    if raw in ("positive", "pos", "label_2", "positif"):
        return "Positif"
    if raw in ("negative", "neg", "label_0", "negatif"):
        return "Negatif"
    return "Netral"

# --- Routing & Inference ---
def get_sentiment(row):
    lang = row["language"]
    text = str(row["text_clean"])[:512]
    try:
        if lang == "id":
            result = pipe_id(text)[0]
        elif lang == "en":
            result = pipe_en(text)[0]
        else:
            result = pipe_multi(text)[0]
        return normalize_label(result["label"]), round(result["score"], 4)
    except Exception:
        return "Netral", 0.0

print("Menjalankan auto-labeling (ini bisa makan waktu beberapa menit)...")
results = df.apply(get_sentiment, axis=1)
df["label"]      = [r[0] for r in results]
df["confidence"] = [r[1] for r in results]

df.to_csv("output/data_labeled.csv", index=False, encoding="utf-8")
print(f"\nSelesai! Distribusi label:\n{df['label'].value_counts()}")
print(df[["text_clean", "language", "label", "confidence"]].head(10).to_string())
