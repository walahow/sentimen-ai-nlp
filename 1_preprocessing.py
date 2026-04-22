"""
Step 1: Merge data Neysa 1 + Neysa 2 + Rizky, lalu preprocessing teks.
Output: output/data_preprocessed.csv
"""

import re
import os
import pandas as pd

os.makedirs("output", exist_ok=True)

# --- Load & Merge ---
neysa1 = pd.read_csv("Neysa_Scrap 1_clean.csv", encoding="utf-8-sig")
neysa2 = pd.read_csv("Neysa_Scrap 2_clean.csv", encoding="utf-8-sig")
rizky  = pd.read_csv("Rizky_Hasil scrap komentar.csv", encoding="utf-8-sig")

print(f"Neysa 1: {len(neysa1)} baris")
print(f"Neysa 2: {len(neysa2)} baris")
print(f"Rizky  : {len(rizky)} baris")

df = pd.concat([neysa1, neysa2, rizky], ignore_index=True)
df.columns = ["text"]
df.drop_duplicates(subset="text", inplace=True)
df.dropna(subset="text", inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv("output/data_merged.csv", index=False, encoding="utf-8")
print(f"Merged: {len(df)} baris")

# --- Preprocessing ---
def preprocess(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)          # hapus URL
    text = re.sub(r"@\w+", "", text)                       # hapus mention
    text = re.sub(r"#\w+", "", text)                       # hapus hashtag
    text = re.sub(r"\[Sticker\]", "", text, flags=re.I)    # hapus [Sticker]
    # normalisasi emoji → hapus (transformer sudah handle emoji, tapi SVM butuh teks bersih)
    text = re.sub(r"[^\w\s.,!?'\"()-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text"].apply(preprocess)
df = df[df["text_clean"].str.len() > 2].reset_index(drop=True)

df.to_csv("output/data_preprocessed.csv", index=False, encoding="utf-8")
print(f"Setelah preprocessing: {len(df)} baris")
print(df[["text", "text_clean"]].head(5).to_string().encode("ascii", "replace").decode())
