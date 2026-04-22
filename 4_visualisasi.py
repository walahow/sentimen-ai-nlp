"""
Step 4: Visualisasi hasil sentimen.
Output: output/figures/*.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

os.makedirs("output/figures", exist_ok=True)

df = pd.read_csv("output/data_labeled.csv", encoding="utf-8")
df = df.dropna(subset=["text_clean", "label"])

COLORS = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#3498db"}
label_counts = df["label"].value_counts()

# --- 1. Pie Chart distribusi sentimen ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    label_counts,
    labels=label_counts.index,
    autopct="%1.1f%%",
    colors=[COLORS.get(l, "#aaa") for l in label_counts.index],
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax.set_title("Distribusi Sentimen Komentar TikTok\n(Tema: AI Menggantikan Manusia)", fontsize=13)
plt.tight_layout()
plt.savefig("output/figures/sentiment_distribution_pie.png", dpi=150)
plt.close()

# --- 2. Bar Chart distribusi sentimen ---
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(label_counts.index, label_counts.values,
              color=[COLORS.get(l, "#aaa") for l in label_counts.index],
              edgecolor="white")
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(val), ha="center", fontsize=11)
ax.set_xlabel("Sentimen")
ax.set_ylabel("Jumlah Komentar")
ax.set_title("Distribusi Sentimen Komentar TikTok", fontsize=13)
plt.tight_layout()
plt.savefig("output/figures/sentiment_distribution_bar.png", dpi=150)
plt.close()

# --- 3. WordCloud per sentimen ---
for label in ["Positif", "Negatif", "Netral"]:
    subset = df[df["label"] == label]["text_clean"].dropna()
    if subset.empty:
        continue
    combined = " ".join(subset.tolist())
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Greens" if label == "Positif" else ("Reds" if label == "Negatif" else "Blues"),
        max_words=100,
        collocations=False,
    ).generate(combined)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"output/figures/wordcloud_{label.lower()}.png", dpi=150)
    plt.close()
    print(f"WordCloud {label} tersimpan")

# --- 4. Distribusi sentimen per bahasa ---
lang_label = df.groupby(["language", "label"]).size().unstack(fill_value=0)
top_langs = df["language"].value_counts().head(6).index
lang_label = lang_label.loc[lang_label.index.isin(top_langs)]

fig, ax = plt.subplots(figsize=(9, 5))
lang_label.plot(kind="bar", ax=ax,
                color=[COLORS.get(c, "#aaa") for c in lang_label.columns],
                edgecolor="white")
ax.set_xlabel("Bahasa")
ax.set_ylabel("Jumlah Komentar")
ax.set_title("Distribusi Sentimen per Bahasa (Top 6)", fontsize=13)
ax.legend(title="Sentimen")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("output/figures/sentiment_per_language.png", dpi=150)
plt.close()

print("\nSemua visualisasi tersimpan di output/figures/")
