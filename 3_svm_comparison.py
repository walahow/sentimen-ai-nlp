"""
Step 3: Train SVM menggunakan label dari auto-labeling.
TF-IDF (unigram+bigram, stopwords EN+ID) + LinearSVC
Output: output/svm_report.txt
        output/figures/confusion_matrix_svm.png
        output/figures/model_comparison.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

os.makedirs("output/figures", exist_ok=True)

df = pd.read_csv("output/data_labeled.csv", encoding="utf-8")
df = df.dropna(subset=["text_clean", "label"])
print(f"Data: {len(df)} baris | Label: {df['label'].value_counts().to_dict()}")

# ── Stopwords EN + ID ──────────────────────────────────────────────────────────
STOPWORDS_EN = {
    "i","me","my","we","our","you","your","he","his","she","her","it","its",
    "they","their","what","which","who","this","that","these","those",
    "am","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","can","could",
    "not","no","nor","but","and","or","so","yet","both","either","neither",
    "if","then","because","as","until","while","of","at","by","for","with",
    "about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over",
    "under","again","further","then","once","a","an","the","just","also",
    "more","most","other","some","such","than","too","very","s","t",
}

STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","dengan","untuk","pada",
    "adalah","juga","akan","ada","tidak","sudah","saya","aku","kamu","dia",
    "mereka","kita","kami","nya","pun","bisa","buat","karena","jadi",
    "kalau","tapi","atau","lebih","agar","apa","siapa","mana","bagaimana",
    "kapan","dimana","kenapa","sangat","sekali","belum","masih",
    "lagi","saja","hanya","setelah","sebelum","seperti","semua","banyak",
    "punya","mau","maka","namun","oleh","hal","cara","tahun","orang","dapat",
}

ALL_STOPWORDS = list(STOPWORDS_EN | STOPWORDS_ID)

# ── Split 80/20 stratified ─────────────────────────────────────────────────────
X = df["text_clean"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── TF-IDF + SVM ──────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words=ALL_STOPWORDS,
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

svm = LinearSVC(max_iter=2000)
svm.fit(X_train_vec, y_train)
y_pred = svm.predict(X_test_vec)

acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n=== SVM Performance ===")
print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(report)

# ── Confusion Matrix ───────────────────────────────────────────────────────────
labels = ["Positif", "Negatif", "Netral"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title(f"Confusion Matrix — SVM\nAccuracy: {acc*100:.1f}%", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual (Transformer Label)")
plt.tight_layout()
plt.savefig("output/figures/confusion_matrix_svm.png", dpi=150)
plt.close()

# ── Bar Chart SVM vs Transformer avg confidence ────────────────────────────────
avg_conf = df["confidence"].mean()

fig, ax = plt.subplots(figsize=(6, 4))
model_names = ["SVM\n(vs Transformer label)", "Transformer\n(avg confidence)"]
scores      = [acc, avg_conf]
colors      = ["#4C72B0", "#DD8452"]
bars = ax.bar(model_names, scores, color=colors, width=0.4, edgecolor="white")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("SVM Accuracy vs Transformer Avg Confidence")
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015, f"{score:.3f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("output/figures/model_comparison.png", dpi=150)
plt.close()

# ── Save report ────────────────────────────────────────────────────────────────
with open("output/svm_report.txt", "w", encoding="utf-8") as f:
    f.write(f"SVM Accuracy: {acc:.4f}\n\n")
    f.write(report)
    f.write(f"\nTransformer avg confidence: {avg_conf:.4f}\n")

print("File tersimpan:")
print("  output/svm_report.txt")
print("  output/figures/confusion_matrix_svm.png")
print("  output/figures/model_comparison.png")
