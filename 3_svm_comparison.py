"""
Step 3: Train SVM menggunakan label dari auto-labeling.
TF-IDF (unigram+bigram, stopwords EN+ID) + LinearSVC
Evaluasi: 5-Fold Stratified Cross Validation + final 80/20 split

Output: output/svm_report.txt
        output/figures/confusion_matrix_svm.png
        output/figures/model_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

X = df["text_clean"].astype(str)
y = df["label"]

# ── Pipeline TF-IDF + SVM ──────────────────────────────────────────────────────
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=ALL_STOPWORDS,
    )),
    ("svm", LinearSVC(max_iter=2000)),
])

# ── 5-Fold Stratified Cross Validation ────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_pipeline, X, y, cv=skf, scoring="accuracy")

print(f"\n=== 5-Fold Cross Validation ===")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}  ({score*100:.1f}%)")
print(f"  -----------------------------")
print(f"  Mean : {cv_scores.mean():.4f}  ({cv_scores.mean()*100:.1f}%)")
print(f"  Std  : {cv_scores.std():.4f}")

# ── Final model 80/20 untuk confusion matrix ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)

acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n=== Final Model (80/20 split) ===")
print(f"Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
print(report)

# ── Confusion Matrix ───────────────────────────────────────────────────────────
labels = ["Positif", "Negatif", "Netral"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title(
    f"Confusion Matrix — SVM\n"
    f"80/20 Acc: {acc*100:.1f}%  |  CV Mean: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%",
    fontsize=12
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual (Transformer Label)")
plt.tight_layout()
plt.savefig("output/figures/confusion_matrix_svm.png", dpi=150)
plt.close()

# ── Bar Chart: CV per-fold + mean ─────────────────────────────────────────────
avg_conf = df["confidence"].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Kiri: per-fold accuracy
fold_labels = [f"Fold {i}" for i in range(1, 6)] + ["Mean"]
fold_scores = list(cv_scores) + [cv_scores.mean()]
fold_colors = ["#7EB6D9"] * 5 + ["#4C72B0"]
bars = axes[0].bar(fold_labels, fold_scores, color=fold_colors, edgecolor="white")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("5-Fold Cross Validation — SVM")
axes[0].axhline(cv_scores.mean(), color="#4C72B0", linewidth=1.5,
                linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
axes[0].legend(fontsize=10)
for bar, score in zip(bars, fold_scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.012, f"{score:.3f}",
                 ha="center", fontsize=10)

# Kanan: SVM CV mean vs Transformer avg confidence
model_names = ["SVM\n(CV Mean ± Std)", "Transformer\n(avg confidence)"]
scores      = [cv_scores.mean(), avg_conf]
colors      = ["#4C72B0", "#DD8452"]
bars2 = axes[1].bar(model_names, scores, color=colors, width=0.4, edgecolor="white")
axes[1].errorbar(0, cv_scores.mean(), yerr=cv_scores.std(),
                 fmt="none", color="black", capsize=6, linewidth=2)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Score")
axes[1].set_title("SVM CV Mean vs Transformer Avg Confidence")
for bar, score in zip(bars2, scores):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.015, f"{score:.3f}",
                 ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("output/figures/model_comparison.png", dpi=150)
plt.close()

# ── Save report ────────────────────────────────────────────────────────────────
with open("output/svm_report.txt", "w", encoding="utf-8") as f:
    f.write("=== 5-Fold Stratified Cross Validation ===\n")
    for i, score in enumerate(cv_scores, 1):
        f.write(f"  Fold {i}: {score:.4f}\n")
    f.write(f"  Mean : {cv_scores.mean():.4f}\n")
    f.write(f"  Std  : {cv_scores.std():.4f}\n\n")
    f.write("=== Final Model (80/20 split) ===\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)
    f.write(f"\nTransformer avg confidence: {avg_conf:.4f}\n")

print("File tersimpan:")
print("  output/svm_report.txt")
print("  output/figures/confusion_matrix_svm.png")
print("  output/figures/model_comparison.png")
