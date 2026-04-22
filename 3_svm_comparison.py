"""
Step 3: Train SVM menggunakan label dari auto-labeling.
Bandingkan performa SVM vs Transformer (hasil auto-label = ground truth transformer).

Hasil grid-search (11 konfigurasi):
  BEST accuracy : + stopwords EN+ID         -> 58.3%  (baseline 57.2%)
  BEST F1 macro : + balanced + stopwords    -> F1 0.55 (lebih fair antar kelas)

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
    classification_report, confusion_matrix, accuracy_score, f1_score
)

os.makedirs("output/figures", exist_ok=True)

df = pd.read_csv("output/data_labeled.csv", encoding="utf-8")
df = df.dropna(subset=["text_clean", "label"])
print(f"Data total: {len(df)} baris")
print(f"Label: {df['label'].value_counts().to_dict()}")
print(f"Confidence rata-rata: {df['confidence'].mean():.4f}\n")

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

# ── Split 80/20 (sama persis untuk kedua model) ────────────────────────────────
X = df["text_clean"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model A: Baseline SVM ──────────────────────────────────────────────────────
tfidf_base = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
svm_base   = LinearSVC(max_iter=2000)
svm_base.fit(tfidf_base.fit_transform(X_train), y_train)
y_pred_base = svm_base.predict(tfidf_base.transform(X_test))
acc_base = accuracy_score(y_test, y_pred_base)
f1_base  = f1_score(y_test, y_pred_base, average="macro")

print("=" * 55)
print(f"[BASELINE] Accuracy: {acc_base:.4f}  F1 macro: {f1_base:.4f}")
print(classification_report(y_test, y_pred_base))

# ── Model B: Improved SVM (stopwords, best accuracy) ──────────────────────────
tfidf_imp = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
    stop_words=ALL_STOPWORDS,
)
svm_imp = LinearSVC(max_iter=2000)
svm_imp.fit(tfidf_imp.fit_transform(X_train), y_train)
y_pred_imp = svm_imp.predict(tfidf_imp.transform(X_test))
acc_imp = accuracy_score(y_test, y_pred_imp)
f1_imp  = f1_score(y_test, y_pred_imp, average="macro")

delta_acc = acc_imp - acc_base
delta_f1  = f1_imp  - f1_base
print("=" * 55)
print(f"[IMPROVED] Accuracy: {acc_imp:.4f}  F1 macro: {f1_imp:.4f}")
print(f"           Delta acc: {delta_acc:+.4f}  Delta F1: {delta_f1:+.4f}")
print(f"           Perbaikan: stopword removal EN+ID")
print(classification_report(y_test, y_pred_imp))

# ── Confusion Matrix (Baseline vs Improved) ───────────────────────────────────
labels = ["Positif", "Negatif", "Netral"]
cm_base = confusion_matrix(y_test, y_pred_base, labels=labels)
cm_imp  = confusion_matrix(y_test, y_pred_imp,  labels=labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, cm, title in [
    (ax1, cm_base, f"Baseline SVM\nAccuracy {acc_base*100:.1f}%  |  F1 {f1_base:.3f}"),
    (ax2, cm_imp,  f"Improved SVM (+stopwords)\nAccuracy {acc_imp*100:.1f}%  |  F1 {f1_imp:.3f}"),
]:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (Transformer Label)")

plt.suptitle("Confusion Matrix — Baseline vs Improved SVM", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("output/figures/confusion_matrix_svm.png", dpi=150)
plt.close()

# ── Bar Chart Model Comparison ─────────────────────────────────────────────────
avg_conf = df["confidence"].mean()

fig, ax = plt.subplots(figsize=(7, 4))
model_names = ["SVM Baseline", "SVM Improved\n(+stopwords)", "Transformer\n(avg confidence)"]
scores      = [acc_base, acc_imp, avg_conf]
colors      = ["#4C72B0", "#2ca02c", "#DD8452"]
bars = ax.bar(model_names, scores, color=colors, width=0.45, edgecolor="white")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: SVM Baseline vs Improved vs Transformer")
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015, f"{score:.3f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("output/figures/model_comparison.png", dpi=150)
plt.close()

# ── Save report ───────────────────────────────────────────────────────────────
with open("output/svm_report.txt", "w", encoding="utf-8") as f:
    f.write("=== BASELINE SVM ===\n")
    f.write(f"Accuracy : {acc_base:.4f}\nF1 macro : {f1_base:.4f}\n\n")
    f.write(classification_report(y_test, y_pred_base))
    f.write("\n\n=== IMPROVED SVM (+stopwords EN+ID) ===\n")
    f.write(f"Accuracy : {acc_imp:.4f}  (delta: {delta_acc:+.4f})\n")
    f.write(f"F1 macro : {f1_imp:.4f}   (delta: {delta_f1:+.4f})\n\n")
    f.write(classification_report(y_test, y_pred_imp))
    f.write(f"\nTransformer avg confidence : {avg_conf:.4f}\n")

print("File tersimpan:")
print("  output/svm_report.txt")
print("  output/figures/confusion_matrix_svm.png")
print("  output/figures/model_comparison.png")
