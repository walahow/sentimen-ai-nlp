"""
Step 3: Train SVM menggunakan label dari auto-labeling.
Bandingkan performa SVM vs Transformer (hasil auto-label = ground truth transformer).
Output: output/svm_report.txt, output/figures/confusion_matrix_svm.png
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

X = df["text_clean"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF + SVM ---
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

svm = LinearSVC(max_iter=2000)
svm.fit(X_train_vec, y_train)
y_pred_svm = svm.predict(X_test_vec)

svm_acc = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm)

# --- Transformer "accuracy" pada test set ---
# Transformer sudah label semua data; pada test set kita ambil confidence sebagai proxy
# Karena transformer = label sumber, akurasi-nya dihitung sebagai seberapa konsisten
# prediksi SVM cocok dengan transformer label
transformer_acc_on_test = accuracy_score(y_test, y_test)  # baseline: transformer = 100% pada label sendiri

print("\n=== SVM Performance ===")
print(f"Accuracy: {svm_acc:.4f}")
print(svm_report)

# --- Confusion Matrix SVM ---
labels = ["Positif", "Negatif", "Netral"]
cm = confusion_matrix(y_test, y_pred_svm, labels=labels)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title("Confusion Matrix — SVM")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual (Transformer Label)")
plt.tight_layout()
plt.savefig("output/figures/confusion_matrix_svm.png", dpi=150)
plt.close()

# --- Perbandingan Bar Chart ---
avg_transformer_conf = df["confidence"].mean()

fig, ax = plt.subplots(figsize=(6, 4))
models  = ["SVM\n(vs Transformer label)", "Transformer\n(avg confidence)"]
scores  = [svm_acc, avg_transformer_conf]
colors  = ["#4C72B0", "#DD8452"]
bars = ax.bar(models, scores, color=colors, width=0.4)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("SVM Accuracy vs Transformer Avg Confidence")
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02, f"{score:.3f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("output/figures/model_comparison.png", dpi=150)
plt.close()

# --- Save report ---
with open("output/svm_report.txt", "w", encoding="utf-8") as f:
    f.write(f"SVM Accuracy: {svm_acc:.4f}\n\n")
    f.write(svm_report)
    f.write(f"\nTransformer avg confidence: {avg_transformer_conf:.4f}\n")

print("\nFile tersimpan:")
print("  output/svm_report.txt")
print("  output/figures/confusion_matrix_svm.png")
print("  output/figures/model_comparison.png")
