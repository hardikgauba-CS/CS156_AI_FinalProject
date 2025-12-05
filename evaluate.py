# ============================================================
# LOAD REQUIRED LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# SET PATH TO PREDICTIONS FOLDER
# ============================================================
base_path = "predictions/"  # <= change if needed

files = {
    "Decision Tree": base_path + "dt_predictions.csv",
    "SVM": base_path + "svm_predictions.csv",
    "Naive Bayes": base_path + "nb_predictions.csv",
    "Random Forest": base_path + "rf_predictions.csv",
    "AdaBoost": base_path + "ada_predictions.csv",
    "XGBoost": base_path + "xgb_predictions.csv",
    "CNN": base_path + "cnn_predictions.csv",
    "RNN": base_path + "rnn_predictions.csv"
}

# ============================================================
# METRICS TABLE
# ============================================================
results = []

for model_name, file_path in files.items():
    df = pd.read_csv(file_path)

    y_true = df["actual_label"]
    y_pred = df["predicted_label"]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    results.append([model_name, acc, precision, recall, f1])

metrics_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\n===== METRICS TABLE =====")
print(metrics_df)

metrics_df.to_csv(base_path + "metrics_summary.csv", index=False)
print(f"\nSaved metrics table → {base_path}metrics_summary.csv")

# ============================================================
# CONFUSION MATRICES
# ============================================================
labels = sorted(df["actual_label"].unique())

for model_name, file_path in files.items():
    df = pd.read_csv(file_path)

    y_true = df["actual_label"]
    y_pred = df["predicted_label"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(base_path + f"{model_name.replace(' ', '_').lower()}_cm.png")
    plt.close()

# ============================================================
# ERROR ANALYSIS
# ============================================================
for model_name, file_path in files.items():
    df = pd.read_csv(file_path)

    errors = df[df["actual_label"] != df["predicted_label"]]

    print("\n======================================")
    print(f"ERROR ANALYSIS → {model_name}")
    print("======================================")
    print(errors.head())

    errors.to_csv(base_path + f"{model_name.replace(' ', '_').lower()}_errors.csv", index=False)
