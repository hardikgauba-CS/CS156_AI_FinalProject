# ============================================================
# LOSO EVALUATION (Leave-One-Subject-Out)
# Member 5 – Task 7 (Advanced Modeling & Evaluation)
# ============================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# LOAD FEATURES
# ============================================================
df = pd.read_csv("features_df.csv")

X_all = df.drop(columns=["label", "subject_id"])
y_all = df["label"]
subjects = df["subject_id"]

label_encoder = LabelEncoder()
y_all_encoded = label_encoder.fit_transform(y_all)

unique_subjects = sorted(df["subject_id"].unique())

output_dir = Path("predictions_loso")
output_dir.mkdir(exist_ok=True)


# ============================================================
# MODELS USED
# ============================================================
models = {
    "dt": DecisionTreeClassifier(random_state=42),
    "svm": SVC(probability=False, random_state=42),
    "nb": GaussianNB(),
    "rf": RandomForestClassifier(random_state=42),
    "ada": AdaBoostClassifier(random_state=42),
    "xgb": xgb.XGBClassifier(random_state=42)
}

# Store LOSO results
loso_results = []

# ============================================================
# LOSO LOOP
# ============================================================
for test_sub in unique_subjects:
    print(f"\n===== LOSO: Leaving Out Subject {test_sub} =====")

    # Split data
    X_train = X_all[subjects != test_sub]
    y_train = y_all_encoded[subjects != test_sub]

    X_test = X_all[subjects == test_sub]
    y_test = y_all_encoded[subjects == test_sub]

    subject_results = {"subject": test_sub}

    # Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        subject_results[name] = acc

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix – LOSO Subject {test_sub} – {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(output_dir / f"sub{test_sub}_{name}_cm.png")
        plt.close()

        # Save error rows
        errors = pd.DataFrame({
            "actual": label_encoder.inverse_transform(y_test),
            "predicted": label_encoder.inverse_transform(y_pred)
        })
        errors = errors[errors["actual"] != errors["predicted"]]
        errors.to_csv(output_dir / f"sub{test_sub}_{name}_errors.csv", index=False)

    loso_results.append(subject_results)


# ============================================================
# SAVE OVERALL LOSO TABLE
# ============================================================
loso_df = pd.DataFrame(loso_results)
loso_df.to_csv(output_dir / "loso_summary.csv", index=False)

print("\n===== LOSO COMPLETE =====")
print(loso_df)
print(f"\nSaved LOSO summary → {output_dir}/loso_summary.csv")