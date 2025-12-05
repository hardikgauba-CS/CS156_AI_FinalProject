import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def run_modeling(X_train: DataFrame, X_test: DataFrame, y_train: np.ndarray, y_test: np.ndarray,
                 label_encoder: LabelEncoder, output_dir: Path):
    """
    :param csv: Path to features csv. Should include label and subject_id columns
    :param output_dir: Output dir for all predictions from trained models
    """

    #%% md
    # ## Non-Deep Learning Model Training
    # Using Scikit-learn:
    # - Decision Tree, SVM, Naive Bayes, RF, AdaBoost, XGBoost (2.0)
    #
    # Note: to run XGBoost on macOS: `brew install libomp`
    #%%


    # Decision Tree
    print("Training Decision Tree")
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)  # y_train is already a NumPy array of integers

    # Support Vector Machine (SVM)
    print("Training SVM")
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train, y_train)

    # Naive Bayes
    print("Training Naive Bayes")
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)

    # Random Forest
    print("Training Random Forest")
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train)

    # AdaBoost
    print("Training AdaBoost")
    ada_clf = AdaBoostClassifier(random_state=42)
    ada_clf.fit(X_train, y_train)

    # XGBoost
    print("Training XGBoost")
    xgb_clf = xgb.XGBClassifier(random_state=42)
    xgb_clf.fit(X_train, y_train)
    #%% md
    # ## Deep Learning Model Training
    # Using Keras/Tensorflow:
    # - CNN, RNN
    #%%


    # normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # reshape data for 1D CNN and RNN.
    # CNN expects input shape: (samples, steps, channels)
    # features will be treated as the steps
    X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # RNN (LSTM) expects input shape: (samples, timesteps, features)
    # feature vectors will be treated as a single timestep
    X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    n_features = X_train.shape[1]
    n_classes = len(label_encoder.classes_)

    # 1D Convolutional Neural Network (CNN)
    print("\nTraining CNN")
    cnn_model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(n_features, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])

    cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    # Recurrent Neural Network (RNN) with LSTM
    print("\nTraining RNN (LSTM)")
    rnn_model = Sequential([
        LSTM(64, input_shape=(1, n_features)),
        Dense(100, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])

    rnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    #%% md
    # ## Generate predictions
    #%%


    output_dir.mkdir(parents=True, exist_ok=True)

    # Non-Deep Learning Predictions

    print("Generating predictions for non-deep learning models...")

    non_dl_models = {
        "dt": dt_clf,
        "svm": svm_clf,
        "nb": nb_clf,
        "rf": rf_clf,
        "ada": ada_clf,
        "xgb": xgb_clf
    }

    for name, model in non_dl_models.items():
        print(f"  - {name}")
        y_pred_encoded = model.predict(X_test)

        predictions_df = pd.DataFrame({
            "actual_encoded": y_test,
            "predicted_encoded": y_pred_encoded
        })

        # decode integer labels back to original string labels
        predictions_df["actual_label"] = label_encoder.inverse_transform(predictions_df["actual_encoded"])
        predictions_df["predicted_label"] = label_encoder.inverse_transform(predictions_df["predicted_encoded"])

        predictions_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    # Deep Learning Predictions

    print("\nGenerating predictions for deep learning models...")

    dl_models = {
        "cnn": (cnn_model, X_test_cnn),
        "rnn": (rnn_model, X_test_rnn)
    }

    for name, (model, data) in dl_models.items():
        print(f"  - {name}")
        y_pred_prob = model.predict(data)
        y_pred_encoded = np.argmax(y_pred_prob, axis=1)

        predictions_df = pd.DataFrame({
            "actual_encoded": y_test,
            "predicted_encoded": y_pred_encoded,
            "actual_label": label_encoder.inverse_transform(y_test),
            "predicted_label": label_encoder.inverse_transform(y_pred_encoded)
        })
        predictions_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    print(f"\nAll predictions saved to the '{output_dir}' directory.")

def main():
    df = pd.read_csv(Path("features_df.csv"))
    X_all: DataFrame = df.drop(columns=["label", "subject_id"])
    y_all: np.ndarray = df["label"].to_numpy()

    # convert string labels ('dws' 'jog' 'sit' 'std' 'ups' 'wlk') to integers (0, 1, etc.).
    label_encoder = LabelEncoder()
    y_all_encoded = label_encoder.fit_transform(y_all.ravel())

    print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_all, y_all_encoded, test_size=0.2,
                                                                                random_state=42)

    run_modeling(X_train, X_test, y_train, y_test, label_encoder, Path("predictions"))

if __name__ == "__main__":
    main()