# # Member 2: Signal Processing & Windowing
# 
# Task 3: Signal Preprocessing - Apply filtering / noise reduction; handle missing data via interpolation; “before vs after” plots; short narrative on why preprocessing matters.
# 
# Task 4: Windowing Strategies - Segment signals. Choose one: Fixed/Sliding windows—test ≥10 window sizes and justify choice; Event-based/Dynamic windows—propose and justify segmentation.
import os
from pathlib import Path

from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier

# 1. SETUP
# ---------------------------------------------------------
dataset_root = "A_DeviceMotion_data"
fs = 50.0

# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a low-pass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def create_windows(data, time_steps, step: int):
    """
    Segments data into sliding windows.
    Returns: Windows (X), Labels (y), and Subject IDs (sub)
    """
    # Columns to exclude from features (Label and Subject ID)
    meta_cols = ['label', 'subject_id']
    feature_cols = [c for c in data.columns if c not in meta_cols]

    segments = []
    labels = []
    subjects = []

    # Slide the window
    for i in range(0, len(data) - time_steps, step):
        # 1. Grab Sensor Data
        window_data = data[feature_cols].iloc[i: i + time_steps].values

        # 2. Grab Label (Mode of the window to be safe)
        window_label = data['label'].iloc[i: i + time_steps].mode()[0]

        # 3. Grab Subject ID (Should be constant, but mode works effectively)
        window_sub = data['subject_id'].iloc[i: i + time_steps].mode()[0]

        segments.append(window_data)
        labels.append(window_label)
        subjects.append(window_sub)

    return np.array(segments), np.array(labels), np.array(subjects)

# 3. TASK 3: PREPROCESSING VISUALIZATION
# ---------------------------------------------------------
# (Kept brief for this version)
# 4. MAIN PROCESSING LOOP
# ---------------------------------------------------------
def extract_features(WINDOW_SIZE: int, STEP_SIZE: int, output_csv: Path):
    """
    :param WINDOW_SIZE: Window size, in samples
    :param STEP_SIZE: Step size to control overlap. Subtract from WINDOW_SIZE to find overlap.
    :param output_csv: Where to store features CSV file
    :return:
    """
    if STEP_SIZE > WINDOW_SIZE:
        raise ValueError(f"Step size {STEP_SIZE} cannot be greater than window size {WINDOW_SIZE}.")
    print("--- Task 3: Preprocessing Checks ---")
    print("\n--- Processing Dataset (Extracting Labels AND IDs) ---")
    all_X = []
    all_y = []
    all_sub = [] # New list for subject IDs

    cols_to_filter = [
        'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z'
    ]

    for folder in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder)
        if os.path.isdir(folder_path):

            # A. Extract Label from Folder (e.g. "dws_1" -> "dws")
            label = folder.split("_")[0]

            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    # B. Extract Subject ID from File (e.g. "sub_1.csv" -> 1)
                    # Split by '_', then take the second part, then remove '.csv'
                    try:
                        subject_id = int(file.split('_')[1].split('.')[0])
                    except:
                        subject_id = 0 # Fallback if naming is weird

                    # Load
                    df = pd.read_csv(os.path.join(folder_path, file))
                    if 'Unnamed: 0' in df.columns: df.drop(columns=['Unnamed: 0'], inplace=True)

                    # Preprocess
                    df = df.interpolate(method='linear')
                    for c in cols_to_filter:
                        df[c] = butter_lowpass_filter(df[c].values, 5.0, fs)

                    # Prepare DataFrame for Windowing
                    df_clean = df[cols_to_filter].copy()
                    df_clean['label'] = label
                    df_clean['subject_id'] = subject_id  # <--- NEW: Add ID to dataframe

                    # Windowing
                    X_chunk, y_chunk, sub_chunk = create_windows(df_clean, WINDOW_SIZE, STEP_SIZE)

                    if len(X_chunk) > 0:
                        all_X.append(X_chunk)
                        all_y.append(y_chunk)
                        all_sub.append(sub_chunk)

    # Stack and Save
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_sub = np.concatenate(all_sub, axis=0) # Stack subjects

    output_X_path = 'X_windows.npy'
    output_y_path = 'y_labels.npy'
    output_sub_path = 'subject_ids.npy' # <--- NEW FILE

    np.save(output_X_path, final_X)
    np.save(output_y_path, final_y)
    np.save(output_sub_path, final_sub)

    print(f"Dataset processed.")
    print(f"X saved: {final_X.shape}")
    print(f"y saved: {final_y.shape}")
    print(f"IDs saved: {final_sub.shape} (Saved to subject_ids.npy)")

    # # Member 3: Feature Engineering & Analysis
    # Task 5: Feature Extraction & Analysis - Time-domain (mean, SD, RMS, ZCR); frequency-domain (dominant freq, spectral energy, spectral entropy); visualize feature importance (bar chart / ranking); concise interpretation.
    # 1. LOAD DATA (Now includes Subject IDs)
    # ---------------------------------------------------------
    print("Loading data from Member 2...")
    try:
        X_windows = np.load('X_windows.npy')
        y_labels = np.load('y_labels.npy')
        subject_ids = np.load('subject_ids.npy') # <--- NEW LOAD
        print(f"Loaded X: {X_windows.shape}, y: {y_labels.shape}, sub: {subject_ids.shape}")
    except FileNotFoundError:
        print("Error: Member 2 output files not found. Run Member 2 code first!")
        exit()

    channel_names = [
        'userAccel_x', 'userAccel_y', 'userAccel_z',
        'rotRate_x', 'rotRate_y', 'rotRate_z'
    ]

    # 2. FEATURE ENGINEERING FUNCTIONS
    # ---------------------------------------------------------
    def calculate_zcr(x):
        x_centered = x - np.mean(x)
        return ((x_centered[:-1] * x_centered[1:]) < 0).sum() / len(x)

    def get_time_features(x, prefix):
        return {
            f"{prefix}_mean": np.mean(x),
            f"{prefix}_std": np.std(x),
            f"{prefix}_rms": np.sqrt(np.mean(x**2)),
            f"{prefix}_zcr": calculate_zcr(x),
            # f"{prefix}_max": np.max(x), # Optional: reduce features if too slow
            # f"{prefix}_min": np.min(x)
        }

    def get_freq_features(x, prefix, fs):
        n = len(x)
        yf = np.abs(rfft(x))
        xf = rfftfreq(n, 1/fs)
        psd = yf**2
        total_energy = np.sum(psd)
        psd_norm = psd / (total_energy + 1e-12)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        dom_freq = xf[np.argmax(psd)]
        return {
            f"{prefix}_energy": total_energy,
            f"{prefix}_entropy": spec_entropy,
            f"{prefix}_dom_freq": dom_freq
        }

    # 3. EXTRACTION LOOP
    # ---------------------------------------------------------
    print("Starting Feature Extraction...")
    feature_rows = []

    for i in range(len(X_windows)):
        window = X_windows[i]
        row_features = {}

        for col_idx, col_name in enumerate(channel_names):
            signal_data = window[:, col_idx]
            row_features.update(get_time_features(signal_data, col_name))
            row_features.update(get_freq_features(signal_data, col_name, fs))

        feature_rows.append(row_features)

    df_features = pd.DataFrame(feature_rows)
    print(f"Feature Extraction Complete. Matrix Shape: {df_features.shape}")

    # 4. TASK 5: VIZ (Brief check)
    # ---------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(df_features, y_labels)
    importances = pd.Series(rf.feature_importances_, index=df_features.columns)
    top_features = importances.nlargest(10)

    plt.figure(figsize=(8, 4))
    top_features.sort_values().plot(kind='barh', color='teal')
    plt.title("Top Features")
    plt.show()

    # 5. SAVE FOR MEMBER 4 & 5 (Features + Label + ID)
    # ---------------------------------------------------------
    # Add the required columns
    df_features['label'] = y_labels
    df_features['subject_id'] = subject_ids  # <--- NEW: Added for Member 5

    df_features.to_csv(output_csv, index=False)

    print(f"✅ Success!")
    print(f"File saved to: {output_csv}")
    print(f"Columns included: {list(df_features.columns)[-3:]} ...")
    # Should print [..., 'label', 'subject_id']

def main():
    extract_features(WINDOW_SIZE=100, STEP_SIZE=50, output_csv=Path('features_df.csv'))

if __name__ == "__main__":
    main()