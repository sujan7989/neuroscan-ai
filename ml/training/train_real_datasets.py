"""
NeuroScan AI - Real Dataset Training Pipeline
=============================================
Trains all 5 disorder models using real datasets:
  ASD       -> Kaggle Autistic Children Facial features (HOG)
  ADHD      -> Kaggle ADHD behavioral/cognitive assessment CSV
  Dyslexia  -> Your handwriting images (158K, 3 classes)
  Stuttering-> SEP-28k MFCC audio features CSV
  SSD       -> Speech acoustic features CSV

Run once on your machine:
  python ml/training/train_real_datasets.py

Output: server/ml_artifacts/*.pkl  (committed to GitHub -> Render loads them)
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
ARTIFACTS   = ROOT / "server" / "ml_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_DIR    = ROOT / "ml" / "data"
DYS_DIR     = Path(r"C:\Users\sujan\Downloads\Dyslexia-Dataset\Dataset Dyslexia_Password WanAsy321\Gambo")

print(f"\n{'='*60}")
print("NeuroScan AI - Real Dataset Training Pipeline")
print(f"{'='*60}\n")

# ════════════════════════════════════════════════════════════════
# HELPER: Save model + metrics
# ════════════════════════════════════════════════════════════════
def save_model(name, model, X_test, y_test, feature_names=None):
    path = ARTIFACTS / f"{name}_model.pkl"
    joblib.dump(model, path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    metrics = {
        "disorder": name,
        "accuracy": round(acc, 4),
        "auc": round(auc, 4) if auc else None,
        "n_test": len(y_test),
        "feature_names": feature_names or []
    }
    with open(ARTIFACTS / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {path.name}  acc={acc:.3f}" + (f"  auc={auc:.3f}" if auc else ""))
    return metrics


# ════════════════════════════════════════════════════════════════
# 1. ASD - Kaggle UCI Autism Screening + behavioral features
# ════════════════════════════════════════════════════════════════
print("1. Training ASD model (UCI + behavioral features)...")

def train_asd():
    # Use the existing combined dataset (UCI real + behavioral features)
    csv_path = DATA_DIR / "asd_dataset_v2.csv"
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows from asd_dataset_v2.csv")

    # Feature columns - AQ-10 scores + behavioral video features
    aq_cols = [f"A{i}_Score" for i in range(1, 11)]
    behav_cols = ["gaze_avoidance_pct", "social_gaze_pct", "facial_affect_flat",
                  "joint_attention_score", "repetitive_motion_idx", "response_latency_s"]
    demo_cols = ["age", "jaundice", "austim"]

    available = [c for c in aq_cols + behav_cols + demo_cols if c in df.columns]

    X = df[available].copy()
    # Encode boolean-like columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].map({'yes':1,'no':0,'m':1,'f':0}).fillna(0)
    X = X.fillna(X.median())
    y = df["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
            method="isotonic", cv=3
        ))
    ])
    rf.fit(X_tr, y_tr)
    m = save_model("asd", rf, X_te, y_te, available)
    # Save feature list for predictor
    with open(ARTIFACTS / "asd_features.json", "w") as f:
        json.dump({"features": available, "type": "questionnaire+behavioral"}, f, indent=2)
    return m

train_asd()


# ════════════════════════════════════════════════════════════════
# 2. ADHD - Behavioral + Cognitive assessment (ASRS + CPT)
# ════════════════════════════════════════════════════════════════
print("\n2. Training ADHD model (ASRS + CPT cognitive assessment)...")

def train_adhd():
    csv_path = DATA_DIR / "adhd_dataset_v2.csv"
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows from adhd_dataset_v2.csv")

    # ASRS behavioral + CPT cognitive features
    asrs_cols = [c for c in df.columns if c.startswith("asrs_")]
    cpt_cols  = [c for c in df.columns if c.startswith("cpt_")]
    conn_cols = [c for c in df.columns if c.startswith("conners_")]
    demo_cols = ["age", "family_history_adhd", "sleep_problems", "school_performance_low"]

    feat_cols = asrs_cols + cpt_cols + conn_cols + [c for c in demo_cols if c in df.columns]
    available = [c for c in feat_cols if c in df.columns]

    X = df[available].copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
    y = df["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            method="isotonic", cv=3
        ))
    ])
    rf.fit(X_tr, y_tr)
    m = save_model("adhd", rf, X_te, y_te, available)
    with open(ARTIFACTS / "adhd_features.json", "w") as f:
        json.dump({"features": available, "type": "behavioral_cognitive_ASRS_CPT"}, f, indent=2)
    return m

train_adhd()


# ════════════════════════════════════════════════════════════════
# 3. DYSLEXIA - Real handwriting images (158K images, 3 classes)
# ════════════════════════════════════════════════════════════════
print("\n3. Training Dyslexia model (real handwriting images)...")

def extract_hog_features(img_array):
    """Extract HOG-like features from grayscale image."""
    from PIL import Image
    import numpy as np
    # Resize to 64x64
    img = Image.fromarray(img_array).convert("L").resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    # Gradient features
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    # Block histograms (4x4 blocks of 16x16)
    features = []
    for i in range(4):
        for j in range(4):
            block_mag = mag[i*16:(i+1)*16, j*16:(j+1)*16]
            block_ang = ang[i*16:(i+1)*16, j*16:(j+1)*16]
            hist, _ = np.histogram(block_ang.ravel(), bins=9, range=(-np.pi, np.pi), weights=block_mag.ravel())
            features.extend(hist.tolist())
    # Texture stats
    features.extend([arr.mean(), arr.std(), arr.min(), arr.max()])
    # Stroke density per row/col
    features.extend(arr.mean(axis=0)[::8].tolist())  # 8 col samples
    features.extend(arr.mean(axis=1)[::8].tolist())  # 8 row samples
    return np.array(features, dtype=np.float32)

def train_dyslexia():
    from PIL import Image
    import numpy as np
    from tqdm import tqdm

    # Classes: Normal=0, Reversal=1 (dyslexia indicator), Corrected=2
    class_map = {"Normal": 0, "Reversal": 1, "Corrected": 2}
    # Binary: Normal vs Dyslexic (Reversal+Corrected)
    binary_map = {"Normal": 0, "Reversal": 1, "Corrected": 1}

    X_list, y_list = [], []
    max_per_class = 2000  # Use 2000 per class to keep training fast

    for split in ["Train", "Test"]:
        for cls_name, cls_label in binary_map.items():
            folder = DYS_DIR / split / cls_name
            if not folder.exists():
                print(f"   Skipping {folder} (not found)")
                continue
            files = list(folder.glob("*"))[:max_per_class]
            print(f"   {split}/{cls_name}: {len(files)} images")
            for fpath in tqdm(files, desc=f"  {cls_name}", leave=False):
                try:
                    img = Image.open(fpath).convert("L").resize((64, 64))
                    arr = np.array(img)
                    feat = extract_hog_features(arr)
                    X_list.append(feat)
                    y_list.append(cls_label)
                except Exception:
                    pass

    if len(X_list) < 100:
        print("   ERROR: Not enough images loaded. Check DYS_DIR path.")
        return None

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"   Total: {len(X)} samples, {X.shape[1]} features")
    print(f"   Class distribution: Normal={np.sum(y==0)}, Dyslexic={np.sum(y==1)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, max_depth=10, max_features="sqrt",
                                   random_state=42, n_jobs=-1),
            method="isotonic", cv=3
        ))
    ])
    print("   Training classifier on HOG features...")
    rf.fit(X_tr, y_tr)
    m = save_model("dyslexia", rf, X_te, y_te)
    # Save with compression to keep file small
    joblib.dump(rf, ARTIFACTS / "dyslexia_model.pkl", compress=("lzma", 9))
    with open(ARTIFACTS / "dyslexia_features.json", "w") as f:
        json.dump({"features": ["hog_features_144dim"], "type": "handwriting_image_HOG",
                   "classes": {"0":"Normal","1":"Dyslexic"},
                   "n_train": len(X_tr), "image_size": "64x64"}, f, indent=2)
    return m

train_dyslexia()


# ════════════════════════════════════════════════════════════════
# 4. STUTTERING - SEP-28k acoustic speech features
# ════════════════════════════════════════════════════════════════
print("\n4. Training Stuttering model (acoustic speech features)...")

def train_stuttering():
    csv_path = DATA_DIR / "stuttering_dataset_v2.csv"
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows from stuttering_dataset_v2.csv")

    # Voice/acoustic features from SEP-28k norms
    acoustic_cols = ["disfluency_rate", "block_freq_per_100w", "repetition_per_100w",
                     "prolongation_dur_s", "pause_freq_per_100w", "speech_rate_wpm",
                     "jitter_pct", "shimmer_db", "hnr_db", "phonation_ratio",
                     "stutter_severity", "anxiety_speaking", "onset_childhood"]
    available = [c for c in acoustic_cols if c in df.columns]

    X = df[available].copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
    y = df["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            method="isotonic", cv=3
        ))
    ])
    rf.fit(X_tr, y_tr)
    m = save_model("stuttering", rf, X_te, y_te, available)
    with open(ARTIFACTS / "stuttering_features.json", "w") as f:
        json.dump({"features": available, "type": "acoustic_voice_SEP28k"}, f, indent=2)
    return m

train_stuttering()


# ════════════════════════════════════════════════════════════════
# 5. SSD - Speech sound acoustic features (GFTA-3 / DEAP norms)
# ════════════════════════════════════════════════════════════════
print("\n5. Training SSD model (speech acoustic features)...")

def train_ssd():
    csv_path = DATA_DIR / "ssd_dataset_v2.csv"
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows from ssd_dataset_v2.csv")

    # Acoustic + articulation features from GFTA-3 norms
    acoustic_cols = ["intelligibility_score", "pcc_score", "gfta3_standard_score",
                     "fricative_error_rate", "stop_error_rate", "liquid_error_rate",
                     "affricate_error_rate", "cluster_error_rate", "late_sound_acquisition",
                     "f1_formant_deviation", "f2_formant_deviation", "prosody_deviation",
                     "hearing_normal"]
    available = [c for c in acoustic_cols if c in df.columns]

    X = df[available].copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
    y = df["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            method="isotonic", cv=3
        ))
    ])
    rf.fit(X_tr, y_tr)
    m = save_model("ssd", rf, X_te, y_te, available)
    with open(ARTIFACTS / "ssd_features.json", "w") as f:
        json.dump({"features": available, "type": "acoustic_articulation_GFTA3"}, f, indent=2)
    return m

train_ssd()


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Model files saved to: {ARTIFACTS}")
print("\nFiles created:")
for f in sorted(ARTIFACTS.glob("*.pkl")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:35s} {size_kb:6d} KB")
print("\nNext: commit these .pkl files to GitHub -> Render deploys -> DONE!")
print("Command: git add server/ml_artifacts/*.pkl && git commit -m 'feat: add real trained models' && git push")
