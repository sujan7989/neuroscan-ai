"""
Retrain ALL NeuroScan AI models with current sklearn version.
Saves to both ml/artifacts/ and server/ml_artifacts/
Run: python retrain_all_fresh.py
"""
import warnings, json, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV

ROOT     = Path(__file__).parent
ML_ART   = ROOT / "ml" / "artifacts"
SRV_ART  = ROOT / "server" / "ml_artifacts"
DATA     = ROOT / "ml" / "data"

ML_ART.mkdir(parents=True, exist_ok=True)
SRV_ART.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("NeuroScan AI — Fresh Model Training (sklearn", end=" ")
import sklearn; print(sklearn.__version__ + ")")
print("=" * 60)

# ── Helpers ──────────────────────────────────────────────────────

def metrics(yte, ypred, yprob):
    cm = confusion_matrix(yte, ypred)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2,2) else (0,0,0,0))
    return {
        "accuracy":    round(accuracy_score(yte, ypred), 4),
        "precision":   round(precision_score(yte, ypred, zero_division=0), 4),
        "recall":      round(recall_score(yte, ypred, zero_division=0), 4),
        "f1":          round(f1_score(yte, ypred, zero_division=0), 4),
        "specificity": round(tn / max(1, tn+fp), 4),
        "roc_auc":     round(roc_auc_score(yte, yprob), 4),
        "confusion_matrix": {"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)},
    }

def train_and_save(disorder, X, y, feature_names, dataset_note):
    print(f"\n[{disorder.upper()}] N={len(y)} pos={y.sum()} neg={(y==0).sum()} features={X.shape[1]}")

    sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=42)
    tr, te = next(sss.split(X, y))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    models = {
        "lr": Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "rf": Pipeline([("sc", StandardScaler()),
                        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                        min_samples_leaf=1, random_state=42, n_jobs=-1))]),
        "gb": Pipeline([("sc", StandardScaler()),
                        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                            learning_rate=0.05, subsample=0.8,
                                                            random_state=42))]),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}

    for mkey, pipe in models.items():
        cv_f1  = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1", n_jobs=-1)
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        yp    = pipe.predict(Xte)
        yprob = pipe.predict_proba(Xte)[:, 1]
        m = metrics(yte, yp, yprob)
        results[mkey] = {
            "cv_f1_mean": round(cv_f1.mean(), 4),  "cv_f1_std": round(cv_f1.std(), 4),
            "cv_auc_mean": round(cv_auc.mean(), 4), "cv_auc_std": round(cv_auc.std(), 4),
            **m
        }
        print(f"  {mkey}: CV_F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}  "
              f"Prec={m['precision']}  Rec={m['recall']}  "
              f"F1={m['f1']}  Spec={m['specificity']}  AUC={m['roc_auc']}")

        # Save to both artifact dirs
        for art_dir in [ML_ART, SRV_ART]:
            joblib.dump(pipe, art_dir / f"{disorder}_{mkey}.joblib", compress=3)

        # Save named aliases for predictor.py (ASD only)
        if disorder == "asd":
            label_map = {"lr": "logistic_regression", "rf": "random_forest", "gb": "gradient_boosting"}
            joblib.dump(pipe, ML_ART / f"{label_map[mkey]}.joblib", compress=3)

    # Save feature list
    feat_meta = {"features": feature_names, "n_features": len(feature_names), "disorder": disorder,
                 "dataset_note": dataset_note, "sklearn_version": sklearn.__version__}
    for art_dir in [ML_ART, SRV_ART]:
        (art_dir / f"{disorder}_features.json").write_text(json.dumps(feat_meta, indent=2))

    # Save metrics JSON
    metrics_out = {"disorder": disorder, "results": results, "n_test": len(te),
                   "dataset_note": dataset_note, "sklearn_version": sklearn.__version__}
    for art_dir in [ML_ART, SRV_ART]:
        (art_dir / f"{disorder}_metrics.json").write_text(json.dumps(metrics_out, indent=2))

    return results


# ════════════════════════════════════════════════════════════════
# 1. ASD — UCI AQ-10 + behavioral proxies (22 features)
# ════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA / "asd_uci_combined.csv")
df["gender_num"]   = df["gender"].map({"m":1,"f":0,"male":1,"female":0}).fillna(0)
df["jaundice_num"] = df["jaundice"].map({"yes":1,"no":0,"Yes":1,"No":0}).fillna(0)
df["austim_num"]   = df["austim"].map({"yes":1,"no":0,"Yes":1,"No":0}).fillna(0)
# Behavioral proxies derived from AQ-10 items (SSBD-correlated)
df["arm_flapping"]   = ((df["A1_Score"] + df["A8_Score"]) / 2).round().clip(0,1)
df["head_banging"]   = (df["A7_Score"] * 0.5).round().clip(0,1)
df["spinning"]       = (df["A8_Score"] * 0.6).round().clip(0,1)
df["rocking"]        = ((df["A3_Score"] + df["A8_Score"]) / 2).round().clip(0,1)
df["gaze_avoidance"] = df["A1_Score"].clip(0,1)
df["echolalia"]      = (df["A5_Score"] * 0.7).round().clip(0,1)
df["hypo_gaze"]      = df["A6_Score"].clip(0,1)
df["social_isolate"] = df["A10_Score"].clip(0,1)

ASD_FEATURES = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
                "age","gender_num","jaundice_num","austim_num",
                "arm_flapping","head_banging","spinning","rocking","gaze_avoidance","echolalia","hypo_gaze","social_isolate"]
X_asd = df[ASD_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
y_asd = df["Class_ASD"].astype(int).values
asd_res = train_and_save("asd", X_asd, y_asd, ASD_FEATURES, "UCI ASD Screening 989 real records (Adults+Children)")

# Save scaler for predictor.py
sc_asd = StandardScaler()
sc_asd.fit(X_asd)
joblib.dump(sc_asd, ML_ART / "scaler.joblib", compress=3)

# ════════════════════════════════════════════════════════════════
# 2. ADHD — WHO ASRS + CPT-3 + Conners
# ════════════════════════════════════════════════════════════════
df2 = pd.read_csv(DATA / "adhd_dataset_v2.csv")
ADHD_FEATURES = ["asrs_inatt_1","asrs_inatt_2","asrs_inatt_3","asrs_inatt_4","asrs_inatt_5","asrs_inatt_6",
                  "asrs_hyper_1","asrs_hyper_2","asrs_hyper_3","asrs_hyper_4","asrs_hyper_5","asrs_hyper_6",
                  "conners_inattention","conners_hyperactivity","conners_executive","conners_learning",
                  "cpt_omission_tscore","cpt_commission_tscore","cpt_rt_mean_ms","cpt_rt_variability","cpt_detectability",
                  "family_history_adhd","sleep_problems","school_performance_low","age"]
ADHD_FEATURES = [c for c in ADHD_FEATURES if c in df2.columns]
X_adhd = df2[ADHD_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
y_adhd = df2["label"].astype(int).values
train_and_save("adhd", X_adhd, y_adhd, ADHD_FEATURES, "WHO ASRS + CPT-3 + Conners 3100 records")

# ════════════════════════════════════════════════════════════════
# 3. DYSLEXIA — Phonological + handwriting features
# ════════════════════════════════════════════════════════════════
df3 = pd.read_csv(DATA / "dyslexia_dataset_v2.csv")
DYS_FEATURES = ["phonological_awareness","phonological_memory","rapid_naming_speed",
                 "word_reading_fluency","nonword_reading","letter_reversal",
                 "reading_avoidance","spelling_errors","slow_reading_rate","family_history_dyslexia",
                 "hw_pen_speed_mms","hw_stroke_consistency","hw_letter_reversal_rate",
                 "hw_line_deviation_mm","hw_spacing_variability","hw_erasure_per_word"]
DYS_FEATURES = [c for c in DYS_FEATURES if c in df3.columns]
X_dys = df3[DYS_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
y_dys = df3["label"].astype(int).values
train_and_save("dyslexia", X_dys, y_dys, DYS_FEATURES, "IDA/Yale phonological + MDPI handwriting kinematics 3100 records")

# ════════════════════════════════════════════════════════════════
# 4. STUTTERING — SEP-28k/UCLASS acoustic features
# ════════════════════════════════════════════════════════════════
df4 = pd.read_csv(DATA / "stuttering_dataset_v2.csv")
STU_FEATURES = ["disfluency_rate","repetition_per_100w","prolongation_dur_s","block_freq_per_100w",
                 "pause_freq_per_100w","speech_rate_wpm","phonation_ratio","jitter_pct","shimmer_db","hnr_db",
                 "age","onset_childhood","family_history_stutter","anxiety_speaking"]
STU_FEATURES = [c for c in STU_FEATURES if c in df4.columns]
X_stu = df4[STU_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
y_stu = df4["label"].astype(int).values
train_and_save("stuttering", X_stu, y_stu, STU_FEATURES, "SEP-28k/UCLASS acoustic norms 3100 records")

# ════════════════════════════════════════════════════════════════
# 5. SSD — GFTA-3 + DEAP articulation
# ════════════════════════════════════════════════════════════════
df5 = pd.read_csv(DATA / "ssd_dataset_v2.csv")
SSD_FEATURES = ["intelligibility_score","pcc_score","gfta3_standard_score","fricative_error_rate",
                 "stop_error_rate","liquid_error_rate","affricate_error_rate","cluster_error_rate",
                 "late_sound_acquisition","f1_formant_deviation","f2_formant_deviation",
                 "prosody_deviation","hearing_normal","age","family_history_ssd"]
SSD_FEATURES = [c for c in SSD_FEATURES if c in df5.columns]
X_ssd = df5[SSD_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
y_ssd = df5["label"].astype(int).values
train_and_save("ssd", X_ssd, y_ssd, SSD_FEATURES, "GFTA-3 DEAP articulation norms 3100 records")

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL MODELS RETRAINED SUCCESSFULLY")
print("=" * 60)
print(f"Saved to: {ML_ART}")
print(f"Saved to: {SRV_ART}")
for f in sorted(ML_ART.glob("*_rf.joblib")):
    m = joblib.load(f)
    n = m.named_steps["sc"].n_features_in_
    print(f"  {f.name}: {n} features")
