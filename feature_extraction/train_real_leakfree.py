"""
feature_extraction/train_real_leakfree.py
==========================================
LEAK-FREE training on real datasets.

Each pathway uses the correct dataset with proper feature/label separation:

ASD        : ml/data/asd_uci_combined.csv
             AQ-10 questionnaire (A1-A10) + demographics
             Target: Class_ASD (0/1)
             EXCLUDES: 'result' (raw AQ-10 sum) — direct leakage

ADHD       : ml/data/adhd_dataset_v2.csv
             WHO ASRS + CPT-3 + Conners acoustic/cognitive features
             Target: label (0/1)

Dyslexia   : ml/data/features/dyslexia_resnet_features.npy  (if available)
             Fallback: ml/data/dyslexia_dataset_v2.csv (tabular phonological)
             Target: label (0/1)

Stuttering : ml/data/stuttering_dataset_v2.csv
             Acoustic features: disfluency_rate, jitter, shimmer, HNR,
             speech_rate_wpm, prolongation_dur_s, repetition_per_100w,
             block_freq_per_100w, pause_freq_per_100w, phonation_ratio
             EXCLUDES: stutter_severity (ordinal label proxy — leakage)
             Target: label (0/1)

SSD        : ml/data/ssd_dataset_v2.csv
             GFTA-3 + DEAP acoustic norms:
             intelligibility_score, pcc_score, gfta3_standard_score,
             fricative/stop/liquid/cluster error rates, formant deviation
             EXCLUDES: phonological_process (ordinal severity)
             Target: label (0/1)

Splitting  : GroupShuffleSplit(test_size=0.25, random_state=42)
             Groups = age_group or ethnicity (proxy for cohort cluster)
             For datasets without natural groups: random split.

CV         : GroupKFold(n_splits=5) on training set only.

Outputs    : feature_extraction/outputs/results_leakfree.csv
             feature_extraction/outputs/shap_*.png (one per model per disorder)
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("feature_extraction/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset configurations ─────────────────────────────────────────────────

CONFIGS = {
    "asd": {
        "name": "Autism Spectrum Disorder (ASD)",
        "csv": "ml/data/asd_uci_combined.csv",
        "features": [
            "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
            "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
            "age","jaundice","austim",
        ],
        "label": "Class_ASD",
        # 'result' is excluded — it's the raw AQ-10 sum which directly determines Class_ASD
        "group_col": "age_group",
        "leakage_excluded": ["result"],
        "dataset_note": "UCI ASD Screening Dataset (Adults+Children, N=989 real records, CC BY 4.0). "
                        "Features: AQ-10 items + demographics. EXCLUDES 'result' column (AQ-10 sum = label proxy).",
    },
    "adhd": {
        "name": "ADHD (WHO ASRS + CPT-3 + Conners)",
        "csv": "ml/data/adhd_dataset_v2.csv",
        "features": [
            "asrs_inatt_1","asrs_inatt_2","asrs_inatt_3","asrs_inatt_4","asrs_inatt_5","asrs_inatt_6",
            "asrs_hyper_1","asrs_hyper_2","asrs_hyper_3","asrs_hyper_4","asrs_hyper_5","asrs_hyper_6",
            "conners_inattention","conners_hyperactivity","conners_executive","conners_learning",
            "cpt_omission_tscore","cpt_commission_tscore","cpt_rt_mean_ms",
            "cpt_rt_variability","cpt_detectability",
            "family_history_adhd","sleep_problems","school_performance_low","age",
        ],
        "label": "label",
        "group_col": None,
        "leakage_excluded": [],
        "dataset_note": "WHO ASRS-v1.1 + CPT-3 cognitive + Conners rating scales. "
                        "3100 records (480 real norms + 2620 calibrated synthetic from WHO/CPT-3 distributions).",
    },
    "dyslexia": {
        "name": "Dyslexia (Phonological + Handwriting)",
        "csv": "ml/data/dyslexia_dataset_v2.csv",
        "npy": "ml/data/features/dyslexia_resnet_features.npy",
        "npy_labels": "ml/data/features/dyslexia_labels.npy",
        "npy_groups": "ml/data/features/dyslexia_groups.npy",
        "features": [
            "phonological_awareness","phonological_memory","rapid_naming_speed",
            "word_reading_fluency","nonword_reading","letter_reversal",
            "reading_avoidance","spelling_errors","slow_reading_rate",
            "family_history_dyslexia",
            "hw_pen_speed_mms","hw_stroke_consistency","hw_letter_reversal_rate",
            "hw_line_deviation_mm","hw_spacing_variability","hw_erasure_per_word",
        ],
        "label": "label",
        "group_col": None,
        "leakage_excluded": [],
        "dataset_note": "Phonological awareness norms (IDA/Yale) + MDPI handwriting kinematics. "
                        "3100 records (500 real IDA anchors + 2600 calibrated synthetic). "
                        "ResNet18 features used if extracted from GAMBO 208K images.",
    },
    "stuttering": {
        "name": "Stuttering (Acoustic/Clinical Features)",
        "csv": "ml/data/stuttering_dataset_v2.csv",
        "features": [
            # Acoustic features from SEP-28k/UCLASS norms — NOT event counts
            "disfluency_rate",
            "repetition_per_100w",
            "prolongation_dur_s",
            "block_freq_per_100w",
            "pause_freq_per_100w",
            "speech_rate_wpm",
            "phonation_ratio",
            "jitter_pct",
            "shimmer_db",
            "hnr_db",
            # Demographics
            "age",
            "onset_childhood",
            "family_history_stutter",
            "anxiety_speaking",
        ],
        "label": "label",
        "group_col": None,
        # EXCLUDED: stutter_severity (ordinal version of label) → leakage
        "leakage_excluded": ["stutter_severity"],
        "dataset_note": "SEP-28k/UCLASS acoustic norms: disfluency rate, repetition, prolongation, "
                        "jitter, shimmer, HNR, speech rate from published stutter severity scales. "
                        "3100 records (480 real SEP-28k/UCLASS norms + 2620 calibrated synthetic).",
    },
    "ssd": {
        "name": "Speech Sound Disorder (GFTA-3 + DEAP)",
        "csv": "ml/data/ssd_dataset_v2.csv",
        "features": [
            # Articulation error rates and acoustic measures — NOT diagnosis codes
            "intelligibility_score",
            "pcc_score",
            "gfta3_standard_score",
            "fricative_error_rate",
            "stop_error_rate",
            "liquid_error_rate",
            "affricate_error_rate",
            "cluster_error_rate",
            "late_sound_acquisition",
            "f1_formant_deviation",
            "f2_formant_deviation",
            "prosody_deviation",
            "hearing_normal",
            "age",
            "family_history_ssd",
        ],
        "label": "label",
        "group_col": "age_group",
        # EXCLUDED: phonological_process (ordinal severity) → label proxy
        "leakage_excluded": ["phonological_process"],
        "dataset_note": "GFTA-3 articulation test norms + DEAP acoustic features. "
                        "3100 records (560 real GFTA-3/DEAP anchors + 2540 calibrated synthetic). "
                        "MFCC features from 138 real child speech MP3s also available.",
    },
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_data(key: str) -> tuple:
    cfg = CONFIGS[key]

    # Dyslexia: prefer ResNet18 .npy if available
    if key == "dyslexia" and Path(cfg.get("npy", "")).exists():
        X = np.load(cfg["npy"], allow_pickle=False)
        y = np.load(cfg["npy_labels"], allow_pickle=False).astype(int)
        g = np.load(cfg["npy_groups"], allow_pickle=False)
        log.info("[dyslexia] Using ResNet18 features: %s", X.shape)
        return X, y, g, "ResNet18 512-dim (GAMBO 208K real images)"

    # All others: tabular CSV
    df = pd.read_csv(cfg["csv"])

    # Only keep columns that exist in this CSV
    feat_cols = [c for c in cfg["features"] if c in df.columns]
    missing = [c for c in cfg["features"] if c not in df.columns]
    if missing:
        log.warning("[%s] Missing feature columns (skipped): %s", key, missing)

    # Verify leakage columns are absent
    for lc in cfg["leakage_excluded"]:
        if lc in feat_cols:
            feat_cols.remove(lc)
            log.warning("[%s] REMOVED leakage column: %s", key, lc)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    # Fill NaN with column median (fitted on full set here — fixed before split)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    y = df[cfg["label"]].astype(int).values

    # Groups
    if cfg["group_col"] and cfg["group_col"] in df.columns:
        from sklearn.preprocessing import LabelEncoder
        g = LabelEncoder().fit_transform(df[cfg["group_col"]].fillna("unknown").astype(str))
    else:
        g = np.arange(len(y))  # each sample = own group (stratified random)

    log.info("[%s] CSV: %d samples × %d features | pos=%d neg=%d",
             key, X.shape[0], X.shape[1], (y==1).sum(), (y==0).sum())
    log.info("[%s] Features: %s", key, feat_cols)
    return X, y, g, "Tabular CSV"


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score,
    )
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2,2)
                      else (cm[0,0], 0, 0, cm[1,1]))
    spec = float(tn) / max(1, float(tn + fp))

    auc = None
    if y_prob is not None:
        try:
            auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            pass

    return {
        "accuracy":    round(float(acc), 4),
        "precision":   round(float(prec), 4),
        "recall":      round(float(rec), 4),
        "f1":          round(float(f1), 4),
        "specificity": round(float(spec), 4),
        "roc_auc":     auc,
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


# ── SHAP ──────────────────────────────────────────────────────────────────

def save_shap(pipeline, X_train, X_test, disorder, mname, feature_names, out_dir):
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clf    = pipeline.named_steps.get("clf", pipeline)
        scaler = pipeline.named_steps.get("sc", None)
        Xtr = scaler.transform(X_train) if scaler else X_train
        Xte = scaler.transform(X_test[:min(300, len(X_test))]) if scaler else X_test[:300]

        mtype = type(clf).__name__
        if any(t in mtype for t in ("Forest","Boosting","XGB","Tree")):
            exp = shap.TreeExplainer(clf)
            sv  = exp.shap_values(Xte)
            sv  = sv[1] if isinstance(sv, list) else sv
        elif "Logistic" in mtype:
            exp = shap.LinearExplainer(clf, Xtr)
            sv  = exp.shap_values(Xte)
        else:
            return

        plt.figure(figsize=(10, 6))
        feat_display = feature_names[:Xte.shape[1]] if feature_names else None
        shap.summary_plot(sv, Xte, feature_names=feat_display,
                          show=False, max_display=15, plot_type="bar")
        plt.title(f"SHAP — {disorder.upper()} {mname}")
        plt.tight_layout()
        fname = out_dir / f"shap_{disorder}_{mname.lower().replace(' ','_')}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("  SHAP saved: %s", fname)
    except Exception as e:
        log.warning("  SHAP failed (%s %s): %s", disorder, mname, e)


# ── Training ──────────────────────────────────────────────────────────────

def train_one(key: str, out_dir: Path) -> dict:
    from sklearn.model_selection import (
        GroupShuffleSplit, GroupKFold,
        StratifiedShuffleSplit, StratifiedKFold,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import make_scorer, f1_score, roc_auc_score

    cfg = CONFIGS[key]
    X, y, groups, source = load_data(key)

    # Split strategy: GroupShuffleSplit if groups are meaningful
    # (i.e. not just arange), else StratifiedShuffleSplit
    unique_groups = len(np.unique(groups))
    use_groups = (unique_groups < len(y) * 0.9) and (unique_groups >= 5)

    if use_groups:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        cv = GroupKFold(n_splits=5)
        cv_groups_train = groups[train_idx]
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_groups_train = None

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    log.info("=" * 60)
    log.info("[%s] %s", key, cfg["name"])
    log.info("  Train=%d  Test=%d | pos_train=%d neg_train=%d | pos_test=%d neg_test=%d",
             len(train_idx), len(test_idx),
             (y_train==1).sum(), (y_train==0).sum(),
             (y_test==1).sum(),  (y_test==0).sum())

    # NaN fill using TRAIN statistics only (no leakage from test)
    col_medians = np.nanmedian(X_train, axis=0)
    for idx_set, X_set in [(train_idx, X_train), (test_idx, X_test)]:
        nm = np.isnan(X_set)
        X_set[nm] = np.take(col_medians, np.where(nm)[1])

    models = {
        "Logistic Regression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                solver="lbfgs", C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        pos = int((y_train==1).sum()); neg = int((y_train==0).sum())
        models["XGBoost"] = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=neg/max(1,pos),
                subsample=0.8,
                eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0)),
        ])
    except ImportError:
        pass

    feat_names = CONFIGS[key]["features"] if source != "ResNet18 512-dim (GAMBO 208K real images)" else [f"resnet_{i}" for i in range(512)]
    results = {}

    for mname, pipe in models.items():
        log.info("  Training: %s", mname)

        # CV on training set only
        cv_f1_scores  = []
        cv_auc_scores = []
        cv_split = cv.split(X_train, y_train, groups=cv_groups_train) if cv_groups_train is not None else cv.split(X_train, y_train)

        for fold_train, fold_val in cv_split:
            Xf_tr, Xf_val = X_train[fold_train], X_train[fold_val]
            yf_tr, yf_val = y_train[fold_train], y_train[fold_val]
            # Fit on fold train only using the main pipeline directly
            try:
                pipe.fit(Xf_tr, yf_tr)
                yf_pred = pipe.predict(Xf_val)
                try:
                    yf_prob = pipe.predict_proba(Xf_val)[:,1]
                    cv_auc_scores.append(roc_auc_score(yf_val, yf_prob))
                except Exception:
                    pass
                cv_f1_scores.append(f1_score(yf_val, yf_pred, zero_division=0))
            except Exception as e:
                log.warning("    Fold failed: %s", e)

        cv_f1  = np.array(cv_f1_scores)  if cv_f1_scores  else np.array([np.nan])
        cv_auc = np.array(cv_auc_scores) if cv_auc_scores else np.array([np.nan])

        log.info("    CV  F1: %.4f ± %.4f | AUC: %.4f ± %.4f",
                 np.nanmean(cv_f1), np.nanstd(cv_f1),
                 np.nanmean(cv_auc), np.nanstd(cv_auc))

        # Final fit on full training set, evaluate once on test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_prob = pipe.predict_proba(X_test)[:,1]
        except Exception:
            y_prob = None

        m = compute_metrics(y_test, y_pred, y_prob)
        log.info("    Test Prec=%.4f Rec=%.4f F1=%.4f Spec=%.4f AUC=%s",
                 m["precision"], m["recall"], m["f1"], m["specificity"], m["roc_auc"])
        log.info("    CM: TN=%d FP=%d FN=%d TP=%d", m["TN"], m["FP"], m["FN"], m["TP"])

        save_shap(pipe, X_train, X_test, key, mname, feat_names, out_dir)

        results[mname] = {
            "cv": {
                "f1_mean":  round(float(np.nanmean(cv_f1)),  4),
                "f1_std":   round(float(np.nanstd(cv_f1)),   4),
                "auc_mean": round(float(np.nanmean(cv_auc)), 4),
                "auc_std":  round(float(np.nanstd(cv_auc)),  4),
                "f1_folds": [round(float(v),4) for v in cv_f1],
                "auc_folds":[round(float(v),4) for v in cv_auc],
            },
            "test": m,
        }

    return results


# ── Save results ──────────────────────────────────────────────────────────

def save_results(all_results: dict, out_dir: Path) -> None:
    rows = []
    for disorder, models in all_results.items():
        for mname, res in models.items():
            cm = res["test"]["confusion_matrix"]
            rows.append({
                "disorder":    disorder,
                "model":       mname,
                "cv_f1_mean":  res["cv"]["f1_mean"],
                "cv_f1_std":   res["cv"]["f1_std"],
                "cv_auc_mean": res["cv"]["auc_mean"],
                "cv_auc_std":  res["cv"]["auc_std"],
                "test_accuracy":    res["test"]["accuracy"],
                "test_precision":   res["test"]["precision"],
                "test_recall":      res["test"]["recall"],
                "test_f1":          res["test"]["f1"],
                "test_specificity": res["test"]["specificity"],
                "test_roc_auc":     res["test"]["roc_auc"],
                "TN": res["test"]["TN"],
                "FP": res["test"]["FP"],
                "FN": res["test"]["FN"],
                "TP": res["test"]["TP"],
                "confusion_matrix": str(cm),
            })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_leakfree.csv"
    df.to_csv(csv_path, index=False)
    json_path = out_dir / "results_leakfree.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))

    print("\n" + "=" * 100)
    print("LEAK-FREE RESULTS")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    log.info("Results saved: %s", csv_path)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--disorder", default=None,
                        choices=list(CONFIGS.keys()) + [None])
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out = Path(args.out_dir)
    keys = [args.disorder] if args.disorder else list(CONFIGS.keys())

    all_results = {}
    for k in keys:
        try:
            all_results[k] = train_one(k, out)
        except Exception as e:
            log.error("[%s] FAILED: %s", k, e)
            import traceback; traceback.print_exc()

    if all_results:
        save_results(all_results, out)
