"""
feature_extraction/train_with_features.py
==========================================
Updated training script that:
  - Loads pre-extracted .npy feature files
  - Uses GroupShuffleSplit (test_size=0.25) for train/test split
  - Uses GroupKFold(n_splits=5) for cross-validation on training data only
  - Trains LR + RF + GB (+ XGBoost if available)
  - Reports Accuracy, Precision, Recall, F1, Specificity, ROC-AUC,
    Confusion Matrix, CV mean ± std for every pathway
  - Saves results to outputs/results.csv
  - Saves SHAP plots for tree models to outputs/shap_<disorder>.png
  - Uses class_weight='balanced'; no SMOTE; scaler fitted on train only
    via sklearn Pipeline; no leakage from test set

Run ALL disorders:
  python -m feature_extraction.train_with_features

Run ONE disorder:
  python -m feature_extraction.train_with_features --disorder dyslexia

Feature file locations (default: ml/data/features/):
  dyslexia_resnet_features.npy / dyslexia_labels.npy / dyslexia_groups.npy
  stuttering_features.npy / stuttering_labels.npy / stuttering_groups.npy
  asd_features.npy / asd_labels.npy / asd_groups.npy
  ssd_features.npy / ssd_labels.npy / ssd_groups.npy
  (ADHD is loaded from ml/data/adhd_dataset_v2.csv — tabular, no change)
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_DIR = Path("ml/data/features")
OUT_DIR     = Path("feature_extraction/outputs")

DISORDERS = ["asd", "adhd", "dyslexia", "stuttering", "ssd"]

DISORDER_META = {
    "asd": {
        "name": "Autism Spectrum Disorder",
        "feature_file": "asd_features.npy",
        "label_file":   "asd_labels.npy",
        "group_file":   "asd_groups.npy",
        "tabular_csv":  None,
    },
    "adhd": {
        "name": "ADHD",
        "feature_file": None,   # tabular only
        "label_file":   None,
        "group_file":   None,
        "tabular_csv":  "ml/data/adhd_dataset_v2.csv",
        "tabular_label": "label",
        "tabular_features": [
            "asrs_inatt_1","asrs_inatt_2","asrs_inatt_3",
            "asrs_inatt_4","asrs_inatt_5","asrs_inatt_6",
            "asrs_hyper_1","asrs_hyper_2","asrs_hyper_3",
            "asrs_hyper_4","asrs_hyper_5","asrs_hyper_6",
            "conners_inattention","conners_hyperactivity",
            "conners_executive","conners_learning",
            "cpt_omission_tscore","cpt_commission_tscore",
            "cpt_rt_mean_ms","cpt_rt_variability","cpt_detectability",
            "family_history_adhd","sleep_problems","school_performance_low",
            "age",
        ],
    },
    "dyslexia": {
        "name": "Dyslexia",
        "feature_file": "dyslexia_resnet_features.npy",
        "label_file":   "dyslexia_labels.npy",
        "group_file":   "dyslexia_groups.npy",
        "tabular_csv":  None,
    },
    "stuttering": {
        "name": "Stuttering",
        "feature_file": "stuttering_features.npy",
        "label_file":   "stuttering_labels.npy",
        "group_file":   "stuttering_groups.npy",
        "tabular_csv":  None,
    },
    "ssd": {
        "name": "Speech Sound Disorder",
        "feature_file": "ssd_features.npy",
        "label_file":   "ssd_labels.npy",
        "group_file":   "ssd_groups.npy",
        "tabular_csv":  None,
    },
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_disorder_data(disorder: str, feature_dir: Path):
    meta = DISORDER_META[disorder]

    if meta["tabular_csv"]:
        # ADHD — load tabular CSV
        df = pd.read_csv(meta["tabular_csv"])
        feat_cols = [c for c in meta["tabular_features"] if c in df.columns]
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        y = df[meta["tabular_label"]].astype(int).values
        # Group = participant index (each row = unique individual → each is own group)
        g = np.arange(len(y))
        log.info("[%s] Tabular: %d rows × %d features", disorder, X.shape[0], X.shape[1])
        return X, y, g

    # Load from pre-extracted .npy files
    feat_path  = feature_dir / meta["feature_file"]
    label_path = feature_dir / meta["label_file"]
    group_path = feature_dir / meta["group_file"]

    for p in [feat_path, label_path, group_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Feature file not found: {p}\n"
                f"Run the feature extractor first:\n"
                f"  python -m feature_extraction.{disorder}_resnet  (dyslexia)\n"
                f"  python -m feature_extraction.{disorder}_wav2vec2  (stuttering/ssd)\n"
                f"  python -m feature_extraction.{disorder}_mediapipe  (asd)\n"
            )

    X = np.load(feat_path, allow_pickle=False)
    y = np.load(label_path, allow_pickle=False).astype(int)
    g = np.load(group_path, allow_pickle=False)

    log.info("[%s] Loaded: %d samples × %d features | pos=%d neg=%d",
             disorder, X.shape[0], X.shape[1], (y == 1).sum(), (y == 0).sum())
    return X, y, g


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # Specificity = TN / (TN + FP)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn) / max(1, float(tn + fp))
    else:
        spec = float("nan")

    auc = float("nan")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass

    return {
        "accuracy":    round(acc, 4),
        "precision":   round(prec, 4),
        "recall":      round(rec, 4),
        "f1":          round(f1, 4),
        "specificity": round(spec, 4),
        "roc_auc":     round(auc, 4) if not np.isnan(auc) else None,
        "confusion_matrix": cm.tolist(),
    }


# ── SHAP ──────────────────────────────────────────────────────────────────

def save_shap_plot(model, X_train: np.ndarray, X_test: np.ndarray,
                   disorder: str, model_name: str, out_dir: Path) -> None:
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use TreeExplainer for RF/GB/XGB, LinearExplainer for LR
        clf = model.named_steps.get("clf", model)
        scaler = model.named_steps.get("sc", None)

        X_tr = scaler.transform(X_train) if scaler else X_train
        X_te = scaler.transform(X_test[:min(200, len(X_test))]) if scaler else X_test[:200]

        model_type = type(clf).__name__
        if "Forest" in model_type or "Boosting" in model_type or "XGB" in model_type:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_te)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # class 1 for binary
        elif "Logistic" in model_type:
            explainer = shap.LinearExplainer(clf, X_tr)
            shap_vals = explainer.shap_values(X_te)
        else:
            return

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_te, show=False,
                          max_display=20, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {disorder.upper()} {model_name}")
        plt.tight_layout()
        fname = out_dir / f"shap_{disorder}_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("SHAP plot saved: %s", fname)
    except Exception as e:
        log.warning("SHAP plot failed for %s %s: %s", disorder, model_name, e)


# ── Training ──────────────────────────────────────────────────────────────

def train_disorder(disorder: str, feature_dir: Path, out_dir: Path) -> dict:
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import make_scorer, f1_score

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Training: %s", DISORDER_META[disorder]["name"])

    X, y, groups = load_disorder_data(disorder, feature_dir)

    # ── Group-based train/test split ──────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train         = groups[train_idx]

    log.info("Train=%d  Test=%d | Train pos=%d neg=%d | Test pos=%d neg=%d",
             len(train_idx), len(test_idx),
             (y_train == 1).sum(), (y_train == 0).sum(),
             (y_test  == 1).sum(), (y_test  == 0).sum())

    # ── Models ───────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced",
                random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5,
                random_state=42)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale = neg / max(1, pos)
        models["XGBoost"] = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=5,
                scale_pos_weight=scale,
                eval_metric="logloss",
                random_state=42, n_jobs=-1,
                verbosity=0)),
        ])
    except ImportError:
        pass

    # ── GroupKFold CV on training data only ───────────────────────────────
    gkf = GroupKFold(n_splits=5)
    scorer_f1  = make_scorer(f1_score, zero_division=0)
    scorer_auc = "roc_auc"

    results = {}

    for mname, pipeline in models.items():
        log.info("  Model: %s", mname)

        # CV on train set only (never touches test)
        try:
            cv_f1  = cross_val_score(pipeline, X_train, y_train,
                                     groups=g_train, cv=gkf,
                                     scoring=scorer_f1, n_jobs=-1)
            cv_auc = cross_val_score(pipeline, X_train, y_train,
                                     groups=g_train, cv=gkf,
                                     scoring=scorer_auc, n_jobs=-1)
        except Exception as e:
            log.warning("    CV failed: %s", e)
            cv_f1  = np.array([float("nan")])
            cv_auc = np.array([float("nan")])

        cv_summary = {
            "cv_f1_mean":  round(float(np.nanmean(cv_f1)), 4),
            "cv_f1_std":   round(float(np.nanstd(cv_f1)),  4),
            "cv_auc_mean": round(float(np.nanmean(cv_auc)), 4),
            "cv_auc_std":  round(float(np.nanstd(cv_auc)),  4),
            "cv_f1_folds": [round(float(v), 4) for v in cv_f1],
            "cv_auc_folds":[round(float(v), 4) for v in cv_auc],
        }
        log.info("    CV F1:  %.4f ± %.4f | CV AUC: %.4f ± %.4f",
                 cv_summary["cv_f1_mean"],  cv_summary["cv_f1_std"],
                 cv_summary["cv_auc_mean"], cv_summary["cv_auc_std"])

        # Train on full training set — test set touched once here
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

        test_metrics = compute_metrics(y_test, y_pred, y_prob)
        log.info("    Test — Prec:%.4f Rec:%.4f F1:%.4f Spec:%.4f AUC:%s",
                 test_metrics["precision"], test_metrics["recall"],
                 test_metrics["f1"],        test_metrics["specificity"],
                 test_metrics["roc_auc"])
        log.info("    CM: %s", test_metrics["confusion_matrix"])

        # SHAP
        save_shap_plot(pipeline, X_train, X_test, disorder, mname, out_dir)

        results[mname] = {"cv": cv_summary, "test": test_metrics}

    return results


# ── Save results ──────────────────────────────────────────────────────────

def save_results(all_results: dict, out_dir: Path) -> None:
    rows = []
    for disorder, models in all_results.items():
        for mname, res in models.items():
            row = {
                "disorder":    disorder,
                "model":       mname,
                "cv_f1_mean":  res["cv"]["cv_f1_mean"],
                "cv_f1_std":   res["cv"]["cv_f1_std"],
                "cv_auc_mean": res["cv"]["cv_auc_mean"],
                "cv_auc_std":  res["cv"]["cv_auc_std"],
                "test_acc":    res["test"]["accuracy"],
                "test_prec":   res["test"]["precision"],
                "test_recall": res["test"]["recall"],
                "test_f1":     res["test"]["f1"],
                "test_spec":   res["test"]["specificity"],
                "test_auc":    res["test"]["roc_auc"],
                "confusion_matrix": str(res["test"]["confusion_matrix"]),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Results saved to %s", csv_path)

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Full results (JSON) saved to %s", json_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train NeuroScan AI models on extracted features")
    parser.add_argument("--disorder",    default=None,
                        choices=DISORDERS + [None],
                        help="Train one disorder only; omit for all")
    parser.add_argument("--feature_dir", default=str(FEATURE_DIR))
    parser.add_argument("--out_dir",     default=str(OUT_DIR))
    args = parser.parse_args()

    feat_dir = Path(args.feature_dir)
    out_dir  = Path(args.out_dir)

    to_train = [args.disorder] if args.disorder else DISORDERS

    all_results = {}
    for d in to_train:
        try:
            all_results[d] = train_disorder(d, feat_dir, out_dir)
        except FileNotFoundError as e:
            log.error("Skipping %s — %s", d, e)
        except Exception as e:
            log.error("Error training %s: %s", d, e)
            import traceback
            traceback.print_exc()

    if all_results:
        save_results(all_results, out_dir)


if __name__ == "__main__":
    main()
