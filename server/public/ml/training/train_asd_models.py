"""
NeuroScan AI — ASD Model Training Pipeline (Real UCI Dataset)
=============================================================
Dataset: UCI ASD Screening (Adults + Children combined, CC BY 4.0)
         996 real clinical records (704 adults + 292 children)

Evaluation:
  1. Strict 25% Stratified Holdout Test Split
  2. 5-Fold Stratified Cross-Validation (training split only)
  3. Age-Group Stratified Evaluation (child / adult)
  4. Sex-Stratified Evaluation (male / female)
  5. Bootstrap 95% Confidence Intervals
  6. Discrimination vs Calibration separation
"""

import os
import sys
import json
import hashlib
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix, balanced_accuracy_score
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.preprocessing.pipeline import batch_transform, MODEL_FEATURE_NAMES
from ml.data.extract_dataset import export_dataset, get_dataset_stats


def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(y_true[in_bin]) - np.mean(y_prob[in_bin])) * prop
    return float(ece)


def bootstrap_ci(y_true, y_prob, y_pred, n_bootstraps=500, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, aucs, praucs, briers = [], [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        praucs.append(average_precision_score(y_true[idx], y_prob[idx]))
        briers.append(brier_score_loss(y_true[idx], y_prob[idx]))

    def ci(arr):
        if not arr:
            return {'ci_lower_95': 0.0, 'ci_upper_95': 0.0}
        return {
            'ci_lower_95': round(float(np.percentile(arr, 2.5)), 4),
            'ci_upper_95': round(float(np.percentile(arr, 97.5)), 4)
        }
    return {
        'accuracy_95ci': ci(accs),
        'roc_auc_95ci': ci(aucs),
        'pr_auc_95ci': ci(praucs),
        'brier_score_95ci': ci(briers)
    }


def subgroup_metrics(model, X, y, records, feature_names, label):
    """Compute metrics on a subgroup of the test set."""
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        'subgroup': label,
        'n': int(len(y)),
        'positive_rate': round(float(np.mean(y)), 4),
        'accuracy': round(float(accuracy_score(y, y_pred)), 4),
        'sensitivity': round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        'specificity': round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        'roc_auc': round(float(roc_auc_score(y, y_prob)), 4),
        'f1_score': round(float(f1_score(y, y_pred, zero_division=0)), 4),
    }


def train_and_evaluate():
    print("=" * 70)
    print("NeuroScan AI — ASD Model Training (Real UCI Dataset)")
    print("=" * 70)

    # ── 1. Load real UCI combined dataset ────────────────────────────────
    records, csv_path, json_path = export_dataset()
    dataset_stats = get_dataset_stats(records)
    dataset_hash = calculate_sha256(json_path)

    print(f"\nDataset: {len(records)} real clinical records")
    print(f"  Positive ASD=1: {dataset_stats['positive']} ({dataset_stats['positive_rate']*100:.1f}%)")
    print(f"  Negative ASD=0: {dataset_stats['negative']}")
    print(f"  Age groups: {dataset_stats['age_groups']}")
    print(f"  Sex (M/F): {dataset_stats['sex_asd_positive']['m']}/{dataset_stats['sex_asd_positive']['f']} ASD+ | {dataset_stats['sex_asd_negative']['m']}/{dataset_stats['sex_asd_negative']['f']} ASD-")
    print(f"  SHA-256: {dataset_hash}")

    # ── 2. Feature extraction ──────────────────────────────────────────────
    X, y, feature_names = batch_transform(records)
    print(f"\nFeature matrix: {X.shape}, Target: {y.shape}")
    print(f"Features: {feature_names}")

    # Keep age_group per record for subgroup analysis
    age_groups_all = np.array([r.get('age_group', 'adult') for r in records])
    genders_all    = np.array([r.get('gender', 'm') for r in records])

    artifacts_dir = os.path.join(os.path.dirname(__file__), '../artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    # ── 3. Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test, ag_train, ag_test, gen_train, gen_test = train_test_split(
        X, y, age_groups_all, genders_all,
        test_size=0.25, stratify=y, random_state=42
    )
    print(f"\nTrain: N={len(y_train)} (Pos={int(sum(y_train==1))}, Neg={int(sum(y_train==0))})")
    print(f"Test:  N={len(y_test)}  (Pos={int(sum(y_test==1))}, Neg={int(sum(y_test==0))})")

    # ── 4. Model definitions ──────────────────────────────────────────────
    model_factories = {
        'logistic_regression': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'random_forest': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=4, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced', random_state=42
        ),
        'gradient_boosting': lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=42
        )
    }

    try:
        from xgboost import XGBClassifier
        model_factories['xgboost'] = lambda: XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8,
            scale_pos_weight=float(sum(y_train==0)) / max(1, sum(y_train==1)),
            eval_metric='logloss', random_state=42
        )
    except Exception as e:
        print(f"XGBoost not available: {e}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_results = {}

    print("\n" + "-" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("-" * 70)

    for name, factory in model_factories.items():
        cv_acc, cv_sens, cv_spec, cv_f1, cv_bal, cv_auc, cv_prauc, cv_brier, cv_ece = [], [], [], [], [], [], [], [], []
        fold_records = {}

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]

            m = factory()
            m.fit(Xtr, ytr)
            yp = m.predict(Xval)
            ypr = m.predict_proba(Xval)[:, 1]

            tn, fp, fn, tp = confusion_matrix(yval, yp, labels=[0, 1]).ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            acc  = float(accuracy_score(yval, yp))
            f1   = float(f1_score(yval, yp, zero_division=0))
            bal  = float(balanced_accuracy_score(yval, yp))
            auc  = float(roc_auc_score(yval, ypr))
            prauc= float(average_precision_score(yval, ypr))
            brier= float(brier_score_loss(yval, ypr))
            ece  = float(compute_ece(yval, ypr))

            cv_acc.append(acc); cv_sens.append(sens); cv_spec.append(spec)
            cv_f1.append(f1);   cv_bal.append(bal);   cv_auc.append(auc)
            cv_prauc.append(prauc); cv_brier.append(brier); cv_ece.append(ece)

            fold_records[f"fold_{fold}"] = {
                'n': len(yval),
                'accuracy': round(acc, 4), 'sensitivity': round(sens, 4),
                'specificity': round(spec, 4), 'f1_score': round(f1, 4),
                'balanced_accuracy': round(bal, 4), 'roc_auc': round(auc, 4),
                'pr_auc': round(prauc, 4), 'brier_score': round(brier, 4),
                'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
            }

        # Train final model on full training set
        final_model = factory()
        final_model.fit(X_train, y_train)

        # ── Holdout test set evaluation ────────────────────────────────
        ytp = final_model.predict(X_test)
        ytpr = final_model.predict_proba(X_test)[:, 1]
        ttn, tfp, tfn, ttp = confusion_matrix(y_test, ytp, labels=[0, 1]).ravel()

        test_metrics = {
            'n': int(len(y_test)),
            'accuracy':          round(float(accuracy_score(y_test, ytp)), 4),
            'sensitivity':       round(float(ttp / (ttp + tfn)) if (ttp + tfn) > 0 else 0.0, 4),
            'specificity':       round(float(ttn / (ttn + tfp)) if (ttn + tfp) > 0 else 0.0, 4),
            'f1_score':          round(float(f1_score(y_test, ytp, zero_division=0)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y_test, ytp)), 4),
            'roc_auc':           round(float(roc_auc_score(y_test, ytpr)), 4),
            'pr_auc':            round(float(average_precision_score(y_test, ytpr)), 4),
            'brier_score':       round(float(brier_score_loss(y_test, ytpr)), 4),
            'ece':               round(float(compute_ece(y_test, ytpr)), 4),
            'confusion_matrix': {'TN': int(ttn), 'FP': int(tfp), 'FN': int(tfn), 'TP': int(ttp)},
            'bootstrap_95ci':    bootstrap_ci(y_test, ytpr, ytp),
        }

        # ── Age-group stratified evaluation ───────────────────────────
        age_stratified = {}
        for grp in ['child', 'adult']:
            mask = (ag_test == grp)
            if mask.sum() >= 5:
                sm = subgroup_metrics(final_model, X_test[mask], y_test[mask], None, feature_names, grp)
                if sm:
                    age_stratified[grp] = sm

        # ── Sex-stratified evaluation ──────────────────────────────────
        sex_stratified = {}
        for sex in ['m', 'f']:
            mask = (gen_test == sex)
            if mask.sum() >= 5:
                sm = subgroup_metrics(final_model, X_test[mask], y_test[mask], None, feature_names, sex)
                if sm:
                    sex_stratified[sex] = sm

        # ── Feature importances ────────────────────────────────────────
        feat_imp = {}
        clf = final_model.named_steps['clf'] if hasattr(final_model, 'named_steps') else final_model
        if hasattr(clf, 'feature_importances_'):
            for fn_name, imp in zip(feature_names, clf.feature_importances_):
                feat_imp[fn_name] = round(float(imp), 4)
        elif hasattr(clf, 'coef_'):
            for fn_name, coef in zip(feature_names, clf.coef_[0]):
                feat_imp[fn_name] = round(float(coef), 4)

        # Save
        artifact_path = os.path.join(artifacts_dir, f"{name}.joblib")
        joblib.dump(final_model, artifact_path)

        evaluation_results[name] = {
            'model_name': name,
            'cross_validation_folds': fold_records,
            'cv_metrics_summary': {
                'mean_accuracy':          round(float(np.mean(cv_acc)), 4),
                'std_accuracy':           round(float(np.std(cv_acc)), 4),
                'mean_sensitivity':       round(float(np.mean(cv_sens)), 4),
                'mean_specificity':       round(float(np.mean(cv_spec)), 4),
                'mean_f1_score':          round(float(np.mean(cv_f1)), 4),
                'mean_balanced_accuracy': round(float(np.mean(cv_bal)), 4),
                'mean_roc_auc':           round(float(np.mean(cv_auc)), 4),
                'mean_pr_auc':            round(float(np.mean(cv_prauc)), 4),
                'mean_brier_score':       round(float(np.mean(cv_brier)), 4),
                'mean_ece':               round(float(np.mean(cv_ece)), 4),
            },
            'test_metrics': test_metrics,
            'age_stratified_metrics': age_stratified,
            'sex_stratified_metrics': sex_stratified,
            'feature_importance': feat_imp,
            'artifact_path': artifact_path,
        }

        print(f"\n[{name.upper()}]")
        print(f"  CV  Acc={np.mean(cv_acc)*100:.1f}% Sens={np.mean(cv_sens)*100:.1f}% Spec={np.mean(cv_spec)*100:.1f}% AUC={np.mean(cv_auc):.3f}")
        print(f"  TEST Acc={test_metrics['accuracy']*100:.1f}% Sens={test_metrics['sensitivity']*100:.1f}% Spec={test_metrics['specificity']*100:.1f}% AUC={test_metrics['roc_auc']:.3f} Brier={test_metrics['brier_score']:.4f}")
        if age_stratified:
            for g, sm in age_stratified.items():
                print(f"  AGE-{g.upper():<12} n={sm['n']:3d} AUC={sm['roc_auc']:.3f} Sens={sm['sensitivity']*100:.1f}% Spec={sm['specificity']*100:.1f}%")
        if sex_stratified:
            for s, sm in sex_stratified.items():
                label = 'MALE' if s == 'm' else 'FEMALE'
                print(f"  SEX-{label:<12} n={sm['n']:3d} AUC={sm['roc_auc']:.3f} Sens={sm['sensitivity']*100:.1f}% Spec={sm['specificity']*100:.1f}%")

    # ── 5. Scaler + SHAP ──────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))

    try:
        import shap
        rf_final = joblib.load(os.path.join(artifacts_dir, "random_forest.joblib"))
        explainer = shap.TreeExplainer(rf_final)
        joblib.dump(explainer, os.path.join(artifacts_dir, "rf_shap_explainer.joblib"))
        print("\nSHAP TreeExplainer serialized.")
    except Exception as e:
        print(f"SHAP note: {e}")

    # ── 6. Save metadata ──────────────────────────────────────────────────
    metadata = {
        'model_suite': 'NeuroScan AI ASD Screening ML Suite',
        'status': 'RESEARCH_PROTOTYPE',
        'is_experimental': True,
        'is_validated_clinical_model': False,
        'dataset_info': {
            'dataset_name': 'UCI ASD Screening Dataset (Adults + Children combined)',
            'sources': [
                {
                    'name': 'Autism Screening Adult',
                    'uci_id': 426,
                    'records': 704,
                    'doi': 'https://doi.org/10.24432/C5F019',
                    'license': 'CC BY 4.0'
                },
                {
                    'name': 'Autistic Spectrum Disorder Screening Data for Children',
                    'uci_id': 419,
                    'records': 292,
                    'doi': 'https://doi.org/10.24432/C5659W',
                    'license': 'CC BY 4.0'
                }
            ],
            'citation': 'Thabtah, F. (2017). UCI ML Repository. CC BY 4.0.',
            'dataset_sha256': dataset_hash,
            'total_real_records': dataset_stats['total'],
            'total_records': dataset_stats['total'],
            'positive_asd': dataset_stats['positive'],
            'negative_asd': dataset_stats['negative'],
            'positive_rate': dataset_stats['positive_rate'],
            'age_groups': dataset_stats['age_groups'],
            'sex_distribution': {
                'asd_positive': dataset_stats['sex_asd_positive'],
                'asd_negative': dataset_stats['sex_asd_negative'],
            },
            'age_range': dataset_stats['age_range'],
            'training_records': int(len(y_train)),
            'test_records': int(len(y_test)),
            'train_test_split': '75% train / 25% test (stratified)',
            'features': list(feature_names),
            'target_variable': 'Class_ASD',
            'augmentation_note': 'NO synthetic augmentation — model trained on real UCI data only',
        },
        'leakage_prevention': [
            'result column (AQ-10 raw sum) excluded from feature matrix',
            'aq10_sum excluded from feature matrix',
            'Class_ASD label excluded from feature matrix',
            'StandardScaler fit strictly within CV training folds',
            'Holdout test set isolated before any model fitting',
        ],
        'ensemble_weights': {
            'random_forest': 0.40,
            'gradient_boosting': 0.40,
            'logistic_regression': 0.20
        },
        'metrics': evaluation_results,
        'disclaimer': (
            'RESEARCH PROTOTYPE: NeuroScan AI screening indicators are for educational '
            'demonstration only. They do NOT constitute clinical diagnoses. '
            'Professional evaluation requires ADOS-2, ADI-R, or CARS administered by specialists.'
        )
    }

    metrics_path = os.path.join(artifacts_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved all artifacts to {artifacts_dir}")
    print(f"Dataset: {dataset_stats['total']} real records | Train={len(y_train)} Test={len(y_test)}")
    return metadata


if __name__ == '__main__':
    train_and_evaluate()
