"""
NeuroScan AI — Train all 5 disorder models with 3000+ datasets each
"""
import os, json, sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(ROOT, 'ml', 'data')
ART_DIR  = os.path.join(ROOT, 'ml', 'artifacts')
os.makedirs(ART_DIR, exist_ok=True)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DISORDER_CONFIGS = {
    'asd': {
        'file': 'asd_combined_3k.json',
        'label_field': 'Class_ASD',
        'feature_prefix': ['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
                           'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score'],
        'extra_features': ['arm_flapping','head_banging','spinning','rocking',
                           'gaze_avoidance','echolalia','hypo_gaze','social_isolate'],
        'name': 'ASD',
    },
    'adhd': {
        'file': 'adhd_dataset.json',
        'label_field': 'label',
        'feature_prefix': ['asrs_inattention_1','asrs_inattention_2','asrs_inattention_3',
                           'asrs_inattention_4','asrs_inattention_5','asrs_inattention_6',
                           'asrs_hyperactive_1','asrs_hyperactive_2','asrs_hyperactive_3',
                           'asrs_hyperactive_4','asrs_hyperactive_5','asrs_hyperactive_6',
                           'conners_inattention','conners_hyperactivity','conners_executive',
                           'conners_learning','family_history_adhd','sleep_problems',
                           'school_performance_low','medication_history'],
        'extra_features': [],
        'name': 'ADHD',
    },
    'dyslexia': {
        'file': 'dyslexia_dataset.json',
        'label_field': 'label',
        'feature_prefix': ['phonological_awareness','phonological_memory','rapid_naming_speed',
                           'word_reading_fluency','nonword_reading','letter_reversal',
                           'reading_avoidance','spelling_errors','slow_reading_rate',
                           'reading_comprehension_low','rhyming_difficulty','sequencing_difficulty',
                           'family_history_dyslexia','iep_reading'],
        'extra_features': [],
        'name': 'Dyslexia',
    },
    'dld': {
        'file': 'dld_dataset.json',
        'label_field': 'label',
        'feature_prefix': ['sentence_comprehension','word_structure','expressive_vocabulary',
                           'following_directions','formulated_sentences','recalling_sentences',
                           'vocabulary_delay','grammar_errors','narrative_coherence_low',
                           'instruction_following_low','word_finding_difficulty',
                           'late_talker_history','family_history_language','hearing_screen_pass'],
        'extra_features': [],
        'name': 'DLD',
    },
    'tourette': {
        'file': 'tourette_dataset.json',
        'label_field': 'label',
        'feature_prefix': ['motor_tic_number','motor_tic_frequency','motor_tic_intensity',
                           'vocal_tic_number','vocal_tic_frequency','vocal_tic_intensity',
                           'ygtss_impairment','puts_urge_score','adhd_comorbid','ocd_comorbid',
                           'stress_worsens_tics','onset_before_18','family_history_tics',
                           'tic_suppression_ability'],
        'extra_features': [],
        'name': 'Tourette Syndrome',
    },
}

def load_dataset(cfg):
    fpath = os.path.join(DATA_DIR, cfg['file'])
    with open(fpath, encoding='utf-8') as fh:
        records = json.load(fh)

    all_features = cfg['feature_prefix'] + cfg['extra_features']
    X_list, y_list = [], []
    for rec in records:
        vec = []
        for feat in all_features:
            val = rec.get(feat, 0)
            try:
                vec.append(float(val) if val is not None else 0.0)
            except (ValueError, TypeError):
                vec.append(0.0)
        X_list.append(vec)
        label_raw = rec.get(cfg['label_field'], rec.get('label', 0))
        y_list.append(1 if str(label_raw) in ('1','1.0','yes','YES','ASD','True','true') else 0)

    return np.array(X_list, dtype=np.float32), np.array(y_list), all_features


def train_disorder(disorder_id, cfg):
    print(f"\n{'='*55}")
    print(f"  Training: {cfg['name']} ({disorder_id})")
    print(f"{'='*55}", flush=True)

    X, y, feature_names = load_dataset(cfg)
    print(f"  Dataset: {X.shape[0]} records, {X.shape[1]} features", flush=True)
    print(f"  Positive: {y.sum()} ({y.mean()*100:.1f}%)  Negative: {(1-y).sum()}", flush=True)

    # Train/test split 75/25 stratified
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000,
                                       class_weight='balanced', random_state=42))
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=4,
            class_weight='balanced', random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }
    if HAS_XGB:
        scale_pw = float((y_tr==0).sum()) / max(1, (y_tr==1).sum())
        models['xgboost'] = __import__('xgboost').XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, scale_pos_weight=scale_pw,
            eval_metric='logloss', random_state=42
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        test_auc = roc_auc_score(y_te, y_prob)
        test_acc = accuracy_score(y_te, y_pred)
        test_f1  = f1_score(y_te, y_pred, zero_division=0)
        test_bal = balanced_accuracy_score(y_te, y_pred)

        print(f"  [{name}]  CV-AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}  "
              f"Test-AUC={test_auc:.3f}  Acc={test_acc*100:.1f}%  "
              f"BalAcc={test_bal*100:.1f}%  F1={test_f1:.3f}", flush=True)

        # Save model
        model_path = os.path.join(ART_DIR, f'{disorder_id}_{name}.joblib')
        joblib.dump(model, model_path)

        results[name] = {
            'cv_auc_mean': round(float(cv_auc.mean()), 4),
            'cv_auc_std':  round(float(cv_auc.std()), 4),
            'test_auc':    round(float(test_auc), 4),
            'test_accuracy': round(float(test_acc), 4),
            'test_balanced_accuracy': round(float(test_bal), 4),
            'test_f1': round(float(test_f1), 4),
            'model_path': model_path,
        }

    return results, feature_names, int(X.shape[0]), int(y.sum()), int((1-y).sum())


def main():
    print("=" * 55)
    print("NeuroScan AI — 5-Disorder Model Training")
    print("=" * 55)

    all_results = {}
    for dis_id, cfg in DISORDER_CONFIGS.items():
        results, features, total, pos, neg = train_disorder(dis_id, cfg)
        all_results[dis_id] = {
            'disorder': cfg['name'],
            'total_records': total,
            'positive': pos,
            'negative': neg,
            'features': features,
            'n_features': len(features),
            'models': results,
        }

    # Save multi-disorder metrics JSON
    meta = {
        'suite': 'NeuroScan AI Multi-Disorder Suite v4.0',
        'disorders': all_results,
        'total_records_all_disorders': sum(v['total_records'] for v in all_results.values()),
        'datasets': {
            'asd':      '3107 records (936 real UCI + 2171 calibrated synthetic)',
            'adhd':     '3100 records (400 anchored WHO-ASRS + 2700 calibrated synthetic)',
            'dyslexia': '3100 records (500 anchored IDA/Yale + 2600 calibrated synthetic)',
            'dld':      '3100 records (500 anchored Tomblin/CELF-5 + 2600 calibrated synthetic)',
            'tourette': '3100 records (450 anchored YGTSS/PUTS + 2650 calibrated synthetic)',
        }
    }
    meta_path = os.path.join(ART_DIR, 'multi_disorder_metrics.json')
    with open(meta_path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nSaved: {meta_path}")
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE — ALL 5 DISORDERS")
    print("=" * 55)
    for dis_id, res in all_results.items():
        best_auc = max(m['test_auc'] for m in res['models'].values())
        print(f"  {dis_id:<10}: {res['total_records']:>5} records  best-AUC={best_auc:.3f}")

    return meta


if __name__ == '__main__':
    main()
