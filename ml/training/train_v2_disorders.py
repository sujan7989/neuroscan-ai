"""NeuroScan AI v2 — Train all 5 disorder models"""
import os, json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA    = os.path.join(ROOT, 'ml', 'data')
ART     = os.path.join(ROOT, 'ml', 'artifacts')
os.makedirs(ART, exist_ok=True)

try:
    from xgboost import XGBClassifier; HAS_XGB = True
except: HAS_XGB = False

DISORDERS = {
    # ASD already trained — skip to save time
    # 'asd': { ... },
    'dyslexia': {
        'file': 'dyslexia_dataset_v2.json',
        'label': 'label',
        'name': 'Dyslexia (Reading + Handwriting)',
        'features': ['phonological_awareness','phonological_memory','rapid_naming_speed',
                     'word_reading_fluency','nonword_reading','letter_reversal',
                     'reading_avoidance','spelling_errors','slow_reading_rate',
                     'family_history_dyslexia',
                     'hw_pen_speed_mms','hw_stroke_consistency','hw_letter_reversal_rate',
                     'hw_line_deviation_mm','hw_spacing_variability','hw_erasure_per_word'],
    },
    'adhd': {
        'file': 'adhd_dataset_v2.json',
        'label': 'label',
        'name': 'ADHD (Questionnaire + Behavioural CPT)',
        'features': [f'asrs_inatt_{i}' for i in range(1,7)] +
                    [f'asrs_hyper_{i}' for i in range(1,7)] +
                    ['conners_inattention','conners_hyperactivity','conners_executive',
                     'conners_learning','cpt_omission_tscore','cpt_commission_tscore',
                     'cpt_rt_mean_ms','cpt_rt_variability','cpt_detectability',
                     'family_history_adhd','sleep_problems','school_performance_low'],
    },
    'stuttering': {
        'file': 'stuttering_dataset_v2.json',
        'label': 'label',
        'name': 'Stuttering (Audio / Acoustic)',
        'features': ['disfluency_rate','repetition_per_100w','prolongation_dur_s',
                     'block_freq_per_100w','speech_rate_wpm','pause_freq_per_100w',
                     'phonation_ratio','jitter_pct','shimmer_db','hnr_db',
                     'stutter_severity','anxiety_speaking',
                     'family_history_stutter','onset_childhood'],
    },
    'ssd': {
        'file': 'ssd_dataset_v2.json',
        'label': 'label',
        'name': 'Speech Sound Disorder (Audio / Articulation)',
        'features': ['pcc_score','gfta3_standard_score','stop_error_rate',
                     'fricative_error_rate','affricate_error_rate','liquid_error_rate',
                     'cluster_error_rate','intelligibility_score','prosody_deviation',
                     'f1_formant_deviation','f2_formant_deviation',
                     'late_sound_acquisition','phonological_process',
                     'family_history_ssd','hearing_normal'],
    },
}

def load(cfg):
    fp = os.path.join(DATA, cfg['file'])
    with open(fp, encoding='utf-8') as fh: recs = json.load(fh)
    feats = cfg['features']
    X, y = [], []
    for r in recs:
        row = []
        for f in feats:
            v = r.get(f, 0)
            try: row.append(float(v) if v is not None else 0.0)
            except: row.append(0.0)
        X.append(row)
        lbl = r.get(cfg['label'], r.get('label', 0))
        y.append(1 if str(lbl) in ('1','1.0','yes','ASD','True','true') else 0)
    return np.array(X, dtype=np.float32), np.array(y), feats

def train_one(dis_id, cfg):
    print(f"\n{'='*55}\n  {cfg['name']}\n{'='*55}", flush=True)
    X, y, feats = load(cfg)
    print(f"  Records={X.shape[0]}  Features={X.shape[1]}  Pos={y.sum()} ({y.mean()*100:.1f}%)", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'lr': Pipeline([('s', StandardScaler()),
                        ('c', LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced', random_state=42))]),
        'rf': RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
    }
    if False:  # XGB disabled — hangs on large datasets with cross-val
        pass
    results = {}
    for name, m in models.items():
        cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring='roc_auc')
        m.fit(Xtr, ytr)
        yp = m.predict(Xte); ypr = m.predict_proba(Xte)[:,1]
        tauc = roc_auc_score(yte, ypr)
        tacc = accuracy_score(yte, yp)
        tbal = balanced_accuracy_score(yte, yp)
        tf1  = f1_score(yte, yp, zero_division=0)
        print(f"  [{name}] CV-AUC={cv.mean():.3f}  Test-AUC={tauc:.3f}  Acc={tacc*100:.1f}%  Bal={tbal*100:.1f}%  F1={tf1:.3f}", flush=True)
        mp = os.path.join(ART, f'{dis_id}_{name}.joblib')
        joblib.dump(m, mp)
        results[name] = {'cv_auc': round(float(cv.mean()),4), 'test_auc': round(float(tauc),4),
                         'accuracy': round(float(tacc),4), 'balanced_acc': round(float(tbal),4),
                         'f1': round(float(tf1),4)}
    return results, feats, int(X.shape[0]), int(y.sum()), int((1-y).sum())

def main():
    print("="*55 + "\nNeuroScan AI v2 — 5-Disorder Training\n" + "="*55)
    all_res = {}
    for did, cfg in DISORDERS.items():
        res, feats, tot, pos, neg = train_one(did, cfg)
        all_res[did] = {'name': cfg['name'], 'total': tot, 'positive': pos,
                        'negative': neg, 'n_features': len(feats),
                        'features': feats, 'models': res}

    meta = {
        'version': 'v2',
        'disorders': list(DISORDERS.keys()),
        'total_all': sum(v['total'] for v in all_res.values()),
        'per_disorder': {k: {
            'name': v['name'], 'total': v['total'], 'n_features': v['n_features'],
            'best_auc': max(m['test_auc'] for m in v['models'].values()),
            'data_source': {
                'asd': 'UCI 996 real + MMASD/AV-ASD video features + calibrated synthetic',
                'dyslexia': 'IDA/Yale phonological + MDPI handwriting kinematics + synthetic',
                'adhd': 'WHO ASRS questionnaire + CPT-3 behavioural + synthetic',
                'stuttering': 'SEP-28k / UCLASS acoustic disfluency (audio-based) + synthetic',
                'ssd': 'GFTA-3 articulation N=2621 + DEAP acoustic features + synthetic',
            }.get(k, '')
        } for k, v in all_res.items()}
    }

    mp = os.path.join(ART, 'multi_disorder_metrics.json')
    with open(mp, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + "="*55 + "\nTRAINING COMPLETE\n" + "="*55)
    for did, v in all_res.items():
        best = max(m['test_auc'] for m in v['models'].values())
        print(f"  {did:<12}: {v['total']:>5} records  best-AUC={best:.3f}")
    return meta

if __name__ == '__main__':
    main()
