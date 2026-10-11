"""Test what the raw models actually output for typical positive profiles"""
import sys, os, json
import numpy as np
import joblib

ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'
ART  = ROOT + r'\ml\artifacts'
DATA = ROOT + r'\ml\data'

# Load a few actual positive records from the training data to verify
print("=== Testing models with ACTUAL training records ===")

for dis in ['asd', 'adhd', 'dyslexia', 'stuttering', 'ssd']:
    fname = f'{dis}_dataset_v2.json' if os.path.exists(os.path.join(DATA, f'{dis}_dataset_v2.json')) else f'{dis}_dataset.json'
    fpath = os.path.join(DATA, fname)
    if not os.path.exists(fpath):
        print(f"  {dis}: dataset not found")
        continue
    
    with open(fpath, encoding='utf-8') as fh:
        records = json.load(fh)
    
    # Get feature names from train_all_disorders.py definitions
    FEATURES = {
        'asd': ['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
                'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
                'age','gender_num','jaundice_num','austim_num',
                'arm_flapping','head_banging','spinning','rocking',
                'gaze_avoidance','echolalia','hypo_gaze','social_isolate',
                'gaze_avoidance_pct','joint_attention_score'],
        'adhd': [f'asrs_inatt_{i}' for i in range(1,7)] +
                [f'asrs_hyper_{i}' for i in range(1,7)] +
                ['conners_inattention','conners_hyperactivity','conners_executive',
                 'conners_learning','cpt_omission_tscore','cpt_commission_tscore',
                 'cpt_rt_mean_ms','cpt_rt_variability','cpt_detectability',
                 'family_history_adhd','sleep_problems','school_performance_low'],
        'dyslexia': ['phonological_awareness','phonological_memory','rapid_naming_speed',
                     'word_reading_fluency','nonword_reading','letter_reversal',
                     'reading_avoidance','spelling_errors','slow_reading_rate',
                     'family_history_dyslexia','hw_pen_speed_mms','hw_stroke_consistency',
                     'hw_letter_reversal_rate','hw_line_deviation_mm','hw_spacing_variability',
                     'hw_erasure_per_word'],
        'stuttering': ['disfluency_rate','repetition_per_100w','prolongation_dur_s',
                       'block_freq_per_100w','speech_rate_wpm','pause_freq_per_100w',
                       'phonation_ratio','jitter_pct','shimmer_db','hnr_db',
                       'stutter_severity','anxiety_speaking','family_history_stutter','onset_childhood'],
        'ssd': ['pcc_score','gfta3_standard_score','stop_error_rate',
                'fricative_error_rate','affricate_error_rate','liquid_error_rate',
                'cluster_error_rate','intelligibility_score','prosody_deviation',
                'f1_formant_deviation','f2_formant_deviation',
                'late_sound_acquisition','phonological_process',
                'family_history_ssd','hearing_normal'],
    }
    
    feats = FEATURES.get(dis, [])
    pos_recs = [r for r in records if r.get('label') == 1][:5]
    
    if not pos_recs:
        print(f"  {dis}: no positive records found")
        continue
    
    # Load RF model
    rf_path = os.path.join(ART, f'{dis}_rf.joblib')
    if not os.path.exists(rf_path):
        print(f"  {dis}: model not found")
        continue
    
    rf = joblib.load(rf_path)
    
    # Build feature vector for first positive record
    r = pos_recs[0]
    vec = []
    for f in feats:
        v = r.get(f, 0)
        try: vec.append(float(v) if v is not None else 0.0)
        except: vec.append(0.0)
    
    X = np.array(vec, dtype=np.float32).reshape(1, -1)
    try:
        prob = rf.predict_proba(X)[0, 1]
        print(f"  {dis:<12}: RF raw prob = {prob:.3f} ({int(prob*100)}%) for actual positive record")
    except Exception as e:
        print(f"  {dis:<12}: ERROR - {e}, expected {len(feats)} features, got {X.shape[1]}")
