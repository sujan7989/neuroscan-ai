import json, os, joblib, numpy as np

ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'
DATA = ROOT + r'\ml\data'
ART  = ROOT + r'\ml\artifacts'

# Check each disorder's actual feature ranges in training data
for dis, feat_file, label_key, feats in [
    ('asd', 'asd_combined_3k.json', 'Class_ASD',
     ['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
      'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
      'age_norm','gender_num','jaundice_num','austim_num']),
    ('adhd', 'adhd_dataset_v2.json', 'label',
     ['asrs_inatt_1','asrs_inatt_2','conners_inattention','conners_hyperactivity']),
    ('dyslexia', 'dyslexia_dataset_v2.json', 'label',
     ['phonological_awareness','word_reading_fluency','hw_pen_speed_mms']),
    ('stuttering', 'stuttering_dataset_v2.json', 'label',
     ['disfluency_rate','speech_rate_wpm','stutter_severity']),
    ('ssd', 'ssd_dataset_v2.json', 'label',
     ['pcc_score','gfta3_standard_score','intelligibility_score']),
]:
    fp = os.path.join(DATA, feat_file)
    if not os.path.exists(fp):
        print(f"{dis}: dataset not found")
        continue
    with open(fp, encoding='utf-8') as f:
        records = json.load(f)
    pos = [r for r in records if str(r.get(label_key,0)) in ('1','1.0','yes','ASD')][:20]
    print(f"\n{dis.upper()} positive record ranges (n={len(pos)}):")
    for f in feats:
        vals = [float(r.get(f,0) or 0) for r in pos]
        if vals:
            print(f"  {f}: min={min(vals):.2f} mean={sum(vals)/len(vals):.2f} max={max(vals):.2f}")

    # Test model with actual positive record
    rf_path = os.path.join(ART, f'{dis}_rf.joblib')
    if not os.path.exists(rf_path) or not pos:
        continue
    
    # Get full feature list
    feat_meta_path = os.path.join(ART, f'{dis}_features.json')
    if os.path.exists(feat_meta_path):
        with open(feat_meta_path) as f:
            meta = json.load(f)
        all_feats = meta['features']
    else:
        continue
    
    rf = joblib.load(rf_path)
    r = pos[0]
    # Compute age_norm for ASD
    if dis == 'asd':
        age = float(r.get('age', 25) or 25)
        r['age_norm'] = min(1.0, age / 60.0)
        g = str(r.get('gender', 'm')).lower()
        r['gender_num'] = 1.0 if g in ('m','male','1') else 0.0
        j = r.get('jaundice', 0)
        r['jaundice_num'] = 1.0 if str(j).lower() in ('yes','1','true') else 0.0
        a = r.get('austim', 0)
        r['austim_num'] = 1.0 if str(a).lower() in ('yes','1','true') else 0.0
    
    vec = [float(r.get(f, 0) or 0) for f in all_feats]
    X = np.array(vec, dtype=np.float32).reshape(1, -1)
    p = rf.predict_proba(X)[0, 1]
    print(f"  MODEL TEST: RF prob for actual positive record = {p:.3f} ({int(p*100)}%)")
