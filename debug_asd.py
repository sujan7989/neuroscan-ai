import numpy as np, joblib, os, sys, json
ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'
ART  = ROOT + r'\ml\artifacts'
DATA = ROOT + r'\ml\data'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

rf = joblib.load(os.path.join(ART, 'asd_rf.joblib'))

# Test with training data feature vector: all A1-A10=1, age_norm=0.12, gender_num=1, jaundice_num=1, austim_num=1
# This is the maximum ASD symptom load
vec_max = np.array([1,1,1,1,1,1,1,1,1,1, 0.12, 1.0, 1.0, 1.0], dtype=np.float32).reshape(1,-1)
p = rf.predict_proba(vec_max)[0,1]
print(f"All-1 AQ + family/jaundice: RF={p:.3f} ({int(p*100)}%)")

# Test with just AQ scores = 1 (matching our mapping for q_ answers = 4)
vec_aq_only = np.array([1,1,1,1,1,1,1,1,1,1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1,-1)
p2 = rf.predict_proba(vec_aq_only)[0,1]
print(f"All-1 AQ, no demo:          RF={p2:.3f} ({int(p2*100)}%)")

# Test what happens when the model sees 0s for all features
vec_zero = np.zeros((1,14), dtype=np.float32)
p3 = rf.predict_proba(vec_zero)[0,1]
print(f"All zeros:                  RF={p3:.3f} ({int(p3*100)}%)")

# Check what aq() function produces for q_eye_contact=4
# Our mapping: aq('q_eye_contact') = 1.0 if 4 >= 2 = True -> 1.0 ✓
# But what if answers dict uses different keys?
# Test the predictor directly
from ml.inference.multi_predictor_v2 import predict_multi_disorder

answers_strong_asd = {
    'q_eye_contact': 4,         # avoids completely = max symptom
    'q_social_interaction': 4,  # completely withdrawn
    'q_repetitive_behavior': 4, # always
    'q_routine': 4,             # extreme distress
    'q_special_interests': 4,   # extreme
    'q_sensory_response': 3,    # over-sensitive
    'q_language_development': 3,# single words only
    'q_emotional_regulation': 3,# frequent meltdowns
    'q_attention_span': 1,      # low (co-occurring)
    'q_anxiety_social': 3,      # very anxious
}
demo = {'age': 7, 'gender': 'm', 'family_history': True, 'jaundice': True}

result = predict_multi_disorder(answers_strong_asd, demo)
print("\nPredictor results for strong ASD profile:")
for d, data in result['disorder_results'].items():
    print(f"  {d}: {data['probability_pct']}%  {data['riskLevel']}  (raw={data['probability_raw']:.3f})")

# Debug: show what feature vector is produced for ASD
import sys; sys.path.insert(0, ROOT)
from ml.inference.multi_predictor_v2 import build_feature_vector
feat = build_feature_vector(answers_strong_asd, demo, 'asd')
print(f"\nASD feature vector: {feat.flatten().tolist()}")
p_direct = rf.predict_proba(feat)[0,1]
print(f"Direct RF prediction: {p_direct:.3f} ({int(p_direct*100)}%)")
