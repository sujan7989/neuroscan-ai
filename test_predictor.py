import sys, os
ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ml.inference.multi_predictor_v2 import predict_multi_disorder, calibrate_prob as calibrate

print("=== Calibration function test ===")
for raw in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"  raw={raw:.1f} -> calibrated={calibrate(raw):.3f} ({int(calibrate(raw)*100)}%)")

print("\n=== Strong ASD profile ===")
ans = {'q_eye_contact':4,'q_social_interaction':4,'q_repetitive_behavior':4,
       'q_routine':4,'q_special_interests':4,'q_sensory_response':3}
demo = {'age':7,'gender':'m','family_history':True,'jaundice':True}
r = predict_multi_disorder(ans, demo)
for dis, data in r['disorder_results'].items():
    print(f"  {dis:<12}: {data['probability_pct']}%  {data['riskLevel']}")

print("\n=== Determinism (same input x2) ===")
r2 = predict_multi_disorder(ans, demo)
same = all(r['disorder_results'][d]['probability_pct'] == r2['disorder_results'][d]['probability_pct']
           for d in r['disorder_results'])
print(f"  Same result: {same}")

print("\n=== ADHD profile ===")
ans_a = {'q_attention_span':4,'q_hyperactivity':4,'q_emotional_regulation':3}
r3 = predict_multi_disorder(ans_a, {'age':9,'gender':'m'})
for dis, data in r3['disorder_results'].items():
    print(f"  {dis:<12}: {data['probability_pct']}%  {data['riskLevel']}")

print("\n=== Low-symptom (should be LOW probability) ===")
ans_low = {'q_eye_contact':0,'q_social_interaction':0,'q_repetitive_behavior':0}
r4 = predict_multi_disorder(ans_low, {'age':10,'gender':'f'})
for dis, data in r4['disorder_results'].items():
    print(f"  {dis:<12}: {data['probability_pct']}%  {data['riskLevel']}")
