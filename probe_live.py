import urllib.request, json

BASE = 'https://neuroscan-ai-n8vw.onrender.com'

def post(path, data):
    req = urllib.request.Request(BASE+path, data=json.dumps(data).encode(),
          headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())

# 1. Test assess with strong ASD symptoms
print("=== ASSESS: strong ASD profile ===")
d = post('/api/ml/assess', {
    'age': 7, 'gender': 'm', 'family_history': True, 'jaundice': True,
    'answers': {
        'q_eye_contact': 4, 'q_social_interaction': 4,
        'q_repetitive_behavior': 4, 'q_routine': 4,
        'q_special_interests': 4, 'q_sensory_response': 3,
        'q_language_development': 2, 'q_hyperactivity': 1,
        'q_emotional_regulation': 3, 'q_reading': 0
    }
})
asd = d.get('disorder_results',{}).get('asd',{})
print(f"  ASD prob_pct={asd.get('probability_pct')}  risk={asd.get('riskLevel')}")
print(f"  aq10_sum={d.get('aq10_sum')}  confidence={asd.get('confidence')}")

# 2. Test same input twice - should give SAME result
print("\n=== SAME INPUT TWICE (determinism check) ===")
same_body = {
    'age': 8, 'gender': 'f', 'family_history': False, 'jaundice': False,
    'answers': {
        'q_eye_contact': 2, 'q_social_interaction': 2,
        'q_repetitive_behavior': 3, 'q_routine': 2
    }
}
r1 = post('/api/ml/assess', same_body)
r2 = post('/api/ml/assess', same_body)
p1 = r1.get('disorder_results',{}).get('asd',{}).get('probability_pct',0)
p2 = r2.get('disorder_results',{}).get('asd',{}).get('probability_pct',0)
print(f"  Run 1: {p1}%  Run 2: {p2}%  Same={p1==p2}")

# 3. Doctor chat - stuttering question
print("\n=== DOCTOR CHAT: stuttering ===")
d2 = post('/api/doctor-chat', {
    'messages': [{'role':'user','content':'My son stutters when speaking. What therapy helps?'}],
    'language': 'en'
})
reply = d2.get('reply','')
print(f"  Mentions stutter/fluency: {'stutter' in reply.lower() or 'fluency' in reply.lower()}")
print(f"  First 200: {reply[:200]}")

# 4. Doctor chat - SSD question
print("\n=== DOCTOR CHAT: SSD ===")
d3 = post('/api/doctor-chat', {
    'messages': [{'role':'user','content':'My child cannot pronounce certain sounds like r and s correctly. What is this?'}],
    'language': 'en'
})
reply3 = d3.get('reply','')
print(f"  Mentions SSD/articulation: {'ssd' in reply3.lower() or 'articulation' in reply3.lower() or 'speech sound' in reply3.lower()}")
print(f"  First 200: {reply3[:200]}")
