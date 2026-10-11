import urllib.request, json, sys

BASE = "https://neuroscan-ai-n8vw.onrender.com"

def post(path, data):
    req = urllib.request.Request(BASE+path, data=json.dumps(data).encode(),
          headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}

def get(path):
    try:
        with urllib.request.urlopen(BASE+path, timeout=15) as r:
            return r.status, json.loads(r.read())
    except: return 0, {}

results = []
def chk(label, val):
    results.append(("OK" if val else "FAIL", label))
    print(f"  [{'OK' if val else 'FAIL'}] {label}")

print("="*55 + "\nLIVE AUDIT\n" + "="*55)

code, d = get("/api/health")
chk("Health 200", code==200)

code, d = get("/api/ml/metrics")
chk("Metrics 200", code==200)
total = d.get("asd_model",{}).get("dataset_info",{}).get("total_records",0)
chk(f"Metrics total={total} (expect 15507)", total>=3100)

code, d = post("/api/ml/assess", {
    "age":7,"gender":"m","family_history":True,"jaundice":False,
    "answers":{"q_eye_contact":2,"q_repetitive_behavior":3,"q_routine":3,
               "q_special_interests":3,"q_social_interaction":2,"q_sensory_response":2}
})
chk("Assess 200", code==200)
prob = d.get("disorder_results",{}).get("asd",{}).get("probability",0)
prob_pct = d.get("disorder_results",{}).get("asd",{}).get("probability_pct",0)
chk(f"ASD probability={prob_pct}% (expect 70-92%)", 50 <= float(prob_pct or prob*100 or 0) <= 95)

code, d = post("/api/doctor-chat", {
    "messages":[{"role":"user","content":"What is stuttering and how is it treated?"}],
    "language":"en"
})
chk("Doctor chat 200", code==200)
reply = d.get("reply","")
chk("Doctor chat about stuttering", "stutter" in reply.lower() or "fluency" in reply.lower())
chk("Reply >100 chars", len(reply)>100)

code, d = post("/api/doctor-chat", {
    "messages":[{"role":"user","content":"What is Speech Sound Disorder SSD?"}],
    "language":"en"
})
reply2 = d.get("reply","")
chk("Doctor chat about SSD", "ssd" in reply2.lower() or "speech sound" in reply2.lower() or "articulation" in reply2.lower())

code, d = post("/api/media-analysis", {
    "fileDesc":"video of 6yo child","mediaPayloads":[],
    "clientMetrics":{"videos":[{"duration":5,"motionIntensity":40,"hasFaceFrame":True}]},
    "files":{"img":[],"vid":["child.mp4"],"aud":[]},"options":{"selectedDisorder":"all"}
})
chk("Media analysis 200", code==200)
probs = d.get("probs") or d.get("probabilities",{})
chk("Media has probs", bool(probs))

for page in ["/","/pages/assess.html","/pages/media-analysis.html",
             "/pages/doctor-chat.html","/pages/dataset.html"]:
    try:
        with urllib.request.urlopen(BASE+page, timeout=15) as r:
            chk(f"Page {page} 200", r.status==200)
    except:
        chk(f"Page {page} 200", False)

print("\n" + "="*55)
passed = sum(1 for s,_ in results if s=="OK")
failed = sum(1 for s,_ in results if s=="FAIL")
print(f"LIVE: {passed}/{passed+failed} passed  ({failed} failed)")
if failed:
    print("FAILED:")
    for s,l in results:
        if s=="FAIL": print(f"  - {l}")
sys.exit(0 if failed==0 else 1)
