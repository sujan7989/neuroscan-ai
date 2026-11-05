"""
End-to-end application check for NeuroScan AI.
Tests every page, API endpoint, ML prediction, and data quality.
Run: python e2e_check.py
"""
import json, sys
import urllib.request, urllib.error

BASE = "http://localhost:3000"
PASS = []
FAIL = []

def get(path, label=None):
    label = label or path
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
            body = r.read().decode()
            status = r.status
        if status == 200:
            PASS.append(f"✅ {status}  {label}")
            return body
        else:
            FAIL.append(f"❌ {status}  {label}")
            return None
    except Exception as e:
        FAIL.append(f"❌ ERR  {label}  ({e})")
        return None

def post(path, data, label=None):
    label = label or path
    try:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{BASE}{path}", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
            status = r.status
        if status in (200, 201):
            PASS.append(f"✅ {status}  {label}")
            return json.loads(body)
        else:
            FAIL.append(f"❌ {status}  {label}")
            return None
    except Exception as e:
        FAIL.append(f"❌ ERR  {label}  ({e})")
        return None

print("=" * 60)
print("NeuroScan AI — End-to-End Check")
print("=" * 60)

# ── PAGES ─────────────────────────────────────────────────────
print("\n[1] HTML Pages")
pages = [
    "/index.html",
    "/pages/assess.html",
    "/pages/dashboard.html",
    "/pages/dataset.html",
    "/pages/model-metrics.html",
    "/pages/doctor-chat.html",
    "/pages/chat.html",
    "/pages/media-analysis.html",
    "/pages/history.html",
    "/pages/profile.html",
    "/pages/about.html",
    "/pages/predict.html",
    "/pages/speech-analysis.html",
    "/pages/drawing-analysis.html",
    "/pages/cognitive-battery.html",
    "/pages/learning-assessment.html",
    "/pages/learning-profile.html",
    "/pages/learning-program.html",
    "/pages/learning-progress.html",
    "/pages/tracker.html",
    "/pages/recommendations.html",
    "/pages/multimodal-dashboard.html",
    "/pages/emotion-detect.html",
    "/pages/auth.html",
]
for p in pages:
    body = get(p)
    if body and len(body) < 200:
        FAIL.append(f"❌ EMPTY  {p}  (only {len(body)} bytes — blank page)")
        PASS.pop()

# ── HEALTH & CONFIG ───────────────────────────────────────────
print("\n[2] Core API Endpoints")
h = get("/api/health", "GET /api/health")
if h:
    hj = json.loads(h)
    assert hj.get("status") == "ok", "Health status not ok"

get("/api/config", "GET /api/config")

# ── ML METRICS ───────────────────────────────────────────────
print("\n[3] ML Metrics Endpoints")
m = get("/api/ml/metrics", "GET /api/ml/metrics")
dm = get("/api/ml/detailed-metrics", "GET /api/ml/detailed-metrics")

if dm:
    dmj = json.loads(dm)
    disorders = list(dmj.get("disorders", {}).keys())
    expected = ["asd", "adhd", "dyslexia", "stuttering", "ssd"]
    missing = [d for d in expected if d not in disorders]
    if missing:
        FAIL.append(f"❌ detailed-metrics missing disorders: {missing}")
    else:
        PASS.append(f"✅ detailed-metrics has all 5 disorders")

    # Check each disorder has ROC, CM, CV data
    for d in expected:
        if d in dmj.get("disorders", {}):
            rf = dmj["disorders"][d]["models"].get("random_forest", {})
            checks = {
                "roc_curve_points": rf.get("roc_curve_points"),
                "confusion_matrix": rf.get("confusion_matrix"),
                "cv_5fold":         rf.get("cv_5fold"),
                "precision":        rf.get("precision"),
                "recall":           rf.get("recall"),
                "f1_score":         rf.get("f1_score"),
                "specificity":      rf.get("specificity"),
            }
            missing_fields = [k for k, v in checks.items() if not v]
            if missing_fields:
                FAIL.append(f"❌ {d}/random_forest missing: {missing_fields}")
            else:
                PASS.append(f"✅ {d}: all 7 metric fields present")

# ── ML PREDICTION ────────────────────────────────────────────
print("\n[4] ML Prediction (ASD)")
strong_asd = {
    "A1_Score":1,"A2_Score":1,"A3_Score":0,"A4_Score":1,"A5_Score":1,
    "A6_Score":0,"A7_Score":1,"A8_Score":1,"A9_Score":0,"A10_Score":1,
    "age":12,"gender":"m","jaundice":0,"family_history":1
}
pred = post("/api/ml/predict", strong_asd, "POST /api/ml/predict (strong ASD)")
if pred:
    prob = pred.get("probability", 0)
    if prob >= 60:
        PASS.append(f"✅ ASD prediction probability={prob}% (expected ≥60%)")
    else:
        FAIL.append(f"❌ ASD prediction too low: {prob}% (expected ≥60%)")

# Low ASD profile
low_asd = {
    "A1_Score":0,"A2_Score":0,"A3_Score":0,"A4_Score":0,"A5_Score":0,
    "A6_Score":0,"A7_Score":0,"A8_Score":0,"A9_Score":0,"A10_Score":0,
    "age":25,"gender":"f","jaundice":0,"family_history":0
}
pred2 = post("/api/ml/predict", low_asd, "POST /api/ml/predict (low risk)")
if pred2:
    prob2 = pred2.get("probability", 100)
    if prob2 <= 40:
        PASS.append(f"✅ Low-risk prediction probability={prob2}% (expected ≤40%)")
    else:
        FAIL.append(f"❌ Low-risk prediction too high: {prob2}% (expected ≤40%)")

# ── MULTI-DISORDER ASSESS ────────────────────────────────────
print("\n[5] Multi-Disorder Assessment")
assess_payload = {
    "answers": {
        "A1_Score":1,"A2_Score":1,"A3_Score":0,"A4_Score":1,
        "A5_Score":1,"A6_Score":0,"A7_Score":1,"A8_Score":1,
        "A9_Score":0,"A10_Score":1
    },
    "demographics": {"age":12,"gender":"m"}
}
assess = post("/api/ml/assess", assess_payload, "POST /api/ml/assess")
if assess:
    disorders_out = assess.get("disorders", assess.get("results", {}))
    if disorders_out:
        PASS.append(f"✅ /api/ml/assess returned {len(disorders_out)} disorder scores")
    else:
        FAIL.append(f"❌ /api/ml/assess returned empty disorders")

# ── AUTH ENDPOINTS ───────────────────────────────────────────
print("\n[6] Auth & User Endpoints")
auth = post("/api/auth/verify", {}, "POST /api/auth/verify (dev mode)")

# ── ASSESSMENTS ──────────────────────────────────────────────
print("\n[7] Assessment CRUD")
asm_payload = {
    "name":"Test User","age":14,"gender":"m","ethnicity":"White-European",
    "country":"India","family_history":False,"jaundice":False,
    "answers":{},"disorder_results":{"ASD":{"score":72,"risk":"High"}},
    "probability":72,"aq10_sum":7,"risk_level":"High","assessment_type":"multi-disorder"
}
asm = post("/api/assessments", asm_payload, "POST /api/assessments")
if asm:
    aid = asm.get("id","")
    if aid:
        single = get(f"/api/assessments/id/{aid}", f"GET /api/assessments/id/{aid}")
        list_r = get("/api/assessments/demo-user-001", "GET /api/assessments/:uid")

# ── CHAT ─────────────────────────────────────────────────────
print("\n[8] Chat Endpoints")
chat = post("/api/chats", {"role":"user","text":"What is ASD?"}, "POST /api/chats")
get("/api/chats/demo-user-001", "GET /api/chats/:uid")

# ── DOCTOR CHAT ──────────────────────────────────────────────
print("\n[9] Doctor Chat API")
dc_payload = {"message":"What are symptoms of ADHD?","conversationHistory":[]}
dc = post("/api/doctor-chat", dc_payload, "POST /api/doctor-chat")
if dc:
    reply = dc.get("reply","")
    if len(reply) > 20:
        PASS.append(f"✅ Doctor chat returned {len(reply)} char reply")
    else:
        FAIL.append(f"❌ Doctor chat reply too short: '{reply[:60]}'")

# ── LEARNING ROUTES ──────────────────────────────────────────
print("\n[10] Learning Routes")
get("/api/learning/modules", "GET /api/learning/modules")
get("/api/learning/abilities", "GET /api/learning/abilities")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for r in PASS: print(r)
print()
for r in FAIL: print(r)
print()
print(f"PASSED: {len(PASS)}  FAILED: {len(FAIL)}  TOTAL: {len(PASS)+len(FAIL)}")
if FAIL:
    print("\n⚠️  Issues need fixing before deploy")
    sys.exit(1)
else:
    print("\n🎉 All checks passed — ready to deploy!")
    sys.exit(0)
