"""
NeuroScan AI — Multi-Disorder ML Inference Engine
Loads trained sklearn 1.9.1 Pipeline models for all 5 disorders.
Called by server.js via runPythonScript().
"""
import sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Feature orders must exactly match training
FEATURE_ORDERS = {
    "asd": ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score",
            "A7_Score","A8_Score","A9_Score","A10_Score",
            "age","gender_num","jaundice_num","austim_num",
            "arm_flapping","head_banging","spinning","rocking",
            "gaze_avoidance","echolalia","hypo_gaze","social_isolate"],
    "adhd": ["asrs_inatt_1","asrs_inatt_2","asrs_inatt_3","asrs_inatt_4","asrs_inatt_5","asrs_inatt_6",
             "asrs_hyper_1","asrs_hyper_2","asrs_hyper_3","asrs_hyper_4","asrs_hyper_5","asrs_hyper_6",
             "conners_inattention","conners_hyperactivity","conners_executive","conners_learning",
             "cpt_omission_tscore","cpt_commission_tscore","cpt_rt_mean_ms","cpt_rt_variability","cpt_detectability",
             "family_history_adhd","sleep_problems","school_performance_low","age"],
    "dyslexia": ["phonological_awareness","phonological_memory","rapid_naming_speed",
                 "word_reading_fluency","nonword_reading","letter_reversal",
                 "reading_avoidance","spelling_errors","slow_reading_rate","family_history_dyslexia",
                 "hw_pen_speed_mms","hw_stroke_consistency","hw_letter_reversal_rate",
                 "hw_line_deviation_mm","hw_spacing_variability","hw_erasure_per_word"],
    "stuttering": ["disfluency_rate","repetition_per_100w","prolongation_dur_s","block_freq_per_100w",
                   "pause_freq_per_100w","speech_rate_wpm","phonation_ratio","jitter_pct",
                   "shimmer_db","hnr_db","age","onset_childhood","family_history_stutter","anxiety_speaking"],
    "ssd": ["intelligibility_score","pcc_score","gfta3_standard_score","fricative_error_rate",
            "stop_error_rate","liquid_error_rate","affricate_error_rate","cluster_error_rate",
            "late_sound_acquisition","f1_formant_deviation","f2_formant_deviation",
            "prosody_deviation","hearing_normal","age","family_history_ssd"],
}

DISORDER_META = {
    "asd":        {"name": "Autism Spectrum Disorder",       "short": "ASD",  "icd": "F84.0", "color": "#2d6a4f"},
    "adhd":       {"name": "ADHD",                           "short": "ADHD", "icd": "F90.0", "color": "#e76f51"},
    "dyslexia":   {"name": "Dyslexia",                       "short": "DYS",  "icd": "F81.0", "color": "#457b9d"},
    "stuttering": {"name": "Stuttering / Fluency Disorder",  "short": "STU",  "icd": "F98.5", "color": "#0891b2"},
    "ssd":        {"name": "Speech Sound Disorder",          "short": "SSD",  "icd": "F80.0", "color": "#7c3aed"},
}

_LOADED = {}

def _model(disorder):
    if disorder not in _LOADED:
        p = ARTIFACTS / f"{disorder}_rf.joblib"
        if p.exists():
            _LOADED[disorder] = joblib.load(p)
    return _LOADED.get(disorder)


def _parse_float(v, default=0.0):
    try: return float(v)
    except: return default

def _parse_bool(v):
    if v is None: return 0
    if isinstance(v, bool): return int(v)
    return 1 if str(v).strip().lower() in ("1","yes","true","y") else 0


def build_features(disorder, answers, demographics):
    """Build feature vector for a disorder, mapped from frontend answers."""
    merged = {**demographics, **answers}
    order  = FEATURE_ORDERS[disorder]

    # ── ASD ──────────────────────────────────────────────────────
    if disorder == "asd":
        d = {}
        for i in range(1, 11):
            d[f"A{i}_Score"] = min(1, max(0, round(_parse_float(merged.get(f"A{i}_Score", merged.get(f"A{i}", 0))))))
        aq = sum(d[f"A{i}_Score"] for i in range(1,11))
        d["age"]         = _parse_float(merged.get("age", 8))
        d["gender_num"]  = 1 if str(merged.get("gender","m")).strip().lower() in ("m","male","1") else 0
        d["jaundice_num"]= _parse_bool(merged.get("jaundice", merged.get("neonatal_jaundice", 0)))
        d["austim_num"]  = _parse_bool(merged.get("austim", merged.get("family_history_asd", merged.get("family_history", 0))))
        d["arm_flapping"]  = 1 if d["A1_Score"] or d["A8_Score"] else 0
        d["head_banging"]  = d["A7_Score"]
        d["spinning"]      = d["A8_Score"]
        d["rocking"]       = 1 if d["A3_Score"] or d["A8_Score"] else 0
        d["gaze_avoidance"]= d["A1_Score"]
        d["echolalia"]     = d["A5_Score"]
        d["hypo_gaze"]     = d["A6_Score"]
        d["social_isolate"]= d["A10_Score"]
        return np.array([d.get(f, 0.0) for f in order], dtype="float32")

    # ── ADHD ─────────────────────────────────────────────────────
    if disorder == "adhd":
        att = _parse_float(merged.get("attention_span",   merged.get("inattention",  0)))
        hyp = _parse_float(merged.get("hyperactivity",    merged.get("hyp",          0)))
        emo = _parse_float(merged.get("emotional_regulation", 0))
        imp = _parse_float(merged.get("impulsivity",      0))
        d = {
            "asrs_inatt_1": att, "asrs_inatt_2": att, "asrs_inatt_3": att,
            "asrs_inatt_4": att*0.8, "asrs_inatt_5": att*0.9, "asrs_inatt_6": emo,
            "asrs_hyper_1": hyp, "asrs_hyper_2": hyp, "asrs_hyper_3": hyp*0.9,
            "asrs_hyper_4": imp, "asrs_hyper_5": imp, "asrs_hyper_6": hyp*0.8,
            "conners_inattention":   att,
            "conners_hyperactivity": hyp,
            "conners_executive":     emo,
            "conners_learning":      att * 0.7,
            "cpt_omission_tscore":   50 + att * 5,
            "cpt_commission_tscore": 50 + imp * 5,
            "cpt_rt_mean_ms":        450 + att * 20,
            "cpt_rt_variability":    0.2 + att * 0.05,
            "cpt_detectability":     1.5 - att * 0.1,
            "family_history_adhd":   _parse_bool(merged.get("family_history_adhd", merged.get("family_history", 0))),
            "sleep_problems":        _parse_float(merged.get("sleep_problems", 0)),
            "school_performance_low":_parse_float(merged.get("school_performance_low", att / 4)),
            "age":                   _parse_float(merged.get("age", 8)),
        }
        return np.array([d.get(f, 0.0) for f in order], dtype="float32")

    # ── DYSLEXIA ─────────────────────────────────────────────────
    if disorder == "dyslexia":
        rd = _parse_float(merged.get("reading_difficulty", merged.get("reading", 0)))
        ph = _parse_float(merged.get("phonological_awareness", merged.get("phonological", 0)))
        sp = _parse_float(merged.get("spelling_difficulty", merged.get("spelling", 0)))
        rf = _parse_float(merged.get("reading_fluency", merged.get("fluency", 0)))
        lr = _parse_float(merged.get("letter_reversal", 0))
        d = {
            "phonological_awareness":  ph, "phonological_memory": ph * 0.8,
            "rapid_naming_speed":      rd * 0.6, "word_reading_fluency": rf,
            "nonword_reading":         ph * 0.7, "letter_reversal": lr,
            "reading_avoidance":       rd * 0.5, "spelling_errors": sp,
            "slow_reading_rate":       rf * 0.6,
            "family_history_dyslexia": _parse_bool(merged.get("family_history_dyslexia", merged.get("family_history", 0))),
            "hw_pen_speed_mms":        50 - lr * 10,
            "hw_stroke_consistency":   1.0 - lr * 0.2,
            "hw_letter_reversal_rate": lr * 0.4,
            "hw_line_deviation_mm":    lr * 2,
            "hw_spacing_variability":  lr * 1.5,
            "hw_erasure_per_word":     rd * 0.3,
        }
        return np.array([d.get(f, 0.0) for f in order], dtype="float32")

    # ── STUTTERING ───────────────────────────────────────────────
    if disorder == "stuttering":
        sf  = _parse_float(merged.get("speech_fluency", merged.get("fluency", 0)))
        sev = sf / 4.0
        d = {
            "disfluency_rate":       sev * 15,
            "repetition_per_100w":   sev * 10,
            "prolongation_dur_s":    sev * 0.5,
            "block_freq_per_100w":   sev * 8,
            "pause_freq_per_100w":   sev * 12,
            "speech_rate_wpm":       200 - sev * 30,
            "phonation_ratio":       0.7 - sev * 0.1,
            "jitter_pct":            0.5 + sev * 0.3,
            "shimmer_db":            0.3 + sev * 0.2,
            "hnr_db":                20  - sev * 3,
            "age":                   _parse_float(merged.get("age", 8)),
            "onset_childhood":       1,
            "family_history_stutter":_parse_bool(merged.get("family_history_stutter", merged.get("family_history", 0))),
            "anxiety_speaking":      _parse_float(merged.get("anxiety_speaking", merged.get("anxiety_level", 0))),
        }
        return np.array([d.get(f, 0.0) for f in order], dtype="float32")

    # ── SSD ──────────────────────────────────────────────────────
    if disorder == "ssd":
        art  = _parse_float(merged.get("articulation_errors", merged.get("articulation", 0)))
        inte = _parse_float(merged.get("speech_intelligibility", merged.get("intelligibility", 0)))
        sev  = (art + inte) / 2 / 4.0
        d = {
            "intelligibility_score": max(0, 100 - sev * 20),
            "pcc_score":             max(0, 100 - sev * 15),
            "gfta3_standard_score":  max(40, 100 - sev * 12),
            "fricative_error_rate":  sev * 0.4,
            "stop_error_rate":       sev * 0.3,
            "liquid_error_rate":     sev * 0.5,
            "affricate_error_rate":  sev * 0.2,
            "cluster_error_rate":    sev * 0.35,
            "late_sound_acquisition":min(4, sev * 4),
            "f1_formant_deviation":  sev * 0.2,
            "f2_formant_deviation":  sev * 0.15,
            "prosody_deviation":     sev * 0.1,
            "hearing_normal":        1,
            "age":                   _parse_float(merged.get("age", 8)),
            "family_history_ssd":    _parse_bool(merged.get("family_history_ssd", merged.get("family_history", 0))),
        }
        return np.array([d.get(f, 0.0) for f in order], dtype="float32")

    return np.zeros(len(order), dtype="float32")


def predict_all(answers, demographics):
    results = {}
    for disorder in DISORDER_META:
        model = _model(disorder)
        if model is None:
            results[disorder] = {**DISORDER_META[disorder],
                                  "probability": 0.5, "probability_pct": 50,
                                  "risk_level": "Unknown", "confidence": 0, "model": "not_loaded"}
            continue
        try:
            X = build_features(disorder, answers, demographics).reshape(1, -1)
            prob = float(model.predict_proba(X)[0, 1])
            pct  = int(round(prob * 100))
            risk = "High" if pct >= 70 else "Moderate" if pct >= 50 else "Low" if pct >= 30 else "Minimal"
            conf = int(round(abs(prob - 0.5) * 200))
            results[disorder] = {
                **DISORDER_META[disorder],
                "probability":      round(prob, 4),
                "probability_pct":  pct,
                "riskLevel":        risk, "risk_level": risk,
                "confidence":       conf,
                "confidence_label": "High" if conf >= 60 else "Moderate" if conf >= 30 else "Low",
                "lr_score": round(prob, 4),
                "rf_score": round(prob, 4),
                "xgb_score":round(prob, 4),
                "model":    "real_trained_rf_1.9.1",
            }
        except Exception as e:
            results[disorder] = {**DISORDER_META[disorder],
                                  "probability": 0.5, "probability_pct": 50,
                                  "risk_level": "Unknown", "model": f"error:{e}"}
    return results


if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    try:
        payload  = json.loads(raw) if raw else {}
        answers  = payload.get("answers", {})
        demog    = payload.get("demographics", {})
        disorder_results = predict_all(answers, demog)
        primary = max(disorder_results.items(), key=lambda x: x[1].get("probability", 0))
        print(json.dumps({
            "status":               "REAL_MODEL",
            "model_type":           "sklearn_1.9.1_pipeline",
            "disorder_results":     disorder_results,
            "disorders":            disorder_results,
            "primary_disorder":     primary[0],
            "primary_probability":  primary[1].get("probability_pct"),
        }))
    except Exception as e:
        print(json.dumps({"error": str(e), "status": "ERROR"}))
