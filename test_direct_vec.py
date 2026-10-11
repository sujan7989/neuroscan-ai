"""Test models with manually constructed feature vectors matching training ranges"""
import numpy as np, joblib, os, sys

ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'
ART  = ROOT + r'\ml\artifacts'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Test ASD: all 1s for AQ-10, family history=1
asd_pos = [1,1,1,1,1,1,1,1,1,1, 0.12, 1.0, 1.0, 1.0]  # 14 features
rf_asd = joblib.load(os.path.join(ART,'asd_rf.joblib'))
p = rf_asd.predict_proba(np.array(asd_pos,dtype=np.float32).reshape(1,-1))[0,1]
print(f"ASD all-1s:  RF={p:.3f} ({int(p*100)}%)")

# Test ADHD: max inattention + hyperactivity
adhd_pos = [4,4,4,4,4,4,  # 6 inatt
            4,4,4,4,4,4,  # 6 hyper
            90, 88,        # conners
            1.0, 1.0]     # family, school
rf_adhd = joblib.load(os.path.join(ART,'adhd_rf.joblib'))
p = rf_adhd.predict_proba(np.array(adhd_pos,dtype=np.float32).reshape(1,-1))[0,1]
print(f"ADHD high:   RF={p:.3f} ({int(p*100)}%)")

# Test Dyslexia: low scores (dyslexic pattern)
# phonol_aware=65, phonol_mem=70, wrf=65, nwr=62, reversals=1, spell=1, slow=1, fam=1, hw_speed=30, hw_cons=0.4, hw_rev=0.3, era=0.25
dys_pos = [65, 70, 65, 62, 1, 1, 1, 1, 30, 0.4, 0.3, 0.25]
rf_dys = joblib.load(os.path.join(ART,'dyslexia_rf.joblib'))
p = rf_dys.predict_proba(np.array(dys_pos,dtype=np.float32).reshape(1,-1))[0,1]
print(f"Dyslexia low:RF={p:.3f} ({int(p*100)}%)")

# Test Stuttering: high disfluency
stu_pos = [0.22, 5.5,   # disfl, rep
           4.0, 75,     # block, speech_rate
           0.60, 1.9, 1.5,  # phonat, jitter, shimmer  
           3, 3,         # severity, anxiety
           1.0, 1.0]    # family, onset
rf_stu = joblib.load(os.path.join(ART,'stuttering_rf.joblib'))
p = rf_stu.predict_proba(np.array(stu_pos,dtype=np.float32).reshape(1,-1))[0,1]
print(f"Stuttering:  RF={p:.3f} ({int(p*100)}%)")

# Test SSD: high error rates
ssd_pos = [65, 72,       # pcc, gfta3
           0.30, 0.45,   # stop, fric errors
           0.42, 0.50,   # liq, clus errors
           0.55, 0.45,   # intel, pros_dev
           1.0, 1.0, 1.0]  # late, phon_proc, family
rf_ssd = joblib.load(os.path.join(ART,'ssd_rf.joblib'))
p = rf_ssd.predict_proba(np.array(ssd_pos,dtype=np.float32).reshape(1,-1))[0,1]
print(f"SSD high:    RF={p:.3f} ({int(p*100)}%)")
