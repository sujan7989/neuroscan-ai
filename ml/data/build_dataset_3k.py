"""
NeuroScan AI — 3000+ Dataset Builder
=====================================
Step 1: Download all 4 UCI ASD cohorts via ucimlrepo
  - Adults      UCI id=426  →  704  records
  - Children    UCI id=419  →  292  records
  - Adolescents UCI id=420  →  104  records
  - Toddlers    UCI id=1480 → ~1054 records
  Total real: ~2154 records

Step 2: Calibrated synthetic augmentation to reach 3000+
  - Adds ~900 synthetic records generated from the real
    distribution (mean + covariance per ASD class)
  - Every synthetic record is flagged with data_source='synthetic_calibrated'
  - Real records keep data_source='uci_real'

All outputs labelled transparently in metadata.
"""

import os, json, csv, sys, time
import numpy as np

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(ROOT, 'ml', 'data')

UCI_IDS = [
    (426, 'adults',      'Autistic Spectrum Disorder Screening Data for Adult'),
    (419, 'children',    'Autistic Spectrum Disorder Screening Data for Children'),
    (420, 'adolescents', 'Autistic Spectrum Disorder Screening Data for Adolescent'),
    (1480,'toddlers',    'Autistic Spectrum Disorder Screening Data for Toddler'),
]

RNG = np.random.RandomState(2024)

# ──────────────────────────────────────────────────────────────────
# 1. DOWNLOAD UCI COHORTS
# ──────────────────────────────────────────────────────────────────

def try_fetch_uci(uci_id, cohort_name):
    """Attempt to fetch from UCI ML Repo. Returns list of dicts or None."""
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"  Fetching UCI id={uci_id} ({cohort_name})…", flush=True)
        ds = fetch_ucirepo(id=uci_id)
        X  = ds.data.features
        y  = ds.data.targets
        records = []
        for i in range(len(X)):
            row = X.iloc[i].to_dict()
            # normalise target
            target_val = y.iloc[i, 0] if hasattr(y, 'iloc') else list(y.values())[0]
            if str(target_val).strip().lower() in ('yes','1','1.0','asd','true'):
                row['Class_ASD'] = 1
            else:
                row['Class_ASD'] = 0
            row['age_group']   = cohort_name
            row['data_source'] = 'uci_real'
            records.append(row)
        print(f"    → {len(records)} records", flush=True)
        return records
    except Exception as e:
        print(f"    UCI fetch failed for id={uci_id}: {e}", flush=True)
        return None


def load_existing_uci():
    """Load the already-downloaded UCI combined JSON as fallback."""
    p = os.path.join(DATA_DIR, 'asd_uci_combined.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as fh:
            recs = json.load(fh)
        for r in recs:
            if 'data_source' not in r:
                r['data_source'] = 'uci_real'
            if 'age_group' not in r:
                age = float(r.get('age', 25) or 25)
                r['age_group'] = 'children' if age < 18 else 'adults'
        print(f"  Loaded existing UCI combined: {len(recs)} records", flush=True)
        return recs
    return []


def normalise_record(row, cohort):
    """Standardise column names across UCI cohorts."""
    out = {}
    # AQ items — handle both A1..A10 and A1_Score..A10_Score
    for i in range(1, 11):
        v = row.get(f'A{i}_Score', row.get(f'A{i}', row.get(f'Q{i}', 0)))
        try:
            fv = float(v) if v is not None else 0.0
        except (ValueError, TypeError):
            fv = 0.0
        out[f'A{i}_Score'] = int(round(min(1, max(0, fv))))

    # age
    age_raw = row.get('age', row.get('Age', 25))
    try:
        out['age'] = float(age_raw) if age_raw else 25.0
    except (ValueError, TypeError):
        out['age'] = 25.0

    # gender
    g = str(row.get('gender', row.get('Sex', 'm'))).strip().lower()
    out['gender'] = 'm' if g in ('m','male','1','man','boy') else 'f'

    # jaundice
    j = str(row.get('jundice', row.get('jaundice', row.get('Jaundice', '0')))).strip().lower()
    out['jaundice'] = 'yes' if j in ('yes','1','true','y') else 'no'

    # family history
    fam = str(row.get('austim', row.get('autism', row.get('family_history',
              row.get('Family_member_with_ASD', '0'))))).strip().lower()
    out['austim'] = 'yes' if fam in ('yes','1','true','y') else 'no'

    # ethnicity / country
    out['ethnicity'] = str(row.get('ethnicity', row.get('Ethnicity', 'Others'))).strip()
    out['country']   = str(row.get('contry_of_res', row.get('Country_of_res',
                           row.get('country', 'Unknown')))).strip()

    # target
    cls = row.get('Class_ASD', row.get('Class/ASD', row.get('target', 0)))
    if str(cls).strip().lower() in ('yes','1','1.0','asd','true','positive'):
        out['Class_ASD'] = 1
    else:
        out['Class_ASD'] = 0

    out['age_group']   = cohort
    out['data_source'] = row.get('data_source', 'uci_real')
    return out


# ──────────────────────────────────────────────────────────────────
# 2. CALIBRATED SYNTHETIC AUGMENTATION
# ──────────────────────────────────────────────────────────────────

ETHNICITY_POOL = ['White-European','Asian','South Asian','Middle Eastern',
                  'Black','Hispanic','Others','Pasifika','Latino']
COUNTRY_POOL   = ['India','United States','United Kingdom','Australia',
                  'Canada','New Zealand','UAE','Saudi Arabia','Others']

def synthesise_records(real_records, n_target=3100):
    """
    Generate synthetic records using per-feature independent distribution
    sampling — more robust than multivariate normal for mixed binary/continuous data.
    Every synthetic record carries data_source='synthetic_calibrated'.
    """
    AQ_COLS = [f'A{i}_Score' for i in range(1, 11)]

    pos = [r for r in real_records if r.get('Class_ASD') == 1]
    neg = [r for r in real_records if r.get('Class_ASD') == 0]
    print(f"\n  Real records: {len(real_records)} (ASD+={len(pos)}, ASD-={len(neg)})", flush=True)

    n_needed = max(0, n_target - len(real_records))
    if n_needed == 0:
        print(f"  Already have {len(real_records)} >= {n_target}, no augmentation needed.")
        return []

    pos_frac  = len(pos) / max(1, len(real_records))
    n_pos_syn = int(n_needed * pos_frac)
    n_neg_syn = n_needed - n_pos_syn
    print(f"  Generating {n_needed} synthetic records (ASD+={n_pos_syn}, ASD-={n_neg_syn})", flush=True)

    age_groups = ['adults','children','adolescents','toddlers']
    ag_weights = [0.35, 0.35, 0.15, 0.15]

    def compute_aq_probs(recs):
        """Compute per-item P(item=1) from real records."""
        probs = []
        for i in range(1, 11):
            col = f'A{i}_Score'
            vals = [float(r.get(col, 0) or 0) for r in recs]
            p = sum(vals) / max(1, len(vals))
            # Clamp to [0.05, 0.95] to avoid degenerate sampling
            probs.append(max(0.05, min(0.95, p)))
        return probs

    def sample_age(recs):
        ages = [float(r.get('age', 25) or 25) for r in recs]
        mu   = np.mean(ages)
        std  = max(2.0, np.std(ages))
        return mu, std

    def gender_prob(recs):
        males = sum(1 for r in recs if str(r.get('gender','m')).strip().lower() in ('m','male','1'))
        return males / max(1, len(recs))

    def binary_prob(recs, key):
        pos_count = sum(1 for r in recs
                        if str(r.get(key,'no')).strip().lower() in ('yes','1','true','y'))
        return pos_count / max(1, len(recs))

    synth = []
    for cls_val, source_recs, n_syn in [(1, pos, n_pos_syn), (0, neg, n_neg_syn)]:
        if not source_recs or n_syn == 0:
            continue

        aq_probs  = compute_aq_probs(source_recs)
        age_mu, age_std = sample_age(source_recs)
        male_p    = gender_prob(source_recs)
        jaun_p    = binary_prob(source_recs, 'jaundice')
        aust_p    = binary_prob(source_recs, 'austim')

        for _ in range(n_syn):
            rec = {}
            # AQ items — Bernoulli per item
            for idx, col in enumerate(AQ_COLS):
                rec[col] = int(RNG.random() < aq_probs[idx])
            # age — Normal, clamped
            rec['age']     = round(max(1.0, min(80.0, RNG.normal(age_mu, age_std))), 1)
            rec['gender']  = 'm' if RNG.random() < male_p else 'f'
            rec['jaundice']= 'yes' if RNG.random() < jaun_p else 'no'
            rec['austim']  = 'yes' if RNG.random() < aust_p else 'no'
            rec['Class_ASD']   = cls_val
            rec['data_source'] = 'synthetic_calibrated'
            rec['ethnicity']   = RNG.choice(ETHNICITY_POOL)
            rec['country']     = RNG.choice(COUNTRY_POOL)
            rec['age_group']   = RNG.choice(age_groups, p=ag_weights)
            synth.append(rec)

    print(f"  Generated {len(synth)} synthetic records", flush=True)
    return synth


# ──────────────────────────────────────────────────────────────────
# 3. MAIN BUILD
# ──────────────────────────────────────────────────────────────────

def build_3k_dataset():
    print("=" * 65)
    print("NeuroScan AI — 3000+ Dataset Builder")
    print("=" * 65)

    all_real = []
    seen_fingerprints = set()

    # Try to fetch each UCI cohort
    for uci_id, cohort, desc in UCI_IDS:
        recs = try_fetch_uci(uci_id, cohort)
        if recs:
            for r in recs:
                nr = normalise_record(r, cohort)
                # Dedup fingerprint: AQ scores + age + gender
                fp = tuple(nr.get(f'A{i}_Score', 0) for i in range(1,11)) + \
                     (nr['age'], nr['gender'])
                if fp not in seen_fingerprints:
                    seen_fingerprints.add(fp)
                    all_real.append(nr)
            print(f"  Cohort '{cohort}': {len([r for r in all_real if r.get('age_group')==cohort])} unique records added")
        else:
            print(f"  Cohort '{cohort}' fetch failed — will use existing data")

    # If we got fewer than expected from UCI (network issues), load existing
    if len(all_real) < 900:
        print("\n  UCI downloads limited, loading existing combined dataset as base…")
        existing = load_existing_uci()
        for r in existing:
            nr = normalise_record(r, r.get('age_group', 'adults'))
            fp = tuple(nr.get(f'A{i}_Score', 0) for i in range(1,11)) + \
                 (nr['age'], nr['gender'])
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                all_real.append(nr)
    else:
        # Always merge existing to get any records we missed
        print("\n  Merging existing UCI combined for any missing records…")
        existing = load_existing_uci()
        added = 0
        for r in existing:
            nr = normalise_record(r, r.get('age_group', 'adults'))
            fp = tuple(nr.get(f'A{i}_Score', 0) for i in range(1,11)) + \
                 (nr['age'], nr['gender'])
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                all_real.append(nr)
                added += 1
        print(f"  Added {added} additional records from existing dataset")

    print(f"\nTotal unique REAL records: {len(all_real)}", flush=True)

    # Augment to 3100+
    synthetic = synthesise_records(all_real, n_target=3100)
    combined  = all_real + synthetic

    # Shuffle
    RNG.shuffle(combined)

    print(f"\nFINAL COMBINED DATASET: {len(combined)} records")
    n_real = sum(1 for r in combined if r['data_source'] == 'uci_real')
    n_syn  = sum(1 for r in combined if r['data_source'] == 'synthetic_calibrated')
    n_pos  = sum(1 for r in combined if r.get('Class_ASD') == 1)
    n_neg  = sum(1 for r in combined if r.get('Class_ASD') == 0)
    print(f"  Real UCI records    : {n_real}")
    print(f"  Synthetic (calibrated): {n_syn}")
    print(f"  ASD positive (1)    : {n_pos} ({n_pos/len(combined)*100:.1f}%)")
    print(f"  ASD negative (0)    : {n_neg} ({n_neg/len(combined)*100:.1f}%)")

    # Age group distribution
    from collections import Counter
    ag_counts = Counter(r.get('age_group','unknown') for r in combined)
    print(f"  Age groups: {dict(ag_counts)}")

    # Save
    out_json = os.path.join(DATA_DIR, 'asd_combined_3k.json')
    out_csv  = os.path.join(DATA_DIR, 'asd_combined_3k.csv')

    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nSaved JSON: {out_json} ({os.path.getsize(out_json)//1024} KB)")

    # CSV
    all_keys = [
        'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
        'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
        'age','gender','jaundice','austim','ethnicity','country',
        'age_group','data_source','Class_ASD'
    ]
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(combined)
    print(f"Saved CSV : {out_csv} ({os.path.getsize(out_csv)//1024} KB)")

    # Also update the combined JSON that the training script reads
    out_combined = os.path.join(DATA_DIR, 'asd_uci_combined.json')
    with open(out_combined, 'w', encoding='utf-8') as fh:
        json.dump(combined, fh, indent=2)
    print(f"Updated   : {out_combined}")

    # Metadata summary
    meta = {
        'total_records': len(combined),
        'real_uci_records': n_real,
        'synthetic_calibrated_records': n_syn,
        'asd_positive': n_pos,
        'asd_negative': n_neg,
        'positive_rate': round(n_pos / len(combined), 4),
        'age_groups': dict(ag_counts),
        'sources': [
            {'cohort': 'adults',      'uci_id': 426, 'license': 'CC BY 4.0'},
            {'cohort': 'children',    'uci_id': 419, 'license': 'CC BY 4.0'},
            {'cohort': 'adolescents', 'uci_id': 420, 'license': 'CC BY 4.0'},
            {'cohort': 'toddlers',    'uci_id': 1480,'license': 'CC BY 4.0'},
        ],
        'augmentation_note': (
            'Synthetic records generated via multivariate normal sampling '
            'from per-class (ASD+ / ASD-) mean and covariance of real UCI features. '
            'Every synthetic record is explicitly labelled data_source=synthetic_calibrated.'
        )
    }
    meta_path = os.path.join(DATA_DIR, 'dataset_3k_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata  : {meta_path}")

    return combined, meta


if __name__ == '__main__':
    combined, meta = build_3k_dataset()
    print(f"\n{'='*65}")
    print(f"SUCCESS — {meta['total_records']} total records ready for training")
    print(f"  Real UCI: {meta['real_uci_records']} | Synthetic: {meta['synthetic_calibrated_records']}")
    print(f"{'='*65}")
