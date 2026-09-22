"""
NeuroScan AI — Real Dataset Builder
Fetches all 3 UCI ASD screening datasets (Adults, Children, Adolescents)
and merges them into one clean combined dataset.

Sources (CC BY 4.0 — Fadi Thabtah, 2017):
  Adults:      UCI id=426  (704 records)
  Children:    UCI id=419  (292 records)
  Adolescents: UCI id=420  (104 records)

Total: 1100 real clinical records
"""

import json
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

FEATURE_COLUMNS = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'country', 'used_app_before', 'result', 'relation', 'Class_ASD',
    'age_group'   # NEW: toddler/child/adolescent/adult
]


def clean_val(v):
    """Normalise a value from the UCI dataframe."""
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ('nan', 'none', '?', ''):
        return None
    return s


def parse_aq(v):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


def age_group(age):
    try:
        a = int(float(str(age).strip()))
    except Exception:
        return 'adult'
    if a <= 3:
        return 'toddler'
    if a <= 12:
        return 'child'
    if a <= 17:
        return 'adolescent'
    return 'adult'


def fetch_uci_datasets():
    """Download all 3 UCI ASD datasets using ucimlrepo."""
    from ucimlrepo import fetch_ucirepo

    datasets = []

    uci_sources = [
        (426, 'adult',       'Autism Screening Adult (UCI id=426)'),
        (419, 'child',       'Autistic Spectrum Disorder Screening Data for Children (UCI id=419)'),
        (420, 'adolescent',  'Autistic Spectrum Disorder Screening Data for Adolescent (UCI id=420)'),
    ]

    for uci_id, grp, desc in uci_sources:
        print(f"  Fetching {desc} ...")
        try:
            ds = fetch_ucirepo(id=uci_id)
            X = ds.data.features
            y = ds.data.targets
            print(f"    -> {len(X)} records, columns: {list(X.columns)[:5]}...")

            for idx in range(len(X)):
                row_X = X.iloc[idx]
                row_y = y.iloc[idx] if y is not None else None

                # AQ-10 scores
                aq = [parse_aq(row_X.get(f'A{i}_Score', 0)) for i in range(1, 11)]

                # Demographics
                raw_age = clean_val(row_X.get('age', ''))
                try:
                    age_val = int(float(raw_age)) if raw_age else (8 if grp == 'child' else 14 if grp == 'adolescent' else 30)
                    # Clamp to clinically valid range
                    if age_val < 1 or age_val > 100:
                        age_val = 8 if grp == 'child' else 14 if grp == 'adolescent' else 30
                except Exception:
                    age_val = 8 if grp == 'child' else 14 if grp == 'adolescent' else 30

                gender = clean_val(row_X.get('gender', 'm')) or 'm'
                gender = 'm' if str(gender).lower() in ('m', 'male', '1') else 'f'

                ethnicity = clean_val(row_X.get('ethnicity', 'Other')) or 'Other'
                jaundice  = 'yes' if str(clean_val(row_X.get('jundice', row_X.get('jaundice', 'no'))) or 'no').lower() in ('yes', '1', 'true') else 'no'
                austim    = 'yes' if str(clean_val(row_X.get('austim', 'no')) or 'no').lower() in ('yes', '1', 'true') else 'no'
                country   = clean_val(row_X.get('contry_of_res', row_X.get('country_of_res', 'Unknown'))) or 'Unknown'
                # Strip surrounding quotes that some UCI rows include
                country = country.strip("'\"").strip()
                used_app  = 'yes' if str(clean_val(row_X.get('used_app_before', 'no')) or 'no').lower() in ('yes', '1', 'true') else 'no'
                result    = int(float(clean_val(row_X.get('result', sum(aq))) or sum(aq)))
                relation  = clean_val(row_X.get('relation', 'Self')) or 'Self'

                # Target
                if row_y is not None:
                    raw_target = str(row_y.iloc[0]).strip().upper()
                    class_asd = 1 if raw_target in ('YES', '1', 'TRUE', 'ASD') else 0
                else:
                    class_asd = 1 if sum(aq) >= 6 else 0

                record = {
                    'A1_Score': aq[0], 'A2_Score': aq[1], 'A3_Score': aq[2],
                    'A4_Score': aq[3], 'A5_Score': aq[4], 'A6_Score': aq[5],
                    'A7_Score': aq[6], 'A8_Score': aq[7], 'A9_Score': aq[8],
                    'A10_Score': aq[9],
                    'age': age_val,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'jaundice': jaundice,
                    'austim': austim,
                    'country': country,
                    'used_app_before': used_app,
                    'result': result,
                    'relation': relation,
                    'Class_ASD': class_asd,
                    'age_group': age_group(age_val),
                }
                datasets.append(record)

        except Exception as e:
            print(f"    ERROR fetching UCI id={uci_id}: {e}")

    return datasets


def export_dataset(records, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    csv_path  = os.path.join(output_dir, 'asd_uci_combined.csv')
    json_path = os.path.join(output_dir, 'asd_uci_combined.json')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    # Stats
    pos = sum(1 for r in records if r['Class_ASD'] == 1)
    neg = sum(1 for r in records if r['Class_ASD'] == 0)
    males   = sum(1 for r in records if r['gender'] == 'm')
    females = sum(1 for r in records if r['gender'] == 'f')
    groups  = {}
    for r in records:
        g = r['age_group']
        groups[g] = groups.get(g, 0) + 1

    print(f"\n=== COMBINED UCI ASD DATASET ===")
    print(f"  Total records : {len(records)}")
    print(f"  Positive ASD=1: {pos} ({pos/len(records)*100:.1f}%)")
    print(f"  Negative ASD=0: {neg} ({neg/len(records)*100:.1f}%)")
    print(f"  Males: {males} | Females: {females}")
    print(f"  Age groups: {groups}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")

    return csv_path, json_path


if __name__ == '__main__':
    print("Fetching real UCI ASD datasets (CC BY 4.0)...")
    records = fetch_uci_datasets()
    if records:
        export_dataset(records)
    else:
        print("ERROR: No records fetched. Check internet connection.")
        sys.exit(1)
