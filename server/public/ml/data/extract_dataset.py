"""
NeuroScan AI — Dataset Extractor
Primary source: Real UCI ASD Screening datasets (CC BY 4.0, Fadi Thabtah 2017)

  Adults:   UCI id=426  704 records
  Children: UCI id=419  292 records
  Combined:             996 records  (stored in asd_uci_combined.json/csv)

The combined file is the PRIMARY dataset for model training.
The original 80-record Kaggle file (asd_kaggle_data.json) is kept for
reference and backward compatibility only.

Data provenance:
  Thabtah, F. (2017). Autism Screening Adult [Dataset].
    UCI ML Repository. https://doi.org/10.24432/C5F019
  Thabtah, F. (2017). Autistic Spectrum Disorder Screening Data for Children.
    UCI ML Repository. https://doi.org/10.24432/C5659W
  License: Creative Commons Attribution 4.0 (CC BY 4.0)
"""

import json
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.dirname(__file__)

FEATURE_COLUMNS = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'country', 'used_app_before', 'result', 'relation', 'Class_ASD',
    'age_group'
]


def load_combined_dataset(output_dir=OUTPUT_DIR):
    """
    Load the real UCI combined dataset.  If it doesn't exist yet, build it
    by downloading from UCI ML Repository.
    """
    combined_json = os.path.join(output_dir, 'asd_uci_combined.json')
    combined_csv  = os.path.join(output_dir, 'asd_uci_combined.csv')

    if not os.path.exists(combined_json):
        print("Combined dataset not found — fetching from UCI ML Repository...")
        try:
            from ml.data.build_dataset import fetch_uci_datasets, export_dataset
            records = fetch_uci_datasets()
            export_dataset(records, output_dir)
        except Exception as e:
            print(f"WARNING: Could not fetch from UCI ({e}). Falling back to original 80-record Kaggle file.")
            return _load_legacy_dataset(output_dir)

    with open(combined_json, encoding='utf-8') as f:
        records = json.load(f)

    print(f"Loaded real UCI combined dataset: {len(records)} records")
    return records, combined_csv, combined_json


def _load_legacy_dataset(output_dir=OUTPUT_DIR):
    """Fallback: load original 80-record Kaggle file."""
    legacy_json = os.path.join(output_dir, 'asd_kaggle_data.json')
    legacy_csv  = os.path.join(output_dir, 'asd_kaggle_data.csv')

    if not os.path.exists(legacy_json):
        raise FileNotFoundError(f"Neither combined nor legacy dataset found in {output_dir}")

    with open(legacy_json, encoding='utf-8') as f:
        records = json.load(f)

    # Add age_group field if missing
    for r in records:
        if 'age_group' not in r:
            age = r.get('age', 25)
            try:
                age = int(float(str(age)))
            except Exception:
                age = 25
            if age <= 3:
                r['age_group'] = 'toddler'
            elif age <= 12:
                r['age_group'] = 'child'
            elif age <= 17:
                r['age_group'] = 'adolescent'
            else:
                r['age_group'] = 'adult'

    print(f"Loaded legacy Kaggle dataset: {len(records)} records (fallback)")
    return records, legacy_csv, legacy_json


def export_dataset(output_dir=OUTPUT_DIR):
    """
    Main entry point used by train_asd_models.py.
    Returns (records, csv_path, json_path).
    """
    return load_combined_dataset(output_dir)


def get_dataset_stats(records):
    """Return a statistics dict for the loaded dataset."""
    pos = [r for r in records if r['Class_ASD'] == 1]
    neg = [r for r in records if r['Class_ASD'] == 0]

    age_groups = {}
    for r in records:
        g = r.get('age_group', 'adult')
        age_groups[g] = age_groups.get(g, 0) + 1

    sex_pos = {'m': sum(1 for r in pos if r['gender'] == 'm'),
               'f': sum(1 for r in pos if r['gender'] == 'f')}
    sex_neg = {'m': sum(1 for r in neg if r['gender'] == 'm'),
               'f': sum(1 for r in neg if r['gender'] == 'f')}

    return {
        'total': len(records),
        'positive': len(pos),
        'negative': len(neg),
        'positive_rate': round(len(pos) / len(records), 4),
        'age_groups': age_groups,
        'sex_asd_positive': sex_pos,
        'sex_asd_negative': sex_neg,
        'age_range': [min(r['age'] for r in records), max(r['age'] for r in records)],
        'countries': len(set(r['country'] for r in records)),
        'ethnicities': len(set(r['ethnicity'] for r in records)),
    }


if __name__ == '__main__':
    records, csv_path, json_path = export_dataset()
    stats = get_dataset_stats(records)
    print(json.dumps(stats, indent=2))
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
