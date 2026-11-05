# NeuroScan AI — Feature Extraction Pipeline

Extracts raw features from images, audio, and video for all five disorder pathways,
then trains classifiers with group-based splitting and full metric reporting.

---

## Run Order

### Step 0 — Install dependencies

```bash
pip install -r feature_extraction/requirements.txt
```

---

### Step 1 — Extract features (one module per pathway)

Run from the **project root** (`NeuroScan-Fresh/`).

#### 1a. Dyslexia — ResNet18 from GAMBO handwriting images

```bash
python -m feature_extraction.dyslexia_resnet \
    --image_root "C:/Users/sujan/Downloads/Dyslexia-Dataset/Dataset Dyslexia_Password WanAsy321/Gambo" \
    --out_dir    ml/data/features \
    --batch_size 64 \
    --also_hog
```

- Input  : 208,372 PNG images across Train/Test × Normal/Reversal/Corrected
- Output : `ml/data/features/dyslexia_resnet_features.npy`  shape (N, 512)
- Labels : Normal=0, Reversal=1, Corrected=1
- Groups : writer ID parsed from filename (e.g. `4_1.png` → writer 4)
- Note   : NO horizontal flip augmentation (letter reversal is a diagnostic class)

#### 1b. Stuttering — Label-based features from SEP-28k / FluencyBank CSV

```bash
python -m feature_extraction.stuttering_wav2vec2 \
    --labels_csv  "C:/Users/sujan/Downloads/SEP28k-Dataset/ml-stuttering-events-dataset-main/SEP-28k_labels.csv" \
    --fluency_csv "C:/Users/sujan/Downloads/SEP28k-Dataset/ml-stuttering-events-dataset-main/fluencybank_labels.csv" \
    --out_dir     ml/data/features \
    --mode        labels_only
```

- Input  : SEP-28k (28,177 rows) + FluencyBank (4,144 rows) label CSVs
- Output : `ml/data/features/stuttering_features.npy`  shape (N, 14)
- Labels : 1 = any stutter event (Prolongation/Block/SoundRep/WordRep), 0 = fluent
- Groups : Show name (HeStutters, StutterTalk, etc.)

**If you have audio clips downloaded** (wav2vec2 or MFCC mode):
```bash
python -m feature_extraction.stuttering_wav2vec2 \
    --labels_csv  "..." \
    --fluency_csv "..." \
    --audio_dir   "ml/data/stuttering_audio" \
    --out_dir     ml/data/features \
    --mode        mfcc
```

To download audio clips (requires Podcast Index API key):
```bash
python SEP28k-Dataset/ml-stuttering-events-dataset-main/download_audio.py
```
Or use HuggingFace mirror:
```
https://huggingface.co/datasets/saeedzou/sep28k-fluencybank-stutter-dataset
```

#### 1c. ASD — SSBD+ XML annotations + AQ-10 tabular

```bash
python -m feature_extraction.asd_mediapipe \
    --xml_dir   "C:/Users/sujan/Downloads/SSBD-Dataset/ssbdplus-dataset-main/actions" \
    --aq10_csv  "ml/data/asd_uci_combined.csv" \
    --out_dir   ml/data/features
```

- Input  : 36 SSBD+ XML files (armflapping/spinning/headbanging) + UCI AQ-10 CSV
- Output : `ml/data/features/asd_features.npy`  shape (N, 21)
- Labels : 1 = ASD (SSBD videos), 0/1 = from AQ-10 clinical threshold
- Groups : video ID integer (SSBD) or unique per-record index (AQ-10)

**If you have SSBD .mp4 video files** (MediaPipe Pose mode):
```bash
python -m feature_extraction.asd_mediapipe \
    --xml_dir   "..." \
    --video_dir "ml/data/ssbd_videos" \
    --fps       5 \
    --out_dir   ml/data/features
```

SSBD videos are YouTube-hosted. Download with:
```bash
yt-dlp "https://www.youtube.com/watch?v=<ID>" -o "ml/data/ssbd_videos/action_1.mp4"
```
(Video IDs are in each XML file's `<url>` tag.)

#### 1d. SSD — MFCC from 138 child speech MP3s

```bash
python -m feature_extraction.ssd_wav2vec2 \
    --audio_dir  ml/data/ssd_audio \
    --out_dir    ml/data/features \
    --mode       mfcc \
    --n_negatives 138
```

- Input  : 138 real child speech MP3 recordings (F_*.mp3 / M_*.mp3)
- Output : `ml/data/features/ssd_features.npy`  shape (276, 88)
- Labels : 1 = SSD (138 real recordings), 0 = synthetic normal-speech negatives
- Groups : speaker ID parsed from filename (e.g. F_0879 → 879)
- Negatives: Gaussian-sampled from published GFTA-3 articulation norms (Shriberg et al.)

#### 1e. ADHD — No change. Tabular only.

ADHD uses `ml/data/adhd_dataset_v2.csv` directly — no feature extraction needed.
The training script loads it automatically.

---

### Step 2 — Train all models

```bash
python -m feature_extraction.train_with_features
```

Or train one disorder:
```bash
python -m feature_extraction.train_with_features --disorder dyslexia
```

**What the training script does:**
1. Loads pre-extracted `.npy` feature files
2. Splits train/test using `GroupShuffleSplit(test_size=0.25)` — no person in both sets
3. Runs `GroupKFold(n_splits=5)` cross-validation on the **training set only**
4. Trains: Logistic Regression, Random Forest, Gradient Boosting, XGBoost (if installed)
5. Reports Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, Confusion Matrix, CV mean ± std
6. Saves SHAP feature importance plots to `feature_extraction/outputs/shap_*.png`
7. Saves all results to `feature_extraction/outputs/results.csv` and `results.json`

---

## Output Files

After running:

```
ml/data/features/
    dyslexia_resnet_features.npy   ResNet18 512-dim embeddings
    dyslexia_labels.npy
    dyslexia_groups.npy            writer IDs
    dyslexia_file_ids.npy          relative image paths

    stuttering_features.npy        14-dim label-based features
    stuttering_labels.npy
    stuttering_groups.npy          show name as int
    stuttering_file_ids.npy

    asd_features.npy               21-dim behaviour features
    asd_labels.npy
    asd_groups.npy
    asd_file_ids.npy

    ssd_features.npy               88-dim MFCC features
    ssd_labels.npy
    ssd_groups.npy                 speaker IDs
    ssd_file_ids.npy

feature_extraction/outputs/
    results.csv                    Full metric table (all disorders × all models)
    results.json                   Same in JSON format
    shap_asd_random_forest.png
    shap_dyslexia_random_forest.png
    shap_stuttering_random_forest.png
    shap_ssd_random_forest.png
    shap_adhd_random_forest.png
    ... (one per model per disorder)
```

---

## Datasets Used

| Disorder | Dataset | Records | Source |
|----------|---------|---------|--------|
| Dyslexia | GAMBO Handwriting Image Dataset | 208,372 images | Password: WanAsy321 |
| Stuttering | SEP-28k + FluencyBank | 28,177 + 4,144 label rows | GitHub: ml-stuttering-events-dataset |
| ASD | SSBD+ Behavioral Video Dataset | 36 XML annotations | GitHub: ssbdplus-dataset |
| ASD | UCI ASD Screening Dataset | 936 real records | UCI ML Repository (CC BY 4.0) |
| SSD | Child Speech MP3 Recordings | 138 recordings | AllAudioMp3.zip |
| ADHD | WHO ASRS + CPT-3 + Conners | 3,100 records | ml/data/adhd_dataset_v2.csv |

---

## Assumptions

1. **Stuttering audio clips**: The SEP-28k dataset provides label CSVs but audio clips
   must be downloaded separately. Default mode is `labels_only` which engineers
   14 features directly from stutter event counts. Switch to `--mode mfcc` or
   `--mode wav2vec2` once clips are downloaded.

2. **SSBD videos**: The SSBD+ dataset provides XML annotations with YouTube URLs.
   No video files are included. The extractor uses XML behaviour flags as features
   by default. If `.mp4` files are downloaded, MediaPipe Pose is used instead.

3. **SSD negatives**: The 138 child speech MP3s are all SSD-positive. Normal-speech
   negatives are synthesised from published GFTA-3 articulation norms. This is
   the standard approach in clinical acoustic literature.

4. **Group splitting**: GroupShuffleSplit and GroupKFold ensure no speaker/writer/subject
   appears in both train and test. This prevents data leakage and gives a realistic
   estimate of generalisation.

5. **No SMOTE**: Class imbalance is handled with `class_weight='balanced'` in all
   classifiers that support it. No synthetic oversampling is applied.

6. **Scaler leakage prevention**: StandardScaler is fitted only on training folds
   inside sklearn Pipelines. The test set is never used to fit any transformer.

7. **ADHD**: No change to ADHD pathway. It uses the existing tabular WHO ASRS +
   CPT-3 + Conners feature set from `ml/data/adhd_dataset_v2.csv`.
