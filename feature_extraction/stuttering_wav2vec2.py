"""
feature_extraction/stuttering_wav2vec2.py
==========================================
Extract wav2vec2-base embeddings (768-dim, mean-pooled) OR
MFCC statistics (80-dim) from SEP-28k / FluencyBank clips.

Two modes:
  --mode wav2vec2   Uses facebook/wav2vec2-base (HuggingFace).
                    Requires audio clips downloaded to --audio_dir.
  --mode mfcc       Uses librosa MFCC(n_mfcc=40) mean+std = 80-dim.
                    Also requires audio clips.
  --mode labels_only  No audio needed. Engineers stutter-event
                    count features directly from the label CSV.
                    Use this if clips are not yet downloaded.

Clip audio download (one-time):
  python SEP28k-Dataset/ml-stuttering-events-dataset-main/download_audio.py
  (Requires Podcast Index API key — see dataset README.)
  Or use the HuggingFace mirror:
  huggingface-cli download saeedzou/sep28k-fluencybank-stutter-dataset

Label CSV columns used:
  Show, EpId, ClipId -> group_id = Show
  Prolongation, Block, SoundRep, WordRep -> stutter events (label = any > 0)
  PoorAudioQuality, NoSpeech, Music -> quality filters

Audio file naming convention (SEP-28k):
  <AUDIO_DIR>/<Show>/<EpId>/<ClipId>.wav  (or .mp3)

Run:
  python -m feature_extraction.stuttering_wav2vec2 \\
      --labels_csv  "SEP28k-Dataset/.../SEP-28k_labels.csv" \\
      --fluency_csv "SEP28k-Dataset/.../fluencybank_labels.csv" \\
      --audio_dir   "ml/data/stuttering_audio" \\
      --out_dir     "ml/data/features" \\
      --mode        mfcc

Outputs:
  stuttering_features.npy     shape (N, D)
  stuttering_labels.npy       shape (N,)   1=stutter 0=fluent
  stuttering_groups.npy       shape (N,)   Show name as category int
  stuttering_file_ids.npy     shape (N,)   "Show/EpId/ClipId" strings
  stuttering_group_names.npy  shape (N,)   Show strings
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STUTTER_COLS  = ["Prolongation", "Block", "SoundRep", "WordRep"]
QUALITY_COLS  = ["PoorAudioQuality", "NoSpeech", "Music"]
CORRUPT_LOG   = "corrupt_stuttering.txt"
SEG_DURATION  = 3.0   # seconds (pad or trim to this)
SAMPLE_RATE   = 16000


def load_label_rows(labels_csv: str, fluency_csv: str | None) -> list[dict]:
    """Load and merge SEP-28k + FluencyBank label rows, filter poor quality."""
    rows = []
    for csvpath in ([labels_csv] + ([fluency_csv] if fluency_csv else [])):
        if not csvpath or not Path(csvpath).exists():
            log.warning("CSV not found: %s", csvpath)
            continue
        with open(csvpath, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # Quality filter
                if any(int(r.get(c, 0)) > 0 for c in QUALITY_COLS):
                    continue
                rows.append(r)
    log.info("Loaded %d usable label rows.", len(rows))
    return rows


def make_label(row: dict) -> int:
    return 1 if any(int(row.get(c, 0)) > 0 for c in STUTTER_COLS) else 0


def find_audio_file(audio_dir: str, show: str, ep: str, clip: str) -> str | None:
    """Try to locate the clip audio file under audio_dir."""
    base = Path(audio_dir)
    show_clean = show.strip().replace(" ", "_")
    ep_clean   = ep.strip().zfill(3)
    clip_clean = clip.strip()

    candidates = [
        base / show_clean / ep_clean / f"{clip_clean}.wav",
        base / show_clean / ep_clean / f"{clip_clean}.mp3",
        base / show_clean / f"{ep_clean}_{clip_clean}.wav",
        base / f"{show_clean}_{ep_clean}_{clip_clean}.wav",
        base / f"{show_clean}_{ep_clean}_{clip_clean}.mp3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def extract_mfcc(audio_path: str) -> np.ndarray:
    """Extract MFCC mean+std (n_mfcc=40) → 80-dim vector."""
    import librosa
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                              duration=SEG_DURATION)
        # Pad or trim to exactly SEG_DURATION * SAMPLE_RATE samples
        target = int(SEG_DURATION * SAMPLE_RATE)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    except Exception as e:
        log.debug("MFCC failed for %s: %s", audio_path, e)
        return None


def extract_wav2vec2(audio_path: str, processor, model, device) -> np.ndarray:
    """Extract wav2vec2-base mean-pooled embedding → 768-dim."""
    import torch
    import librosa
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                              duration=SEG_DURATION)
        target = int(SEG_DURATION * SAMPLE_RATE)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]

        inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        # Mean-pool over time dimension
        hidden = out.last_hidden_state.squeeze(0)  # (T, 768)
        return hidden.mean(dim=0).cpu().numpy()
    except Exception as e:
        log.debug("wav2vec2 failed for %s: %s", audio_path, e)
        return None


def extract_label_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    No-audio fallback: engineer features directly from the label CSV counts.
    Returns X (N, 14), y (N,), groups (N,), file_ids (N,)
    """
    log.info("Extracting label-based features (no audio required).")
    X_list, y_list, groups, file_ids = [], [], [], []

    all_shows = sorted(set(r["Show"].strip() for r in rows))
    show_map  = {s: i for i, s in enumerate(all_shows)}

    for r in rows:
        show = r["Show"].strip()
        ep   = r["EpId"].strip()
        clip = r["ClipId"].strip()

        prol = int(r.get("Prolongation", 0))
        blk  = int(r.get("Block", 0))
        srep = int(r.get("SoundRep", 0))
        wrep = int(r.get("WordRep", 0))
        intj = int(r.get("Interjection", 0))
        npau = int(r.get("NaturalPause", 0))
        unsure = int(r.get("Unsure", 0))
        ditund = int(r.get("DifficultToUnderstand", 0))

        total_stutter = prol + blk + srep + wrep
        n_types = sum([prol > 0, blk > 0, srep > 0, wrep > 0])

        feat = np.array([
            prol, blk, srep, wrep,
            total_stutter,
            float(n_types),
            float(prol > 0), float(blk > 0),
            float(srep > 0), float(wrep > 0),
            intj, npau, unsure, ditund,
        ], dtype=np.float32)

        X_list.append(feat)
        y_list.append(make_label(r))
        groups.append(show_map[show])
        file_ids.append(f"{show}/{ep}/{clip}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int8)
    g = np.array(groups, dtype=np.int32)
    f = np.array(file_ids, dtype=object)
    log.info("Label-based features shape: %s  Stutter: %d  Fluent: %d",
             X.shape, (y == 1).sum(), (y == 0).sum())
    return X, y, g, f


def extract_audio_features(rows: list[dict], audio_dir: str,
                            mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract MFCC or wav2vec2 features from audio clips."""
    import torch

    all_shows = sorted(set(r["Show"].strip() for r in rows))
    show_map  = {s: i for i, s in enumerate(all_shows)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Audio extraction mode: %s  device: %s", mode, device)

    processor = model_w2v = None
    if mode == "wav2vec2":
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        log.info("Loading facebook/wav2vec2-base (this may take a moment)…")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        model_w2v.eval()

    X_list, y_list, groups, file_ids = [], [], [], []
    corrupt = []

    for r in tqdm(rows, desc=f"Stuttering {mode}"):
        show = r["Show"].strip()
        ep   = r["EpId"].strip()
        clip = r["ClipId"].strip()
        fid  = f"{show}/{ep}/{clip}"

        audio_path = find_audio_file(audio_dir, show, ep, clip)
        if audio_path is None:
            corrupt.append(f"NOT_FOUND: {fid}")
            continue

        if mode == "mfcc":
            feat = extract_mfcc(audio_path)
        else:
            feat = extract_wav2vec2(audio_path, processor, model_w2v, device)

        if feat is None:
            corrupt.append(f"CORRUPT: {audio_path}")
            continue

        X_list.append(feat)
        y_list.append(make_label(r))
        groups.append(show_map[show])
        file_ids.append(fid)

    if corrupt:
        log.warning("Skipped %d files. See corrupt_stuttering.txt", len(corrupt))
        Path("corrupt_stuttering.txt").write_text("\n".join(corrupt))

    if not X_list:
        raise RuntimeError("No features extracted. Check audio_dir path or use --mode labels_only")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int8)
    g = np.array(groups, dtype=np.int32)
    f = np.array(file_ids, dtype=object)
    log.info("Audio features shape: %s  Stutter: %d  Fluent: %d",
             X.shape, (y == 1).sum(), (y == 0).sum())
    return X, y, g, f


def run(labels_csv: str, fluency_csv: str, audio_dir: str,
        out_dir: str, mode: str) -> None:

    rows = load_label_rows(labels_csv, fluency_csv)
    if not rows:
        raise RuntimeError("No label rows loaded. Check CSV paths.")

    if mode == "labels_only":
        X, y, g, f = extract_label_features(rows)
    else:
        X, y, g, f = extract_audio_features(rows, audio_dir, mode)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "stuttering_features.npy",    X)
    np.save(out / "stuttering_labels.npy",       y)
    np.save(out / "stuttering_groups.npy",       g)
    np.save(out / "stuttering_file_ids.npy",     f)

    log.info("Saved stuttering features to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract stuttering features from SEP-28k / FluencyBank")
    parser.add_argument("--labels_csv",  required=True)
    parser.add_argument("--fluency_csv", default=None)
    parser.add_argument("--audio_dir",   default="ml/data/stuttering_audio",
                        help="Root folder containing downloaded audio clips")
    parser.add_argument("--out_dir",     default="ml/data/features")
    parser.add_argument("--mode",        default="labels_only",
                        choices=["labels_only", "mfcc", "wav2vec2"],
                        help="labels_only: no audio needed; mfcc/wav2vec2: need downloaded clips")
    args = parser.parse_args()
    run(args.labels_csv, args.fluency_csv, args.audio_dir, args.out_dir, args.mode)
