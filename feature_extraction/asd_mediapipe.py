"""
feature_extraction/asd_mediapipe.py
=====================================
ASD video pathway feature extractor.

Two sources are combined into a single feature matrix:

1. SSBD+ XML annotations (36 videos, behavioural category labels)
   - armflapping, spinning, headbanging
   - Video ID used as group ID
   - Feature: binary behaviour flags + temporal density (events/minute)
   NOTE: SSBD videos are YouTube-hosted. If you have the .mp4 files,
   pass --video_dir and MediaPipe Pose will be run instead.

2. AQ-10 tabular features from UCI dataset (merged by subject ID if available)

If --video_dir is supplied AND videos are present, MediaPipe Pose is run
to compute per-video kinematic features:
  - mean / std of joint velocities for 33 landmarks
  - movement amplitude (range of x/y across frames)
  - dominant repetition frequency via FFT on wrist Y-trajectory
  - fraction of frames with a detected person

Run:
  # XML-only mode (no videos needed):
  python -m feature_extraction.asd_mediapipe \\
      --xml_dir  "SSBD-Dataset/ssbdplus-dataset-main/actions" \\
      --aq10_csv "ml/data/asd_uci_combined.csv" \\
      --out_dir  "ml/data/features"

  # Full video mode (if .mp4 files downloaded):
  python -m feature_extraction.asd_mediapipe \\
      --xml_dir   "SSBD-Dataset/.../actions" \\
      --video_dir "ml/data/ssbd_videos" \\
      --aq10_csv  "ml/data/asd_uci_combined.csv" \\
      --fps       5 \\
      --out_dir   "ml/data/features"

Outputs:
  asd_features.npy     shape (N, D)
  asd_labels.npy       shape (N,)   1=ASD  0=non-ASD
  asd_groups.npy       shape (N,)   video ID int
  asd_file_ids.npy     shape (N,)   video ID strings
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BEHAVIOUR_FEATURES = [
    "armflapping", "spinning", "headbanging",
    "body_rocking", "jumping", "hand_movements",
    "facial_grimace", "head_shaking",
]

CORRUPT_LOG = "corrupt_asd_videos.txt"


# ── XML annotation parsing ─────────────────────────────────────────────────

def parse_xml_annotations(xml_dir: str) -> list[dict]:
    """
    Parse SSBD+ XML files. Each XML = one video.
    Returns list of {video_id, url, behaviours: [(category, start_s, end_s)], ...}
    All SSBD videos are ASD-positive (label=1).
    Negatives are generated synthetically from AQ-10 tabular data.
    """
    records = []
    for xf in sorted(Path(xml_dir).glob("*.xml")):
        try:
            root = ET.parse(xf).getroot()
        except Exception as e:
            log.warning("Failed to parse %s: %s", xf, e)
            continue

        vid_id = root.get("id", xf.stem)
        url    = root.findtext("url", "")
        behav_nodes = root.findall(".//behaviour")

        behaviours = []
        for b in behav_nodes:
            cat = b.findtext("category", "").strip().lower()
            time_str = b.findtext("time", "")
            # Time format is "start:end" in seconds
            try:
                start_s, end_s = [float(t) for t in time_str.split(":")]
            except Exception:
                start_s, end_s = 0.0, 0.0
            behaviours.append({"category": cat, "start_s": start_s, "end_s": end_s})

        records.append({
            "video_id": vid_id,
            "url":      url,
            "behaviours": behaviours,
            "label":    1,  # All SSBD videos are ASD positive
        })

    log.info("Parsed %d SSBD XML annotation files.", len(records))
    return records


def xml_to_feature_vector(record: dict) -> np.ndarray:
    """
    Convert SSBD XML annotation to a behaviour feature vector.
    Features (len = len(BEHAVIOUR_FEATURES) * 2 + 4):
      - binary presence flags for each behaviour category
      - event count per category
      - total events, n_behaviour_types, has_repetitive, has_social
    """
    cats = [b["category"] for b in record["behaviours"]]
    durations = [max(0.0, b["end_s"] - b["start_s"]) for b in record["behaviours"]]
    total_dur = sum(durations) if durations else 0.0

    flags  = np.zeros(len(BEHAVIOUR_FEATURES), dtype=np.float32)
    counts = np.zeros(len(BEHAVIOUR_FEATURES), dtype=np.float32)

    for cat in cats:
        for i, feat_name in enumerate(BEHAVIOUR_FEATURES):
            if feat_name in cat or cat in feat_name:
                flags[i]  = 1.0
                counts[i] += 1.0

    total_events   = float(len(record["behaviours"]))
    n_types        = float(flags.sum())
    has_repetitive = float(any(c in ["armflapping", "spinning", "headbanging", "body_rocking"]
                                for c in cats))
    has_social     = float(any(c in ["gaze_avoidance", "social_isolation", "echolalia"]
                                for c in cats))

    return np.concatenate([flags, counts,
                           [total_events, n_types, has_repetitive, has_social,
                            total_dur]])  # 8+8+5 = 21 dims


# ── MediaPipe Pose (video mode) ────────────────────────────────────────────

def run_mediapipe_pose(video_path: str, target_fps: int = 5) -> np.ndarray | None:
    """
    Run MediaPipe Pose on a video file at target_fps.
    Returns kinematic feature vector (shape: 139) or None if video unreadable.
    Features: mean/std joint velocities (66), movement amplitude (66),
              dominant FFT frequency (1), detection rate (1), n_landmarks_mean (1),
              wrist velocity mean/std (2), hip sway mean/std (2).
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        log.error("cv2 or mediapipe not installed. Install with: pip install opencv-python mediapipe")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open video: %s", video_path)
        return None

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(round(vid_fps / target_fps)))

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(static_image_mode=False,
                           model_complexity=1,
                           min_detection_confidence=0.4,
                           min_tracking_confidence=0.4)

    # Collect 33 (x,y) landmarks per frame
    frames_xy: list[np.ndarray] = []
    detected_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            detected_frames += 1
            kp = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                          dtype=np.float32)  # (33, 2)
            frames_xy.append(kp.flatten())  # 66-dim
        else:
            frames_xy.append(np.full(66, np.nan, dtype=np.float32))

    cap.release()
    pose.close()

    if len(frames_xy) < 2:
        return None

    kp_seq = np.vstack(frames_xy)  # (T, 66)
    total_frames = len(frames_xy)
    detection_rate = detected_frames / max(1, total_frames)

    # Fill NaN by forward-fill then zero
    for i in range(kp_seq.shape[1]):
        col = kp_seq[:, i]
        mask = np.isnan(col)
        if mask.all():
            kp_seq[:, i] = 0.0
        else:
            idx = np.where(~mask)[0]
            kp_seq[:, i] = np.interp(np.arange(len(col)), idx, col[idx])

    # Velocities
    velocities = np.diff(kp_seq, axis=0)  # (T-1, 66)
    vel_mean = velocities.mean(axis=0)    # 66
    vel_std  = velocities.std(axis=0)     # 66

    # Amplitude (range)
    amplitude = kp_seq.max(axis=0) - kp_seq.min(axis=0)  # 66

    # Dominant repetition frequency from right-wrist Y (landmark 16, y = col 33)
    wrist_y = kp_seq[:, 33]
    wrist_y_centered = wrist_y - wrist_y.mean()
    fft_mag = np.abs(np.fft.rfft(wrist_y_centered))
    dom_freq = float(np.argmax(fft_mag[1:]) + 1) / max(1, total_frames) * target_fps

    # Scalar features
    wrist_vel_col = np.abs(velocities[:, 32:34])  # wrist x,y velocity
    wrist_vel_mean = wrist_vel_col.mean()
    wrist_vel_std  = wrist_vel_col.std()

    hip_col = np.abs(velocities[:, 46:50])  # hip landmarks
    hip_mean = hip_col.mean()
    hip_std  = hip_col.std()

    n_lm_mean = float(np.sum(~np.isnan(kp_seq).any(axis=1))) / max(1, total_frames)

    feat = np.concatenate([
        vel_mean, vel_std, amplitude,
        [dom_freq, detection_rate, n_lm_mean,
         wrist_vel_mean, wrist_vel_std,
         hip_mean, hip_std]
    ])  # 66+66+66+7 = 205 dims

    return feat.astype(np.float32)


# ── AQ-10 tabular merge ────────────────────────────────────────────────────

def load_aq10_negatives(aq10_csv: str | None) -> list[dict]:
    """Load AQ-10 CSV to generate non-ASD feature vectors as class-0 samples."""
    if not aq10_csv or not Path(aq10_csv).exists():
        return []
    import csv
    records = []
    with open(aq10_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_str = str(row.get("Class_ASD", row.get("label", "0"))).strip().upper()
            label = 1 if label_str in ("1", "YES") else 0
            records.append({"row": row, "label": label})
    log.info("Loaded %d AQ-10 records from %s.", len(records), aq10_csv)
    return records


def aq10_to_feature_vector(row: dict) -> np.ndarray:
    """Convert an AQ-10 CSV row to a 21-dim behaviour feature vector."""
    # AQ-10 questions map loosely onto SSBD behaviour flags
    a = {f"A{i}_Score": float(row.get(f"A{i}_Score", 0)) for i in range(1, 11)}
    aq_sum = sum(a.values())

    flags = np.array([
        a.get("A1_Score", 0),  # armflapping proxy
        a.get("A8_Score", 0),  # spinning proxy
        a.get("A7_Score", 0),  # headbanging proxy
        a.get("A9_Score", 0),  # body_rocking proxy
        0.0,                   # jumping
        a.get("A2_Score", 0),  # hand_movements proxy
        a.get("A6_Score", 0),  # facial_grimace proxy
        a.get("A3_Score", 0),  # head_shaking proxy
    ], dtype=np.float32)

    counts = flags.copy()
    total_events   = aq_sum
    n_types        = float((flags > 0).sum())
    has_repetitive = float(aq_sum >= 5)
    has_social     = float(a.get("A10_Score", 0) + a.get("A5_Score", 0) >= 1)
    total_dur      = aq_sum * 2.0  # proxy: 2s per positive item

    return np.concatenate([flags, counts,
                           [total_events, n_types, has_repetitive,
                            has_social, total_dur]])  # 21 dims


# ── Main extraction ────────────────────────────────────────────────────────

def run(xml_dir: str, aq10_csv: str, out_dir: str,
        video_dir: str | None = None, fps: int = 5) -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_list, y_list, groups, file_ids = [], [], [], []

    # ── SSBD+ XML annotations (ASD positive) ──────────────────────────────
    xml_records = parse_xml_annotations(xml_dir)
    corrupt = []

    for rec in tqdm(xml_records, desc="ASD SSBD XML"):
        vid_id = rec["video_id"]

        # Try video if available
        feat = None
        if video_dir:
            vid_dir = Path(video_dir)
            for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                vpath = vid_dir / f"{vid_id}{ext}"
                if vpath.exists():
                    feat = run_mediapipe_pose(str(vpath), target_fps=fps)
                    if feat is None:
                        corrupt.append(str(vpath))
                    break

        # Fallback: XML behaviour features
        if feat is None:
            feat = xml_to_feature_vector(rec)

        # Numerical group ID from video ID (e.g. "action_3" → 3)
        try:
            gid = int(vid_id.split("_")[-1])
        except Exception:
            gid = abs(hash(vid_id)) % 100000

        X_list.append(feat)
        y_list.append(rec["label"])
        groups.append(gid)
        file_ids.append(vid_id)

    # ── AQ-10 tabular samples (both positive and negative) ─────────────────
    aq10_records = load_aq10_negatives(aq10_csv)
    for i, rec in enumerate(tqdm(aq10_records, desc="ASD AQ-10 tabular")):
        feat  = aq10_to_feature_vector(rec["row"])
        label = rec["label"]
        gid   = 10000 + i   # unique group per AQ-10 record (individual questionnaire)

        X_list.append(feat)
        y_list.append(label)
        groups.append(gid)
        file_ids.append(f"aq10_{i}")

    if corrupt:
        (out / CORRUPT_LOG).write_text("\n".join(corrupt))
        log.warning("Skipped %d corrupt video files.", len(corrupt))

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int8)
    g = np.array(groups, dtype=np.int32)
    f = np.array(file_ids, dtype=object)

    np.save(out / "asd_features.npy",  X)
    np.save(out / "asd_labels.npy",    y)
    np.save(out / "asd_groups.npy",    g)
    np.save(out / "asd_file_ids.npy",  f)

    log.info("ASD features saved: shape=%s  Positive=%d  Negative=%d",
             X.shape, (y == 1).sum(), (y == 0).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ASD features from SSBD+ XML + AQ-10 tabular")
    parser.add_argument("--xml_dir",   required=True,
                        help="Path to folder containing action_*.xml files")
    parser.add_argument("--aq10_csv",  default="ml/data/asd_uci_combined.csv",
                        help="Path to AQ-10 tabular CSV (adds non-ASD negatives)")
    parser.add_argument("--video_dir", default=None,
                        help="Optional: path to downloaded SSBD .mp4 video files")
    parser.add_argument("--fps",       type=int, default=5,
                        help="Target FPS for MediaPipe Pose (only used if --video_dir set)")
    parser.add_argument("--out_dir",   default="ml/data/features")
    args = parser.parse_args()
    run(args.xml_dir, args.aq10_csv, args.out_dir, args.video_dir, args.fps)
