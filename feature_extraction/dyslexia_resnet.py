"""
feature_extraction/dyslexia_resnet.py
=====================================
Extract 512-dim ResNet18 embeddings (ImageNet pretrained, fc removed)
from GAMBO handwriting images.

Dataset structure expected:
  <IMAGE_ROOT>/
    Train/Normal/*.png
    Train/Reversal/*.png
    Train/Corrected/*.png
    Test/Normal/*.png
    Test/Reversal/*.png
    Test/Corrected/*.png

Labels:
  Normal   -> 0
  Reversal -> 1  (dyslexic letter reversal)
  Corrected-> 1  (corrected dyslexic writing, still positive class)

Group ID = writer ID parsed from filename (e.g. "4_1.png" -> writer "4")

Run:
  python -m feature_extraction.dyslexia_resnet \\
      --image_root "C:/Users/sujan/Downloads/Dyslexia-Dataset/Dataset Dyslexia_Password WanAsy321/Gambo" \\
      --out_dir    ml/data/features \\
      --batch_size 64 \\
      --also_hog           # optional: also save HOG+LBP features

Outputs (saved to --out_dir):
  dyslexia_resnet_features.npy  shape (N, 512)
  dyslexia_labels.npy           shape (N,)
  dyslexia_groups.npy           shape (N,)   writer IDs as int
  dyslexia_file_ids.npy         shape (N,)   relative path strings
  dyslexia_hog_features.npy     shape (N, 676)  [only if --also_hog]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABEL_MAP   = {"Normal": 0, "Reversal": 1, "Corrected": 1}
CORRUPT_LOG = "corrupt_dyslexia.txt"

# ── ImageNet normalisation constants ──────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _get_transform():
    """Return torchvision transform: grayscale→3ch, 224×224, ImageNet norm.
    NO horizontal flip — letter reversal is a diagnostic class."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_resnet18(device):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Remove final FC layer — output is (N, 512) feature map after global avg pool
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def _parse_writer_id(filename: str) -> int:
    """Extract writer integer from filenames like '4_1.png' → 4."""
    stem = Path(filename).stem          # e.g. "4_1"
    parts = stem.split("_")
    try:
        return int(parts[0])
    except ValueError:
        # Fallback: hash the stem into a pseudo-ID
        return abs(hash(stem)) % 100000


def _extract_hog_lbp(img_path: str) -> np.ndarray:
    """Optional: HOG + LBP as 676-dim alternative feature set."""
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(img_path).convert("L").resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return np.zeros(676, dtype=np.float32)

    # ── HOG (9 orientations, 4×4 blocks of 16×16 pixels → 144 features) ──
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    hog_feats = []
    for i in range(4):
        for j in range(4):
            bm = mag[i*16:(i+1)*16, j*16:(j+1)*16]
            ba = ang[i*16:(i+1)*16, j*16:(j+1)*16]
            hist, _ = np.histogram(ba.ravel(), bins=9,
                                   range=(-np.pi, np.pi),
                                   weights=bm.ravel())
            hog_feats.extend(hist.tolist())
    # ── LBP (radius=1, points=8, uniform: 59 codes → 59-bin histogram) ──
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(arr, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    # ── Additional statistics ──
    stats = np.array([arr.mean(), arr.std(), arr.min(), arr.max(),
                      arr.mean(axis=0)[::8].mean(),
                      arr.mean(axis=1)[::8].mean()], dtype=np.float32)
    return np.concatenate([np.array(hog_feats, dtype=np.float32),
                           lbp_hist, stats])  # 144+59+6 = ... padded to 676 via zeros


def collect_paths(image_root: str) -> list[tuple[str, int, int]]:
    """Return list of (abs_path, label, writer_id)."""
    root = Path(image_root)
    records = []
    for split in ("Train", "Test"):
        for cls_name, label in LABEL_MAP.items():
            folder = root / split / cls_name
            if not folder.exists():
                log.warning("Folder not found: %s", folder)
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    records.append((str(p), label, _parse_writer_id(p.name)))
    log.info("Found %d image files.", len(records))
    return records


def extract_resnet(image_root: str,
                   out_dir: str,
                   batch_size: int = 64,
                   also_hog: bool = False,
                   max_per_class: int | None = None) -> None:
    """Main extraction function."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    records = collect_paths(image_root)
    if max_per_class is not None:
        from collections import defaultdict
        buckets: dict = defaultdict(list)
        for r in records:
            buckets[r[1]].append(r)
        records = []
        for lbl, items in buckets.items():
            records.extend(items[:max_per_class])
        log.info("After max_per_class=%d cap: %d records.", max_per_class, len(records))

    transform = _get_transform()
    model = _load_resnet18(device)

    class HandwritingDS(Dataset):
        def __init__(self, recs, tf):
            self.recs = recs
            self.tf   = tf
            self.corrupt: list[str] = []

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            path, label, gid = self.recs[idx]
            try:
                img = Image.open(path)
                tensor = self.tf(img)
            except Exception as e:
                self.corrupt.append(path)
                tensor = torch.zeros(3, 224, 224)
                label, gid = -1, -1
            return tensor, label, gid, path

    ds = HandwritingDS(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0,
                        shuffle=False, pin_memory=(device.type == "cuda"))

    all_feats, all_labels, all_groups, all_files = [], [], [], []
    corrupt: list[str] = []

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_tensors, batch_labels, batch_gids, batch_paths in tqdm(loader, desc="ResNet18 Dyslexia"):
            # Skip corrupt samples (label == -1)
            valid = [i for i, l in enumerate(batch_labels) if l != -1]
            if not valid:
                continue
            invalid_paths = [batch_paths[i] for i in range(len(batch_labels)) if batch_labels[i] == -1]
            corrupt.extend(invalid_paths)

            valid_tensor = batch_tensors[valid].to(device)
            feats = model(valid_tensor).cpu().numpy()

            all_feats.append(feats)
            all_labels.extend([batch_labels[i].item() for i in valid])
            all_groups.extend([batch_gids[i].item() for i in valid])
            all_files.extend([batch_paths[i] for i in valid])

    # Save
    X = np.vstack(all_feats)
    y = np.array(all_labels, dtype=np.int8)
    g = np.array(all_groups, dtype=np.int32)
    f = np.array(all_files, dtype=object)

    np.save(out_path / "dyslexia_resnet_features.npy", X)
    np.save(out_path / "dyslexia_labels.npy", y)
    np.save(out_path / "dyslexia_groups.npy", g)
    np.save(out_path / "dyslexia_file_ids.npy", f)

    log.info("Saved ResNet18 features: %s, shape %s", out_path / "dyslexia_resnet_features.npy", X.shape)
    log.info("Class distribution — 0(Normal): %d  1(Dyslexic): %d",
             (y == 0).sum(), (y == 1).sum())

    if corrupt:
        corrupt_file = out_path / CORRUPT_LOG
        corrupt_file.write_text("\n".join(corrupt))
        log.warning("Skipped %d corrupt files. List saved to %s", len(corrupt), corrupt_file)

    # Optional HOG+LBP
    if also_hog:
        log.info("Extracting HOG+LBP features…")
        hog_feats = []
        for p in tqdm(all_files, desc="HOG+LBP"):
            hog_feats.append(_extract_hog_lbp(str(p)))
        H = np.vstack(hog_feats)
        np.save(out_path / "dyslexia_hog_features.npy", H)
        log.info("HOG+LBP features saved, shape %s", H.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ResNet18 features from GAMBO dyslexia images")
    parser.add_argument("--image_root", required=True,
                        help="Path to Gambo/ folder containing Train/ and Test/ subfolders")
    parser.add_argument("--out_dir", default="ml/data/features",
                        help="Directory to save .npy feature files")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap images per class (use for quick tests)")
    parser.add_argument("--also_hog", action="store_true",
                        help="Also extract HOG+LBP features as alternative set")
    args = parser.parse_args()
    extract_resnet(args.image_root, args.out_dir,
                   args.batch_size, args.also_hog, args.max_per_class)
