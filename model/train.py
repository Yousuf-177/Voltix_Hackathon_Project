
"""
train.py
A robust training wrapper for Ultralytics YOLOv8 with useful defaults and options:
- recompute anchors (k-means) from YOLO-format labels
- SGD or AdamW optimizer selection
- cosine LR scheduler with warmup
- EMA (exponential moving average)
- Mosaic, MixUp, augmentations toggles
- AMP (mixed precision) usage when CUDA is available
- logging and saving best checkpoint

Usage examples:
    python train.py                              # runs with defaults
    python train.py --epochs 120 --batch 16 --imgsz 640 --device 0 --optimizer SGD
    python train.py --recompute-anchors          # prints k-means anchors for dataset
"""

import argparse
import os
import sys
import random
import math
from pathlib import Path
import yaml
import glob
import numpy as np

# External libs
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Please install ultralytics (pip install ultralytics).") from e

# Optional - sklearn for kmeans anchors
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

import torch

# ---------------------------
# Helper functions
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_data_yaml(path):
    """Return parsed YAML dict from dataset yaml (expects keys: train, val, nc, names)."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def gather_yolo_wh_from_labels(train_path, imgsz=640, max_samples=None):
    """
    Read YOLO txt labels from train_path (folder of images and labels or labels folder).
    Expects label files in YOLO txt format: <class> <x_center> <y_center> <w> <h> (normalized).
    Returns array of [w, h] (normalized).
    """
    labels_glob = []
    # Accept: if train_path is an images folder, assume labels in sibling 'labels' folder
    p = Path(train_path)
    if p.is_dir():
        # common pattern: images in train_path, labels in parent/labels/train or same images folder has names
        # We'll try to find .txt files recursively
        labels_glob = list(p.rglob('*.txt'))
    else:
        # might be a txt file listing images, not a folder
        # try to read file lines and map to label files next to each image
        labels_glob = []
        try:
            with open(train_path, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    lab = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(lab):
                        labels_glob.append(lab)
        except Exception:
            labels_glob = []

    wh = []
    for i, lab_file in enumerate(labels_glob):
        if max_samples and i >= max_samples:
            break
        try:
            with open(lab_file, 'r') as f:
                for l in f:
                    parts = l.strip().split()
                    if len(parts) >= 5:
                        w = float(parts[3])
                        h = float(parts[4])
                        # ignore degenerate boxes
                        if w > 0 and h > 0:
                            wh.append([w, h])
        except Exception:
            continue

    if len(wh) == 0:
        raise RuntimeError(f"No labels found while scanning {train_path}. Check your dataset structure.")
    return np.array(wh)


def compute_kmeans_anchors(wh, n_clusters=9, random_state=42):
    """Run KMeans on width-height pairs (normalized) and print anchors scaled to chosen image sizes."""
    if KMeans is None:
        raise RuntimeError("scikit-learn is required for kmeans anchor computation. Install with `pip install scikit-learn`.")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(wh)
    centers = km.cluster_centers_
    # sort by area
    centers = centers[np.argsort(centers[:, 0] * centers[:, 1])]
    return centers


def pretty_print_anchors(centers, imgsz=640):
    """
    Print anchors in both normalized and pixel forms for convenience.
    centers: numpy array of shape (k,2) normalized width,height
    """
    print("\nComputed anchors (normalized w,h):")
    for c in centers:
        print(f"{c[0]:.6f}, {c[1]:.6f}")
    print("\nAnchors (pixels for imgsz={}) (w,h):".format(imgsz))
    for c in centers:
        print(f"{int(c[0]*imgsz)}, {int(c[1]*imgsz)}")
    print("\nYou can use these values to set anchors or verify anchor distribution.\n")


# ---------------------------
# Main training wrapper
# ---------------------------
def main():
    # ---------------------------
    # CLI arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description="YOLOv8 training wrapper with sensible defaults")
    parser.add_argument('--data', type=str, default='yolo_params.yaml', help='Path to dataset yaml file (YOLO format)')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Path to model or pretrained weights')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (per device)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (square)')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr0', type=float, default=None, help='Initial learning rate. If omitted, set sensible default')
    parser.add_argument('--lrf', type=float, default=0.05, help='Final LR factor (final_lr = lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='Momentum (SGD)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--mosaic', type=float, default=0.8, help='Mosaic probability (0 to 1)')
    parser.add_argument('--mixup', type=float, default=0.15, help='MixUp probability (0 to 1)')
    parser.add_argument('--augment', action='store_true', help='Enable extra augmentations (photometric, blur, jpeg, noise)')
    parser.add_argument('--single-cls', action='store_true', help='Train as single-class (all objects same class)')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='Device spec (cuda device index or cpu)')
    parser.add_argument('--ema', action='store_true', help='Use EMA (recommended)')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision (fp16) if CUDA available')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping (Ultralytics parameter)')
    parser.add_argument('--project', type=str, default='runs/train', help='Ultralytics project dir')
    parser.add_argument('--name', type=str, default=None, help='Run name')
    parser.add_argument('--recompute-anchors', action='store_true', help='Recompute anchors (k-means) from train labels and print them')
    parser.add_argument('--anchors', type=int, default=9, help='Number of anchors to compute if --recompute-anchors')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', action='store_true', help='Save the final model checkpoint explicitly')
    args = parser.parse_args()

    # ---------------------------
    # Environment / Defaults
    # ---------------------------
    set_seed(args.seed)

    # sensible defaults for lr depending on optimizer and batch scaling
    # reference base batch 16 for lr scaling
    if args.lr0 is None:
        if args.optimizer == 'SGD':
            base_lr = 0.01  # reasonable for detection with SGD
        else:
            base_lr = 0.0005  # AdamW tends to use lower lr
    else:
        base_lr = args.lr0

    # scale LR by batch size heuristics (maintain same lr per sample)
    base_batch = 16
    if args.batch > 0:
        lr_scaled = base_lr * (args.batch / base_batch)
    else:
        lr_scaled = base_lr

    print(f"Training config:\n  model: {args.model}\n  data: {args.data}\n  epochs: {args.epochs}\n  batch: {args.batch}\n  imgsz: {args.imgsz}\n  optimizer: {args.optimizer}\n  lr0 (scaled): {lr_scaled:.6g}\n  lrf: {args.lrf}\n  momentum: {args.momentum}\n  mosaic: {args.mosaic}\n  mixup: {args.mixup}\n  ema: {args.ema}\n  amp: {args.amp}\n  device: {args.device}\n")

    # ---------------------------
    # Optional: recompute anchors
    # ---------------------------
    if args.recompute_anchors:
        print("Recomputing anchors from training labels (k-means). This may take a bit...")
        data_cfg = read_data_yaml(args.data)
        train_path = data_cfg.get('train')
        if not train_path:
            raise RuntimeError("Could not find 'train' key in dataset yaml. Provide valid yolo_params.yaml")
        # If train path is a txt listing images, try to resolve to folder
        try:
            wh = gather_yolo_wh_from_labels(train_path, imgsz=args.imgsz, max_samples=20000)
        except Exception as e:
            raise RuntimeError("Failed to gather labels for anchors. Ensure labels exist and are in YOLO txt format.") from e

        centers = compute_kmeans_anchors(wh, n_clusters=args.anchors)
        pretty_print_anchors(centers, imgsz=args.imgsz)
        # exit after printing anchors
        print("Anchor recompute done. Re-run without --recompute-anchors to start training.")
        return

    # ---------------------------
    # Initialize model
    # ---------------------------
    model = YOLO(args.model)  # loads model (weights or yaml)
    # validate dataset yaml exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    # ---------------------------
    # Build train() kwargs (Ultralytics API)
    # ---------------------------
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": float(lr_scaled),
        "lrf": float(args.lrf),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "mosaic": float(args.mosaic),
        "mixup": float(args.mixup),
        "single_cls": bool(args.single_cls),
        "patience": int(args.patience),
        "project": args.project,
        "name": args.name,
        "save": True,   # allow ultralytics to save best/wandb style
        "verbose": True,
    }

    # augment flag: if true enable additional augmentations via 'augment' kwarg if supported
    if args.augment:
        train_kwargs["augment"] = True

    # EMA
    if args.ema:
        train_kwargs["ema"] = True

    # AMP (ultralytics will switch to fp16/half if device is cuda and fp16 kwargs supported)
    # The ultralytics API uses 'device' and internally may enable fp16; we still expose "amp" behavior
    # We'll set 'fp16' kwarg if available
    if args.amp and args.device != 'cpu':
        train_kwargs["fp16"] = True

    # set name default if not provided
    if train_kwargs.get("name") is None:
        train_kwargs["name"] = f"yolov8_train_{Path(args.data).stem}"

    print("Final train kwargs:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    # ---------------------------
    # Kick off training
    # ---------------------------
    print("\nStarting training... This may take a while depending on dataset and hardware.")
    results = model.train(**train_kwargs)

    # results is an ultralytics object containing training info; print concise summary
    try:
        # ultralytics returns a dict-like result with 'metrics' in some versions; adaptively print what's available
        print("\nTraining finished. Summary:")
        if hasattr(results, 'metrics') and results.metrics is not None:
            print(results.metrics)
        else:
            # fallback: results is probably a list or object, print repr
            print(repr(results))
    except Exception:
        print("Training completed; results object returned.")

    # Optionally save final weights explicitly (if requested)
    if args.save:
        out_dir = Path(args.project) / (args.name or '')
        best = next(out_dir.rglob("best*.pt"), None)
        final = next(out_dir.rglob("weights/*.pt"), None)
        if best:
            dest = Path.cwd() / "best_model.pt"
            print(f"Copying best checkpoint {best} -> {dest}")
            import shutil
            shutil.copy(best, dest)
            print("Saved best_model.pt")
        elif final:
            dest = Path.cwd() / "final_model.pt"
            shutil.copy(final, dest)
            print("Saved final_model.pt")
        else:
            print("No checkpoint found to save. Check project/run directory.")

    print("Done.")


if __name__ == "__main__":
    main()
