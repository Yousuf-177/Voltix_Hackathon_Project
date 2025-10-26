"""
yolo_detect.py
Batch inference for images, folders, and video files using Ultralytics YOLOv8.
- Supports: single image, folder of images, video file.
- Device selection, FP16, warm-up, batch inference for folder processing.
- Saves YOLO-format .txt labels and optionally annotated images.
- Uses model.predict for efficient inference and robust parsing of results.
"""

import argparse
import os
import time
from pathlib import Path
import yaml

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# -------------------------
# Helpers
# -------------------------
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def is_image_file(p: Path):
    return p.suffix.lower() in IMG_EXTS

def annotate_image_cv2(img: np.ndarray, boxes_xyxy, scores, classes, names):
    """Draw boxes and labels on copy of image. xyxy are in pixel coordinates."""
    out = img.copy()
    # dynamic color palette for classes
    num_names = len(names) if isinstance(names, (list, dict)) else 10
    for xyxy, conf, cls in zip(boxes_xyxy, scores, classes):
        x1, y1, x2, y2 = map(int, xyxy)
        color = tuple(int(c) for c in ((hash(cls) & 0xFF), (hash(cls*31) & 0xFF), (hash(cls*97) & 0xFF)))
        label = f"{names[int(cls)]}:{conf:.2f}" if names else f"{int(cls)}:{conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # CHANGED: Increased fontScale from 0.45 to 0.7 for bigger text
        t_size = cv2.getTextSize(label, 0, fontScale=1.5, thickness=2)[0]
        cv2.rectangle(out, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 2, y1), color, -1)
        # CHANGED: Increased fontScale from 0.45 to 0.7 for bigger text
        cv2.putText(out, label, (x1, y1 - 4), 0, 1.5, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return out

def write_yolo_txt(txt_path: Path, boxes_xywhn, classes):
    """Write YOLO-format lines: cls x_center y_center width height (normalized)."""
    with open(txt_path, 'w') as f:
        for cls, wh in zip(classes, boxes_xywhn):
            f.write(f"{int(cls)} {wh[0]:.6f} {wh[1]:.6f} {wh[2]:.6f} {wh[3]:.6f}\n")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 batch predict (images/folder/video) — no camera support.")
    parser.add_argument('--source', type=str, required=True,
                        help='Path to single image, folder with images, or video file.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to weights (default: autodetect best.pt in runs/detect/train/*).')
    parser.add_argument('--data', type=str, default='yolo_params.yaml',
                        help='Dataset yaml (used for optional validation at end).')
    parser.add_argument('--out', type=str, default='predictions', help='Output folder for images/labels.')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: "cpu" or "cuda:0" etc.')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 (half precision) if using GPU.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for folder inference.')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size.')
    parser.add_argument('--warmup', type=int, default=3, help='Warm-up iterations on GPU.')
    # Modified to save annotated images by default (from previous request)
    parser.add_argument('--no-save-annotated', dest='save_annotated', action='store_false', default=True, 
                        help='DO NOT save annotated images to output/images (Saving is default).')
    parser.add_argument('--video-out', type=str, default=None,
                        help='If source is video and you want to save annotated video, provide output filename (e.g. out.mp4).')
    parser.add_argument('--select-run', action='store_true',
                        help='If model not provided, interactively select runs/detect/train/* folder to choose best.pt.')
    args = parser.parse_args()

    # MODIFIED: Go up one directory to the project root (CWD is 'assets', project is 'cwd')
    this_dir = Path(__file__).parent.parent.resolve()
    os.chdir(this_dir)

    source = Path(args.source)
    output_dir = Path(args.out)
    images_out_dir = output_dir / 'images'
    labels_out_dir = output_dir / 'labels'
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path: provided or auto-select best.pt
    model_path = args.model
    if model_path is None:
        # auto-select best.pt only if model not explicitly given
        detect_path = this_dir / "runs" / "detect"
        train_folders = [f for f in os.listdir(detect_path) if (detect_path / f).is_dir() and f.startswith("train")]
        if not train_folders:
            raise FileNotFoundError("No training folders found; provide --model manually.")
        idx = 0
        model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    print(f"[INFO] Using model: {model_path}")

    # Load model
    model = YOLO(str(model_path))  # ultralytics model wrapper
    try:
        model.model.to(args.device)
    except Exception:
        pass

    use_fp16 = args.fp16 and ('cpu' not in args.device)
    if use_fp16:
        try:
            model.model.half()
            print("[INFO] Converted model to FP16 for inference.")
        except Exception as e:
            print(f"[WARN] FP16 conversion failed: {e}")
            use_fp16 = False

    # Attempt to read names from model; fallback to data yaml
    names = getattr(model.model, 'names', None)
    if names is None:
        try:
            data_cfg = yaml.safe_load(open(args.data))
            nc = data_cfg.get('nc', None)
            if nc:
                names = {i: str(i) for i in range(nc)}
            else:
                names = {}
        except Exception:
            names = {}

    # Warm-up (GPU)
    if 'cpu' not in args.device and args.warmup > 0:
        print(f"[INFO] Warming up for {args.warmup} iterations (imgsz={args.imgsz}, batch=1)")
        dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        for _ in range(args.warmup):
            _ = model.predict(source=dummy, imgsz=args.imgsz, device=args.device,
                              conf=args.conf, iou=args.iou, half=use_fp16)

    # Decide source type: image file, folder, or video file
    if source.is_file():
        if is_image_file(source):
            source_type = 'image'
        else:
            source_type = 'video'
    elif source.is_dir():
        source_type = 'folder'
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    # PROCESS: IMAGE
    if source_type == 'image':
        img = cv2.imread(str(source))
        if img is None:
            raise RuntimeError(f"Failed to read image: {source}")
        results = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                device=args.device, half=use_fp16)
        res = results[0]
        # parse boxes robustly
        boxes_xyxy, boxes_xywhn, scores, classes = [], [], [], []
        if hasattr(res, 'boxes') and res.boxes is not None:
            xyxy_arr = getattr(res.boxes, 'xyxy', None)
            xywhn_arr = getattr(res.boxes, 'xywhn', None)
            conf_arr = getattr(res.boxes, 'conf', None)
            cls_arr = getattr(res.boxes, 'cls', None)
            if xyxy_arr is not None:
                xyxy_np = xyxy_arr.cpu().numpy() if hasattr(xyxy_arr, 'cpu') else np.array(xyxy_arr)
            else:
                xyxy_np = np.zeros((0,4))
            if xywhn_arr is not None:
                xywhn_np = xywhn_arr.cpu().numpy() if hasattr(xywhn_arr, 'cpu') else np.array(xywhn_arr)
            else:
                xywhn_np = None
            conf_np = conf_arr.cpu().numpy() if (conf_arr is not None and hasattr(conf_arr, 'cpu')) else (np.array(conf_arr) if conf_arr is not None else None)
            cls_np = cls_arr.cpu().numpy() if (cls_arr is not None and hasattr(cls_arr, 'cpu')) else (np.array(cls_arr) if cls_arr is not None else None)

            h, w = img.shape[:2]
            for i_box in range(xyxy_np.shape[0]):
                xy = xyxy_np[i_box].tolist()
                boxes_xyxy.append([float(x) for x in xy])
                if xywhn_np is not None:
                    boxes_xywhn.append([float(x) for x in xywhn_np[i_box].tolist()])
                else:
                    x1,y1,x2,y2 = xy
                    xc = (x1 + x2) / 2.0 / w
                    yc = (y1 + y2) / 2.0 / h
                    ww = (x2 - x1) / w
                    hh = (y2 - y1) / h
                    boxes_xywhn.append([xc, yc, ww, hh])
                scores.append(float(conf_np[i_box]) if conf_np is not None else 0.0)
                classes.append(int(cls_np[i_box]) if cls_np is not None else 0)

        # Save label text
        label_file = labels_out_dir / (source.stem + '.txt')
        write_yolo_txt(label_file, boxes_xywhn, classes)

        # Save annotated image
        if args.save_annotated:
            ann = annotate_image_cv2(img, boxes_xyxy, scores, classes, names)
            save_to = images_out_dir / source.name
            cv2.imwrite(str(save_to), ann)

        print(f"[INFO] Done. Labels -> {label_file}")
        if args.save_annotated:
            print(f"[INFO] Annotated image -> {save_to}")

    # PROCESS: FOLDER (batch inference)
    elif source_type == 'folder':
        all_images = sorted([p for p in source.rglob('*') if p.is_file() and is_image_file(p)])
        if not all_images:
            raise RuntimeError(f"No images found in folder: {source}")
        total = len(all_images)
        print(f"[INFO] Found {total} images. Running inference in batches of {args.batch} ...")
        t0 = time.time()
        processed = 0
        batch_size = max(1, args.batch)
        for i in range(0, total, batch_size):
            batch_files = all_images[i:i+batch_size]
            imgs = [cv2.imread(str(p)) for p in batch_files]
            # filter out failed reads and keep mapping
            valid_pairs = [(p,img) for p,img in zip(batch_files, imgs) if img is not None]
            if not valid_pairs:
                processed += len(batch_files)
                continue
            imgs_np = [img for _, img in valid_pairs]
            preds = model.predict(source=imgs_np, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                  device=args.device, half=use_fp16)
            for (p, _), res in zip(valid_pairs, preds):
                boxes_xyxy, boxes_xywhn, scores, classes = [], [], [], []
                if hasattr(res, 'boxes') and res.boxes is not None:
                    xyxy_arr = getattr(res.boxes, 'xyxy', None)
                    xywhn_arr = getattr(res.boxes, 'xywhn', None)
                    conf_arr = getattr(res.boxes, 'conf', None)
                    cls_arr = getattr(res.boxes, 'cls', None)
                    if xyxy_arr is not None:
                        xyxy_np = xyxy_arr.cpu().numpy() if hasattr(xyxy_arr, 'cpu') else np.array(xyxy_arr)
                    else:
                        xyxy_np = np.zeros((0,4))
                    if xywhn_arr is not None:
                        xywhn_np = xywhn_arr.cpu().numpy() if hasattr(xywhn_arr, 'cpu') else np.array(xywhn_arr)
                    else:
                        xywhn_np = None
                    conf_np = conf_arr.cpu().numpy() if (conf_arr is not None and hasattr(conf_arr, 'cpu')) else (np.array(conf_arr) if conf_arr is not None else None)
                    cls_np = cls_arr.cpu().numpy() if (cls_arr is not None and hasattr(cls_arr, 'cpu')) else (np.array(cls_arr) if cls_arr is not None else None)

                    img_shape = cv2.imread(str(p)).shape
                    h, w = img_shape[:2]
                    for ib in range(xyxy_np.shape[0]):
                        xy = xyxy_np[ib].tolist()
                        boxes_xyxy.append([float(x) for x in xy])
                        if xywhn_np is not None:
                            boxes_xywhn.append([float(x) for x in xywhn_np[ib].tolist()])
                        else:
                            x1,y1,x2,y2 = xy
                            xc = (x1 + x2) / 2.0 / w
                            yc = (y1 + y2) / 2.0 / h
                            ww = (x2 - x1) / w
                            hh = (y2 - y1) / h
                            boxes_xywhn.append([xc, yc, ww, hh])
                        scores.append(float(conf_np[ib]) if conf_np is not None else 0.0)
                        classes.append(int(cls_np[ib]) if cls_np is not None else 0)

                # Save outputs
                label_file = labels_out_dir / (p.stem + '.txt')
                write_yolo_txt(label_file, boxes_xywhn, classes)
                if args.save_annotated:
                    img = cv2.imread(str(p))
                    ann = annotate_image_cv2(img, boxes_xyxy, scores, classes, names)
                    cv2.imwrite(str(images_out_dir / p.name), ann)

                processed += 1
            # progress
            print(f"[INFO] Processed {min(processed, total)}/{total}", end='\r')
        dt = time.time() - t0
        print(f"\n[INFO] Finished. Processed {total} images in {dt:.2f}s ({total/dt:.2f} img/s).")
        print(f"[INFO] Labels -> {labels_out_dir}")
        if args.save_annotated:
            print(f"[INFO] Annotated images -> {images_out_dir}")

    # PROCESS: VIDEO
    elif source_type == 'video':
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0

        # Setup video writer if requested
        writer = None
        if args.video_out:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(args.video_out), fourcc, fps, (width, height))

        print("[INFO] Starting video inference...")
        t0 = time.time()
        batch_frames = []
        frame_paths = []  # placeholder (we'll name saved labels by frame index)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            batch_frames.append(frame)
            frame_paths.append(frame_idx)  # use frame index for label names

            if len(batch_frames) >= max(1, args.batch):
                preds = model.predict(source=batch_frames, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                      device=args.device, half=use_fp16)
                for fid, res in zip(frame_paths, preds):
                    boxes_xyxy, boxes_xywhn, scores, classes = [], [], [], []
                    if hasattr(res, 'boxes') and res.boxes is not None:
                        xyxy_arr = getattr(res.boxes, 'xyxy', None)
                        xywhn_arr = getattr(res.boxes, 'xywhn', None)
                        conf_arr = getattr(res.boxes, 'conf', None)
                        cls_arr = getattr(res.boxes, 'cls', None)
                        if xyxy_arr is not None:
                            xyxy_np = xyxy_arr.cpu().numpy() if hasattr(xyxy_arr, 'cpu') else np.array(xyxy_arr)
                        else:
                            xyxy_np = np.zeros((0,4))
                        if xywhn_arr is not None:
                            xywhn_np = xywhn_arr.cpu().numpy() if hasattr(xywhn_arr, 'cpu') else np.array(xywhn_arr)
                        else:
                            xywhn_np = None
                        conf_np = conf_arr.cpu().numpy() if (conf_arr is not None and hasattr(conf_arr, 'cpu')) else (np.array(conf_arr) if conf_arr is not None else None)
                        cls_np = cls_arr.cpu().numpy() if (cls_arr is not None and hasattr(cls_arr, 'cpu')) else (np.array(cls_arr) if cls_arr is not None else None)

                        h, w = frame.shape[:2]
                        for ib in range(xyxy_np.shape[0]):
                            xy = xyxy_np[ib].tolist()
                            boxes_xyxy.append([float(x) for x in xy])
                            if xywhn_np is not None:
                                boxes_xywhn.append([float(x) for x in xywhn_np[ib].tolist()])
                            else:
                                x1,y1,x2,y2 = xy
                                xc = (x1 + x2) / 2.0 / w
                                yc = (y1 + y2) / 2.0 / h
                                ww = (x2 - x1) / w
                                hh = (y2 - y1) / h
                                boxes_xywhn.append([xc, yc, ww, hh])
                            scores.append(float(conf_np[ib]) if conf_np is not None else 0.0)
                            classes.append(int(cls_np[ib]) if cls_np is not None else 0)

                    # Save label file for this frame
                    label_file = labels_out_dir / f"frame_{fid:06d}.txt"
                    write_yolo_txt(label_file, boxes_xywhn, classes)

                    # Annotate frame and maybe write to output video
                    if args.save_annotated or writer:
                        # note: frame order corresponds to batch_frames; we will use index position
                        idx_in_batch = preds.index(res) if res in preds else 0
                        frame_to_use = batch_frames[idx_in_batch]
                        annotated = annotate_image_cv2(frame_to_use, boxes_xyxy, scores, classes, names)
                        if args.save_annotated:
                            cv2.imwrite(str(images_out_dir / f"frame_{fid:06d}.jpg"), annotated)
                        if writer:
                            writer.write(annotated)

                # clear batch buffers
                batch_frames = []
                frame_paths = []

        dt = time.time() - t0
        total_frames = frame_idx
        print(f"[INFO] Video done. Processed {total_frames} frames in {dt:.2f}s ({total_frames/dt:.2f} fps).")
        if writer:
            writer.release()
        cap.release()

    # REMOVED: The section that called model.val() on the dataset YAML is removed
    # to ensure the script only performs inference and does not require dataset files.

    print("[INFO] All done.")

if __name__ == "__main__":
    main()