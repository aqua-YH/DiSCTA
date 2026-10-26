from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Sequence, Tuple
import numpy as np
from pathlib import Path


def load_crossvlad_samples(images_root: str, labels_root: str) -> List[Dict]:
    image_paths = sorted(glob.glob(os.path.join(images_root, "*.jpg")))
    samples: List[Dict] = []
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_root, stem + ".json")
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["image_size"] = (int(meta["image_width"]), int(meta["image_height"]))
        samples.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "image_id": stem,
                "meta": meta,
            }
        )
    return samples


def collect_category_vocab(samples: Sequence[Dict]) -> List[str]:
    vocab = set()
    for sample in samples:
        meta = sample["meta"]
        vocab.add(str(meta.get("source_object_category", "")).strip().lower())
        vocab.add(str(meta.get("target_object_category", "")).strip().lower())
    vocab = sorted((x for x in vocab if x))
    return vocab


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def scale_xyxy(
    box_xyxy: Sequence[float], src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> List[float]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return [float(v) for v in box_xyxy]
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    x1, y1, x2, y2 = box_xyxy
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def map_box_to_canvas(
    box_xywh: Sequence[float], src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> List[float]:
    return scale_xyxy(xywh_to_xyxy(box_xywh), src_size, dst_size)


def clamp_xyxy(box_xyxy: Sequence[float], size: int) -> List[float]:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(x1), float(size)))
    y1 = max(0.0, min(float(y1), float(size)))
    x2 = max(0.0, min(float(x2), float(size)))
    y2 = max(0.0, min(float(y2), float(size)))
    if x2 <= x1:
        x2 = min(float(size), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(size), y1 + 1.0)
    return [x1, y1, x2, y2]


def box_to_token_indices(
    box_xyxy: Sequence[float], canvas_size: int = 768, grid: int = 24
) -> List[int]:
    patch_size = canvas_size // grid
    x1, y1, x2, y2 = clamp_xyxy(box_xyxy, canvas_size)
    gx1 = int(np.floor(x1 / patch_size))
    gy1 = int(np.floor(y1 / patch_size))
    gx2 = int(np.floor(max(x2 - 1.0, x1) / patch_size))
    gy2 = int(np.floor(max(y2 - 1.0, y1) / patch_size))
    gx1 = max(0, min(gx1, grid - 1))
    gy1 = max(0, min(gy1, grid - 1))
    gx2 = max(0, min(gx2, grid - 1))
    gy2 = max(0, min(gy2, grid - 1))
    indices = []
    for gy in range(gy1, gy2 + 1):
        for gx in range(gx1, gx2 + 1):
            indices.append(gy * grid + gx)
    return indices


def box_to_loc_tokens(box_xyxy: Sequence[float], image_size: Tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = box_xyxy

    def _q(v, upper):
        upper = max(float(upper), 1.0)
        return int(np.clip(round(float(v) / upper * 999.0), 0, 999))

    return f"<loc_{_q(x1, width)}><loc_{_q(y1, height)}><loc_{_q(x2, width)}><loc_{_q(y2, height)}>"


def build_box_bundle(meta: Dict, canvas_size: int = 768) -> Dict:
    box_xywh_orig = list(meta["source_object_boxes"][0])
    image_size = (int(meta["image_width"]), int(meta["image_height"]))
    box_xyxy_orig = xywh_to_xyxy(box_xywh_orig)
    canvas_size = max(1, int(canvas_size))
    box_xyxy_canvas = clamp_xyxy(
        map_box_to_canvas(box_xywh_orig, image_size, (canvas_size, canvas_size)),
        canvas_size,
    )
    token_grid = max(1, canvas_size // 32)
    token_indices_canvas = box_to_token_indices(
        box_xyxy_canvas, canvas_size=canvas_size, grid=token_grid
    )
    return {
        "box_xywh_orig": box_xywh_orig,
        "box_xyxy_orig": box_xyxy_orig,
        "box_xyxy_canvas": box_xyxy_canvas,
        "token_indices_canvas": token_indices_canvas,
        "token_grid": token_grid,
        "box_xyxy_768": box_xyxy_canvas,
        "token_indices_24x24": token_indices_canvas,
        "orig_size": image_size,
        "canvas_size": canvas_size,
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_records(run):
    path = Path(run) / "per_sample_metrics.jsonl"
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["image_id"] in result:
                raise ValueError("Duplicate sample record")
            result[row["image_id"]] = row
    return result


def save_records(run, records):
    path = Path(run) / "per_sample_metrics.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in records.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def save_image(path, tensor):
    from PIL import Image
    import numpy as np

    array = tensor.detach().clamp(0, 1).cpu().numpy()[0]
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)


def validate_config(c):
    a = c["attack"]
    if not 1 <= a["start_step"] < a["diffusion_steps"]:
        raise ValueError("Invalid diffusion interval")
    if a["iterations"] < 1 or a["latent_lr"] <= 0:
        raise ValueError("Iterations and learning rate must be positive")
