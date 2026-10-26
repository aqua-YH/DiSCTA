from __future__ import annotations

import json
import os
import re
import string
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, modeling_utils
from .data import scale_xyxy, xywh_to_xyxy
import importlib
import math
from pathlib import Path


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def _normalize_word(w: str):
    w = w.lower().strip()
    w = w.translate(str.maketrans("", "", string.punctuation))
    if len(w) > 3 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 2 and w.endswith("s"):
        w = w[:-1]
    return w


def _normalize_text(text: str):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub("\\s+", " ", text).strip()
    return text


def _normalize_category_phrase(value: str):
    tokens = [_normalize_word(token) for token in _normalize_text(value).split()]
    return " ".join((token for token in tokens if token))


def normalize_category_name(value: str):
    return _normalize_category_phrase(value)


_SAFE_CONTAINS_BLOCKLIST = {
    ("dog", "hot dog"),
    ("bat", "baseball bat"),
    ("glass", "wine glass"),
    ("light", "traffic light"),
}


def match_category(pred: str, target: str, mode: str = "safe_contains"):
    left = normalize_category_name(pred)
    right = normalize_category_name(target)
    if not left or not right:
        return False
    if left == right:
        return True
    pair = (left, right)
    if mode == "safe_contains" and (
        pair in _SAFE_CONTAINS_BLOCKLIST or pair[::-1] in _SAFE_CONTAINS_BLOCKLIST
    ):
        return False

    def contains_phrase(haystack, needle):
        return re.search(f"(?:^|\\s){re.escape(needle)}(?:\\s|$)", haystack) is not None

    return contains_phrase(left, right) or contains_phrase(right, left)


def _category_labels_match(left: str, right: str):
    return match_category(left, right, mode="safe_contains")


def _contains_category(text: str, category: str):
    norm_text = _normalize_text(text)
    norm_cat = _normalize_text(category)
    if not norm_cat:
        return False
    if re.search(f"(^|\\s){re.escape(norm_cat)}(\\s|$)", norm_text):
        return True
    words = [_normalize_word(x) for x in re.split("\\s+", norm_text)]
    words = [w for w in words if w]
    cat_tokens = [_normalize_word(x) for x in norm_cat.split() if _normalize_word(x)]
    if len(cat_tokens) == 1:
        return cat_tokens[0] in words
    return all((tok in words for tok in cat_tokens))


def _contains_category_bidirectional(text: str, category: str):
    if _contains_category(text, category):
        return True
    norm_cat = _normalize_category_phrase(category)
    if not norm_cat:
        return False
    words = [_normalize_word(x) for x in _normalize_text(text).split()]
    words = [w for w in words if w]
    max_len = min(len(words), len(norm_cat.split()))
    for n in range(1, max_len + 1):
        for i in range(0, len(words) - n + 1):
            if _category_labels_match(" ".join(words[i : i + n]), norm_cat):
                return True
    return False


def _strip_loc_tokens(text: str):
    return re.sub("<loc_\\d+>", " ", str(text))


def _quantize_box_to_loc_tokens(box, image_size: Tuple[int, int]):
    width, height = image_size
    x1, y1, x2, y2 = box

    def _q(v, upper):
        upper = max(float(upper), 1.0)
        return int(np.clip(round(v / upper * 999.0), 0, 999))

    qx1 = _q(x1, width)
    qy1 = _q(y1, height)
    qx2 = _q(x2, width)
    qy2 = _q(y2, height)
    return f"<loc_{qx1}><loc_{qy1}><loc_{qx2}><loc_{qy2}>"


def _as_text(parsed, raw_text: str):
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for _, v in parsed.items():
            if isinstance(v, str):
                return v
    return raw_text


def _unwrap_task_output(parsed):
    if isinstance(parsed, dict) and len(parsed) == 1:
        k = list(parsed.keys())[0]
        if isinstance(k, str) and k.startswith("<"):
            return parsed[k]
    return parsed


def _as_box_pred(parsed):
    parsed = _unwrap_task_output(parsed)
    if isinstance(parsed, dict):
        if "bboxes" in parsed:
            labels = parsed.get("labels", parsed.get("bboxes_labels", []))
            return {"bboxes": parsed.get("bboxes", []), "labels": labels}
        if "polygons" in parsed:
            polys = parsed.get("polygons", [])
            labels = parsed.get("labels", parsed.get("polygons_labels", []))
            bboxes = []
            for poly_group in polys:
                flat = []
                if isinstance(poly_group, list):
                    for p in poly_group:
                        if isinstance(p, list):
                            flat.extend(p)
                if len(flat) >= 4:
                    xs = flat[0::2]
                    ys = flat[1::2]
                    bboxes.append([min(xs), min(ys), max(xs), max(ys)])
            return {"bboxes": bboxes, "labels": labels}
        for _, v in parsed.items():
            if isinstance(v, dict) and "bboxes" in v:
                labels = v.get("labels", v.get("bboxes_labels", []))
                return {"bboxes": v.get("bboxes", []), "labels": labels}
            if isinstance(v, dict) and "polygons" in v:
                polys = v.get("polygons", [])
                labels = v.get("labels", v.get("polygons_labels", []))
                bboxes = []
                for poly_group in polys:
                    flat = []
                    if isinstance(poly_group, list):
                        for p in poly_group:
                            if isinstance(p, list):
                                flat.extend(p)
                    if len(flat) >= 4:
                        xs = flat[0::2]
                        ys = flat[1::2]
                        bboxes.append([min(xs), min(ys), max(xs), max(ys)])
                return {"bboxes": bboxes, "labels": labels}
    return {"bboxes": [], "labels": []}


def _box_payload(box_pred: Dict):
    return {
        "boxes": box_pred.get("bboxes", []),
        "labels": box_pred.get("labels", []),
        "scores": box_pred.get("scores", []),
    }


class FlorenceRunner:

    def __init__(self, model_path: str, use_amp: bool = True, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and (not torch.cuda.is_available()):
            device = "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        if not hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa"):
            modeling_utils.PreTrainedModel._supports_sdpa = False
        if not hasattr(modeling_utils.PreTrainedModel, "_supports_flash_attn_2"):
            modeling_utils.PreTrainedModel._supports_flash_attn_2 = False
        if not hasattr(modeling_utils.PreTrainedModel, "_supports_flex_attn"):
            modeling_utils.PreTrainedModel._supports_flex_attn = False
        model_dtype = torch.float16 if self.use_amp else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                dtype=model_dtype,
                attn_implementation="eager",
            ).to(self.device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=model_dtype,
                attn_implementation="eager",
            ).to(self.device)
        self.model.eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.early_stopping = None

    @torch.no_grad()
    def run_task(self, image: Image.Image, task_prompt: str, max_new_tokens: int = 96):
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for (k, v) in inputs.items()
        }
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=1,
            do_sample=False,
            use_cache=False,
        )
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.generate(**inputs, **gen_kwargs)
        else:
            outputs = self.model.generate(**inputs, **gen_kwargs)
        transition_scores = None
        try:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
        except Exception:
            transition_scores = None
        raw_text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )[0]
        task_token = (
            task_prompt[: task_prompt.find(">") + 1]
            if task_prompt.startswith("<")
            else "<CAPTION>"
        )
        parsed = self.processor.post_process_generation(
            sequence=outputs.sequences[0],
            transition_beam_score=(
                transition_scores[0] if transition_scores is not None else None
            ),
            task=task_token,
            image_size=image.size,
        )
        return (raw_text, parsed)


def _det_success(od_pred: Dict, source_box, target_cat: str, theta_box: float):
    bboxes = od_pred.get("bboxes", [])
    labels = [str(x).lower() for x in od_pred.get("labels", [])]
    for box, label in zip(bboxes, labels):
        if not _category_labels_match(label, target_cat):
            continue
        if _iou(box, source_box) >= theta_box:
            return True
    return False


def _loc_success(loc_pred: Dict, source_box, theta_loc: float):
    bboxes = loc_pred.get("bboxes", [])
    best = 0.0
    for box in bboxes:
        best = max(best, _iou(box, source_box))
    return best >= theta_loc


def _reg_success(region_text: str, target_cat: str):
    return _contains_category_bidirectional(_strip_loc_tokens(region_text), target_cat)


def _cap_success(caption_text: str, source_cat: str, target_cat: str):
    has_target = _contains_category_bidirectional(caption_text, target_cat)
    has_source = _contains_category_bidirectional(caption_text, source_cat)
    return has_target and (not has_source)


def score_cleanimage_predictions(
    *,
    caption_text: str,
    od_pred: Dict,
    region_text: str,
    loc_pred: Dict,
    source_box,
    source_category: str,
    theta_box: float,
    theta_loc: float,
) -> Dict[str, bool]:
    return {
        "cleanimage_ic_hit": _contains_category(caption_text, source_category),
        "cleanimage_od_hit": _det_success(
            od_pred, source_box, source_category, theta_box
        ),
        "cleanimage_rc_hit": _reg_success(region_text, source_category),
        "cleanimage_ol_hit": _loc_success(loc_pred, source_box, theta_loc),
    }


def evaluate_crossvlad_metrics(records: List[Dict], args):
    runner = FlorenceRunner(
        args.florence_model_path,
        use_amp=bool(getattr(args, "use_amp", True)),
        device=getattr(args, "eval_device", None),
    )
    theta_box = float(
        getattr(args, "theta_box", getattr(args, "od_iou_threshold", 0.5))
    )
    theta_loc = float(
        getattr(args, "theta_loc", getattr(args, "vg_iou_threshold", 0.5))
    )
    s_cap_all = []
    s_det_all = []
    s_reg_all = []
    s_loc_all = []
    ctsr4_all = []
    ctsr3_all = []
    non_empty_od = 0
    non_empty_loc = 0
    clean_ic_all = []
    clean_od_all = []
    clean_rc_all = []
    clean_ol_all = []
    valid_all = []
    ctsr4_valid_all = []
    ctsr3_valid_all = []
    per_sample = []
    for rec in records:
        meta = rec["meta"]
        source_box = xywh_to_xyxy(meta["source_object_boxes"][0])
        source_cat = str(meta["source_object_category"])
        target_cat = str(meta["target_object_category"])
        src_image_size = (
            (int(meta["image_width"]), int(meta["image_height"]))
            if "image_width" in meta
            else tuple(meta["image_size"])
        )
        clean_img = Image.open(rec["clean_path"]).convert("RGB")
        clean_image_size = clean_img.size
        source_box_clean = scale_xyxy(source_box, src_image_size, clean_image_size)
        clean_cap_raw, clean_cap_pred = runner.run_task(clean_img, "<CAPTION>")
        clean_od_raw, clean_od_pred = runner.run_task(clean_img, "<OD>")
        clean_loc_raw, clean_loc_pred = runner.run_task(
            clean_img, f"<OPEN_VOCABULARY_DETECTION>{source_cat}"
        )
        clean_region_token = _quantize_box_to_loc_tokens(
            source_box_clean, clean_image_size
        )
        clean_reg_raw, clean_reg_pred = runner.run_task(
            clean_img, f"<REGION_TO_CATEGORY>{clean_region_token}"
        )
        clean_cap_text = _as_text(_unwrap_task_output(clean_cap_pred), clean_cap_raw)
        clean_reg_text = _as_text(_unwrap_task_output(clean_reg_pred), clean_reg_raw)
        clean_od_box = _as_box_pred(clean_od_pred)
        clean_loc_box = _as_box_pred(clean_loc_pred)
        clean_scores = score_cleanimage_predictions(
            caption_text=clean_cap_text,
            od_pred=clean_od_box,
            region_text=clean_reg_text,
            loc_pred=clean_loc_box,
            source_box=source_box_clean,
            source_category=source_cat,
            theta_box=theta_box,
            theta_loc=theta_loc,
        )
        valid_clean = any(
            (
                bool(clean_scores[key])
                for key in (
                    "cleanimage_ic_hit",
                    "cleanimage_od_hit",
                    "cleanimage_rc_hit",
                    "cleanimage_ol_hit",
                )
            )
        )
        valid_all.append(float(valid_clean))
        clean_ic_all.append(float(clean_scores["cleanimage_ic_hit"]))
        clean_od_all.append(float(clean_scores["cleanimage_od_hit"]))
        clean_rc_all.append(float(clean_scores["cleanimage_rc_hit"]))
        clean_ol_all.append(float(clean_scores["cleanimage_ol_hit"]))
        adv_img = Image.open(rec["adv_path"]).convert("RGB")
        eval_image_size = adv_img.size
        source_box_eval = scale_xyxy(source_box, src_image_size, eval_image_size)
        cap_raw, cap_pred = runner.run_task(adv_img, "<CAPTION>")
        od_raw, od_pred = runner.run_task(adv_img, "<OD>")
        loc_raw, loc_pred = runner.run_task(
            adv_img, f"<OPEN_VOCABULARY_DETECTION>{target_cat}"
        )
        region_token = _quantize_box_to_loc_tokens(source_box_eval, eval_image_size)
        reg_raw, reg_pred = runner.run_task(
            adv_img, f"<REGION_TO_CATEGORY>{region_token}"
        )
        cap_text = _as_text(_unwrap_task_output(cap_pred), cap_raw)
        reg_text = _as_text(_unwrap_task_output(reg_pred), reg_raw)
        reg_text_clean = _strip_loc_tokens(reg_text)
        od_pred = _as_box_pred(od_pred)
        loc_pred = _as_box_pred(loc_pred)
        if len(od_pred.get("bboxes", [])) > 0:
            non_empty_od += 1
        if len(loc_pred.get("bboxes", [])) > 0:
            non_empty_loc += 1
        s_cap = _cap_success(cap_text, source_cat, target_cat)
        s_det = _det_success(od_pred, source_box_eval, target_cat, theta_box)
        s_reg = _reg_success(reg_text, target_cat)
        s_loc = _loc_success(loc_pred, source_box_eval, theta_loc)
        ctsr4 = s_cap and s_det and s_reg and s_loc
        ctsr3 = int(s_cap) + int(s_det) + int(s_reg) + int(s_loc) >= 3
        s_cap_all.append(float(s_cap))
        s_det_all.append(float(s_det))
        s_reg_all.append(float(s_reg))
        s_loc_all.append(float(s_loc))
        ctsr4_all.append(float(ctsr4))
        ctsr3_all.append(float(ctsr3))
        if valid_clean:
            ctsr4_valid_all.append(float(ctsr4))
            ctsr3_valid_all.append(float(ctsr3))
        per_sample.append(
            {
                "id": rec["id"],
                "S_cap": bool(s_cap),
                "S_det": bool(s_det),
                "S_reg": bool(s_reg),
                "S_loc": bool(s_loc),
                "CTSR_4_hit": bool(ctsr4),
                "CTSR_3_hit": bool(ctsr3),
                "valid_subset": bool(valid_clean),
                **clean_scores,
                "raw_outputs": {
                    "caption_text": cap_text,
                    "det_boxes": _box_payload(od_pred)["boxes"],
                    "det_labels": _box_payload(od_pred)["labels"],
                    "det_scores": _box_payload(od_pred)["scores"],
                    "region_cls_text": reg_text_clean,
                    "loc_boxes": _box_payload(loc_pred)["boxes"],
                    "loc_scores": _box_payload(loc_pred)["scores"],
                },
                "clean_raw_outputs": {
                    "caption_text": clean_cap_text,
                    "det_boxes": _box_payload(clean_od_box)["boxes"],
                    "det_labels": _box_payload(clean_od_box)["labels"],
                    "det_scores": _box_payload(clean_od_box)["scores"],
                    "region_cls_text": _strip_loc_tokens(clean_reg_text),
                    "loc_boxes": _box_payload(clean_loc_box)["boxes"],
                    "loc_scores": _box_payload(clean_loc_box)["scores"],
                },
                "cleanimage_caption_raw": clean_cap_raw,
                "cleanimage_od_raw": clean_od_raw,
                "cleanimage_loc_raw": clean_loc_raw,
                "cleanimage_reg_raw": clean_reg_raw,
                "caption_raw": cap_raw,
                "od_raw": od_raw,
                "loc_raw": loc_raw,
                "reg_text": reg_text,
                "reg_text_clean": reg_text_clean,
                "reg_raw": reg_raw,
            }
        )
    summary = {
        "num_samples": len(records),
        "ASR_cap": float(np.mean(s_cap_all)) if s_cap_all else 0.0,
        "ASR_det": float(np.mean(s_det_all)) if s_det_all else 0.0,
        "ASR_reg": float(np.mean(s_reg_all)) if s_reg_all else 0.0,
        "ASR_loc": float(np.mean(s_loc_all)) if s_loc_all else 0.0,
        "CTSR_4": float(np.mean(ctsr4_all)) if ctsr4_all else 0.0,
        "CTSR_3": float(np.mean(ctsr3_all)) if ctsr3_all else 0.0,
        "CTSR4_all": float(np.mean(ctsr4_all)) if ctsr4_all else 0.0,
        "CTSR3_all": float(np.mean(ctsr3_all)) if ctsr3_all else 0.0,
        "CTSR4_valid": float(np.mean(ctsr4_valid_all)) if ctsr4_valid_all else 0.0,
        "CTSR3_valid": float(np.mean(ctsr3_valid_all)) if ctsr3_valid_all else 0.0,
        "valid_rate": float(np.mean(valid_all)) if valid_all else 0.0,
        "cleanimage_ic": float(np.mean(clean_ic_all)) if clean_ic_all else 0.0,
        "cleanimage_od": float(np.mean(clean_od_all)) if clean_od_all else 0.0,
        "cleanimage_rc": float(np.mean(clean_rc_all)) if clean_rc_all else 0.0,
        "cleanimage_ol": float(np.mean(clean_ol_all)) if clean_ol_all else 0.0,
        "theta_box": theta_box,
        "theta_loc": theta_loc,
        "non_empty_od_ratio": float(non_empty_od / len(records)) if records else 0.0,
        "non_empty_loc_ratio": float(non_empty_loc / len(records)) if records else 0.0,
        "definition_note": "Metrics follow One Object, Multiple Lies (ICCV 2025): S_cap/S_det/S_reg/S_loc + CTSR-4/CTSR-3.",
    }
    out = {"summary": summary, "per_sample": per_sample}
    output_path = os.path.join(
        args.save_dir, getattr(args, "metrics_output", "crossvlad_metrics.json")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n========== CrossVLAD Metrics (Paper-style) ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved metrics to: {output_path}")
    return out


try:
    import cv2

    CV2_IMPORT_ERROR = ""
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)


def compute_outside_leak(clean, adversarial, mask) -> float:
    clean_array = np.asarray(clean, dtype=np.float32)
    adversarial_array = np.asarray(adversarial, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=np.float32)
    if clean_array.shape != adversarial_array.shape:
        raise ValueError("clean and adversarial images must have identical shapes")
    if mask_array.shape != clean_array.shape[:2]:
        raise ValueError("mask shape must match image height and width")
    outside = (mask_array < 0.5).astype(np.float32)[..., None]
    denominator = float(outside.sum() * clean_array.shape[-1])
    if denominator <= 0:
        return 0.0
    return float(
        (np.abs(adversarial_array - clean_array) * outside).sum() / denominator
    )


def compute_ssim(clean, adversarial) -> float:
    if cv2 is None:
        raise RuntimeError("cv2 unavailable: " + CV2_IMPORT_ERROR)
    clean_array = np.asarray(clean, dtype=np.float32)
    adversarial_array = np.asarray(adversarial, dtype=np.float32)
    if clean_array.shape != adversarial_array.shape:
        raise ValueError("clean and adversarial images must have identical shapes")
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(clean_array.shape[-1]):
        x = clean_array[..., channel]
        y = adversarial_array[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x_sq = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def _metric_device():
    import torch

    requested = os.environ.get("DIFFATTACK_METRIC_DEVICE", "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_lpips_metric(importer=importlib.import_module):
    try:
        module = importer("lpips")
        metric = module.LPIPS(net="alex")
        metric.eval()
        return (metric, "")
    except Exception as exc:
        return (None, f"LPIPS unavailable: {exc}")


def _lpips_tensor(image_array):
    import torch

    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
    return tensor.mul(2.0).sub(1.0).unsqueeze(0)


def compute_lpips_for_run(image_pairs) -> tuple[float | None, str]:
    metric, error = load_lpips_metric()
    if metric is None:
        return (None, error)
    try:
        import torch

        device = _metric_device()
        metric = metric.to(device)
        values = []
        with torch.no_grad():
            for clean, adversarial in image_pairs:
                values.append(
                    float(
                        metric(
                            _lpips_tensor(clean).to(device),
                            _lpips_tensor(adversarial).to(device),
                        ).item()
                    )
                )
        return (float(np.mean(values)), "")
    except Exception as exc:
        return (None, f"LPIPS failed: {exc}")


def compute_fid_for_run(clean_paths, adversarial_paths) -> tuple[float | None, str]:
    if len(clean_paths) != len(adversarial_paths):
        return (None, "FID requires equal numbers of clean and adversarial images")
    if len(clean_paths) < 2:
        return (None, "FID requires at least 2 samples")
    try:
        from pytorch_fid.fid_score import (
            calculate_activation_statistics,
            calculate_frechet_distance,
        )
        from pytorch_fid.inception import InceptionV3

        device = _metric_device()
        dims = 2048
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)
        kwargs = {
            "model": model,
            "batch_size": min(50, len(clean_paths)),
            "dims": dims,
            "device": device,
            "num_workers": 1,
        }
        mu_clean, sigma_clean = calculate_activation_statistics(clean_paths, **kwargs)
        mu_adv, sigma_adv = calculate_activation_statistics(adversarial_paths, **kwargs)
        return (
            float(calculate_frechet_distance(mu_clean, sigma_clean, mu_adv, sigma_adv)),
            "",
        )
    except Exception as exc:
        return (None, f"FID failed: {exc}")


def compute_run_preservation(run_dir) -> dict:
    root = Path(run_dir)
    selected_ids = json.loads((root / "selected_ids.json").read_text(encoding="utf-8"))
    if not selected_ids:
        raise ValueError("selected_ids.json is empty")
    mae_values, psnr_values, ssim_values, ssim_errors = ([], [], [], [])
    outside_values, missing_masks, image_pairs = ([], [], [])
    clean_paths, adversarial_paths = ([], [])
    for image_id in selected_ids:
        clean_path = root / f"{image_id}_clean.png"
        adversarial_path = root / "adv_images" / f"{image_id}.png"
        clean_paths.append(str(clean_path))
        adversarial_paths.append(str(adversarial_path))
        clean = (
            np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        adversarial = (
            np.asarray(Image.open(adversarial_path).convert("RGB"), dtype=np.float32)
            / 255.0
        )
        image_pairs.append((clean, adversarial))
        difference = clean - adversarial
        mae = float(np.mean(np.abs(difference)))
        mse = float(np.mean(np.square(difference)))
        mae_values.append(mae)
        psnr_values.append(float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse))
        try:
            ssim_values.append(compute_ssim(clean, adversarial))
        except Exception as exc:
            ssim_errors.append(str(exc))
        mask_path = root / "debug" / f"{image_id}_mask.png"
        if mask_path.is_file():
            mask = (
                np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
            )
            outside_values.append(compute_outside_leak(clean, adversarial, mask))
        else:
            missing_masks.append(str(mask_path))
    lpips_value, lpips_error = compute_lpips_for_run(image_pairs)
    fid_value, fid_error = compute_fid_for_run(clean_paths, adversarial_paths)
    return {
        "mae": float(np.mean(mae_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "ssim_error": "" if not ssim_errors else "; ".join(sorted(set(ssim_errors))),
        "outside_leak": float(np.mean(outside_values)) if outside_values else None,
        "outside_leak_error": (
            "" if not missing_masks else f"missing {len(missing_masks)} mask files"
        ),
        "lpips": lpips_value,
        "lpips_error": lpips_error,
        "fid": fid_value,
        "fid_error": fid_error,
    }
