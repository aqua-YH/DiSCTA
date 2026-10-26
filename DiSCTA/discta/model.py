from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, modeling_utils
from .data import box_to_loc_tokens, clamp_xyxy
import json, re
from pathlib import Path


def _normalize_category(text: str) -> str:
    return str(text).strip().lower()


def token_target_margin_loss(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    source_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"expected [batch, sequence, vocab] logits, got {tuple(logits.shape)}"
        )
    positions = target_positions.to(device=logits.device, dtype=torch.long).flatten()
    target_ids = target_token_ids.to(device=logits.device, dtype=torch.long).flatten()
    source_ids = (
        source_token_ids.to(device=logits.device, dtype=torch.long).flatten().unique()
    )
    if positions.numel() == 0 or target_ids.numel() == 0 or source_ids.numel() == 0:
        return logits.sum() * 0.0
    positions = positions.clamp(0, logits.shape[1] - 1)
    target_ids = target_ids[
        torch.arange(positions.numel(), device=logits.device) % target_ids.numel()
    ]
    step_logits = logits[0, positions]
    target_scores = step_logits.gather(1, target_ids[:, None]).squeeze(1)
    source_scores = step_logits[:, source_ids].logsumexp(dim=1)
    return F.relu(source_scores - target_scores + float(margin)).mean()


def token_source_unlikelihood_loss(
    logits: torch.Tensor, source_token_ids: torch.Tensor, target_positions: torch.Tensor
) -> torch.Tensor:
    positions = target_positions.to(device=logits.device, dtype=torch.long).flatten()
    source_ids = (
        source_token_ids.to(device=logits.device, dtype=torch.long).flatten().unique()
    )
    if positions.numel() == 0 or source_ids.numel() == 0:
        return logits.sum() * 0.0
    positions = positions.clamp(0, logits.shape[1] - 1)
    probabilities = logits[0, positions].float().softmax(dim=-1)
    source_probability = probabilities[:, source_ids].sum(dim=1).clamp(max=1.0 - 1e-06)
    return -torch.log1p(-source_probability).mean()


class FlorenceFeatureAdapter:

    def __init__(
        self,
        model_path: str,
        image_size: int = 768,
        use_fp16: bool = True,
        categories: Optional[Sequence[str]] = None,
        tau: float = 0.9,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and (not torch.cuda.is_available()):
            device = "cpu"
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.image_size = int(image_size)
        self.tau = float(tau)
        self.categories = [
            _normalize_category(x) for x in categories or [] if _normalize_category(x)
        ]
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa"):
            modeling_utils.PreTrainedModel._supports_sdpa = False
        if not hasattr(modeling_utils.PreTrainedModel, "_supports_flash_attn_2"):
            modeling_utils.PreTrainedModel._supports_flash_attn_2 = False
        if not hasattr(modeling_utils.PreTrainedModel, "_supports_flex_attn"):
            modeling_utils.PreTrainedModel._supports_flex_attn = False
        model_dtype = torch.float16 if self.use_fp16 else torch.float32
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
        self.model.requires_grad_(False)
        self.pixel_mean = torch.tensor(
            getattr(
                self.processor.image_processor, "image_mean", [0.485, 0.456, 0.406]
            ),
            device=self.device,
        ).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(
            getattr(self.processor.image_processor, "image_std", [0.229, 0.224, 0.225]),
            device=self.device,
        ).view(1, 3, 1, 1)
        self.text_embed_cache: Dict[str, torch.Tensor] = {}

    def _task_prompt(self, task_prompt: str) -> str:
        if hasattr(self.processor, "_construct_prompts"):
            try:
                return self.processor._construct_prompts([task_prompt])[0]
            except Exception:
                return task_prompt
        return task_prompt

    def _normalize_pixel_values(self, image_01: torch.Tensor) -> torch.Tensor:
        image_01 = image_01.to(self.device)
        pixel_values = F.interpolate(
            image_01,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (pixel_values - self.pixel_mean) / self.pixel_std
        if self.use_fp16:
            pixel_values = pixel_values.half()
        return pixel_values

    def _project_local_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            local_tokens = self.model.vision_tower.forward_features_unpool(pixel_values)
            if local_tokens.ndim != 3:
                raise ValueError(
                    f"unexpected Florence token shape: {tuple(local_tokens.shape)}"
                )
            batch_size, token_count, token_dim = local_tokens.shape
            side = int(token_count**0.5)
            if side * side != token_count:
                raise ValueError(f"Florence token count is not square: {token_count}")
            if getattr(self.model, "image_pos_embed", None) is not None:
                local_tokens_2d = local_tokens.view(batch_size, side, side, token_dim)
                local_tokens_2d = local_tokens_2d + self.model.image_pos_embed(
                    local_tokens_2d
                )
                local_tokens = local_tokens_2d.view(batch_size, token_count, token_dim)
            local_tokens = local_tokens @ self.model.image_projection
            local_tokens = self.model.image_proj_norm(local_tokens)
            local_tokens = torch.nan_to_num(local_tokens.float())
        return local_tokens

    def _token_grid_weights(
        self, box_xyxy_canvas: Sequence[float], token_count: int
    ) -> torch.Tensor:
        box_xyxy_canvas = clamp_xyxy(box_xyxy_canvas, self.image_size)
        x1, y1, x2, y2 = box_xyxy_canvas
        grid = int(round(token_count**0.5))
        if grid * grid != token_count:
            raise ValueError(
                f"Florence visual token count must be square, got {token_count}"
            )
        patch = float(self.image_size) / grid
        weights = torch.zeros(
            (1, token_count, 1), device=self.device, dtype=torch.float32
        )
        for gy in range(grid):
            for gx in range(grid):
                cell_x1 = gx * patch
                cell_y1 = gy * patch
                cell_x2 = (gx + 1) * patch
                cell_y2 = (gy + 1) * patch
                inter_w = max(0.0, min(cell_x2, x2) - max(cell_x1, x1))
                inter_h = max(0.0, min(cell_y2, y2) - max(cell_y1, y1))
                inter_area = inter_w * inter_h
                weights[0, gy * grid + gx, 0] = float(inter_area) / float(patch * patch)
        if float(weights.sum().item()) <= 0.0:
            cx = int(np.clip((x1 + x2) * 0.5 / patch, 0, grid - 1))
            cy = int(np.clip((y1 + y2) * 0.5 / patch, 0, grid - 1))
            weights.zero_()
            weights[0, cy * grid + cx, 0] = 1.0
        return weights

    def encode_region_feature(
        self, image_01: torch.Tensor, box_xyxy_canvas: Sequence[float]
    ) -> torch.Tensor:
        pixel_values = self._normalize_pixel_values(image_01)
        local_tokens = self._project_local_tokens(pixel_values)
        box_weights = self._token_grid_weights(
            box_xyxy_canvas, token_count=local_tokens.shape[1]
        ).to(device=local_tokens.device, dtype=local_tokens.dtype)
        weights = box_weights
        pooled = (local_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
            1e-06
        )
        pooled = F.normalize(pooled.float(), dim=-1)
        return pooled

    def encode_category(self, text: str) -> torch.Tensor:
        key = _normalize_category(text)
        if key in self.text_embed_cache:
            return self.text_embed_cache[key]
        with torch.no_grad():
            tokenized = self.tokenizer(
                key, return_tensors="pt", padding=False, truncation=True, max_length=16
            )
            tokenized = {k: v.to(self.device) for (k, v) in tokenized.items()}
            embeds = self.model.get_input_embeddings()(tokenized["input_ids"])
            if embeds.shape[1] > 2:
                embeds = embeds[:, 1:-1, :]
            category_embed = embeds.mean(dim=1)
            category_embed = F.normalize(
                torch.nan_to_num(category_embed.float()), dim=-1
            )
            self.text_embed_cache[key] = category_embed.detach()
        return self.text_embed_cache[key]

    def encode_text_target(self, text: str) -> torch.Tensor:
        key = "text:" + str(text).strip()
        if key in self.text_embed_cache:
            return self.text_embed_cache[key]
        with torch.no_grad():
            tokenized = self.tokenizer(
                str(text).strip(),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=64,
            )
            tokenized = {k: v.to(self.device) for (k, v) in tokenized.items()}
            embeds = self.model.get_input_embeddings()(tokenized["input_ids"])
            if embeds.shape[1] > 2:
                embeds = embeds[:, 1:-1, :]
            value = F.normalize(torch.nan_to_num(embeds.float().mean(dim=1)), dim=-1)
            self.text_embed_cache[key] = value.detach()
        return self.text_embed_cache[key]

    def _category_token_ids(self, category: str) -> torch.Tensor:
        encoded = self.tokenizer(
            _normalize_category(category), add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        return encoded.flatten()

    @staticmethod
    def _find_token_positions(
        label_ids: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        labels = label_ids.flatten()
        tokens = token_ids.flatten()
        if tokens.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=labels.device)
        for start in range(max(0, labels.numel() - tokens.numel() + 1)):
            if torch.equal(labels[start : start + tokens.numel()], tokens):
                return torch.arange(start, start + tokens.numel(), device=labels.device)
        return torch.empty(0, dtype=torch.long, device=labels.device)

    def _targeted_sequence_loss(
        self,
        image_01: torch.Tensor,
        prompt: str,
        label_text: str,
        target_category: str,
        source_category: str,
        margin_weight: float,
        source_unlikelihood_weight: float,
        margin: float,
        max_length: int = 64,
    ) -> torch.Tensor:
        pixel_values = self._normalize_pixel_values(image_01)
        prompt_tokens = self.tokenizer(
            self._task_prompt(prompt), return_tensors="pt", truncation=True
        ).to(self.device)
        raw_labels = self.tokenizer(
            str(label_text).strip(),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).input_ids.to(self.device)
        labels = raw_labels.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.model(
            input_ids=prompt_tokens.input_ids,
            attention_mask=prompt_tokens.attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        target_ids = self._category_token_ids(target_category)
        source_ids = self._category_token_ids(source_category)
        positions = self._find_token_positions(raw_labels, target_ids)
        if positions.numel() == 0:
            spaced_target = (
                self.tokenizer(
                    " " + _normalize_category(target_category),
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                .input_ids.to(self.device)
                .flatten()
            )
            spaced_positions = self._find_token_positions(raw_labels, spaced_target)
            if spaced_positions.numel() > 0:
                target_ids = spaced_target
                source_ids = (
                    self.tokenizer(
                        " " + _normalize_category(source_category),
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    .input_ids.to(self.device)
                    .flatten()
                )
                positions = spaced_positions
        if positions.numel() == 0:
            valid = torch.nonzero(raw_labels.flatten().ge(0), as_tuple=False).flatten()
            positions = valid[: max(1, target_ids.numel())]
        margin_loss = token_target_margin_loss(
            outputs.logits, target_ids, source_ids, positions, margin=margin
        )
        source_loss = token_source_unlikelihood_loss(
            outputs.logits, source_ids, positions
        )
        return (
            outputs.loss.float()
            + float(margin_weight) * margin_loss.float()
            + float(source_unlikelihood_weight) * source_loss.float()
        )

    def region_targeted_loss(
        self,
        image_01: torch.Tensor,
        target: Dict,
        box_xyxy_canvas: Sequence[float],
        margin_weight: float = 0.0,
        source_unlikelihood_weight: float = 0.0,
        margin: float = 1.0,
    ) -> torch.Tensor:
        region_token = box_to_loc_tokens(
            box_xyxy_canvas, (self.image_size, self.image_size)
        )
        return self._targeted_sequence_loss(
            image_01,
            f"<REGION_TO_CATEGORY>{region_token}",
            _normalize_category(target["target_object_category"]),
            target["target_object_category"],
            target["source_object_category"],
            margin_weight,
            source_unlikelihood_weight,
            margin,
            max_length=16,
        )

    def caption_targeted_loss(
        self,
        image_01: torch.Tensor,
        caption: str,
        target: Dict,
        margin_weight: float = 0.0,
        source_unlikelihood_weight: float = 0.0,
        margin: float = 1.0,
    ) -> torch.Tensor:
        return self._targeted_sequence_loss(
            image_01,
            "<CAPTION>",
            caption,
            target["target_object_category"],
            target["source_object_category"],
            margin_weight,
            source_unlikelihood_weight,
            margin,
            max_length=64,
        )


def build_counterfactual_caption_bank(target: Dict, minimum_size: int = 1) -> List[str]:
    source = str(target.get("source_object_category", "")).strip()
    replacement = str(target.get("target_object_category", "")).strip()
    values = [str(target.get("caption_label", "")).strip()]
    if source and replacement:
        pattern = re.compile(f"\\b{re.escape(source)}\\b", flags=re.IGNORECASE)
        for caption in target.get("captions", []) or []:
            caption = str(caption).strip()
            if caption and pattern.search(caption):
                values.append(pattern.sub(replacement, caption))
    seen = set()
    bank = []
    for value in values:
        key = value.casefold()
        if (
            value
            and key not in seen
            and (
                not (
                    source
                    and re.search(
                        f"\\b{re.escape(source)}\\b", key, flags=re.IGNORECASE
                    )
                )
            )
        ):
            seen.add(key)
            bank.append(value)
    if not bank:
        bank.append(f"A {replacement}." if replacement else "An object.")
    minimum_size = max(1, int(minimum_size))
    base = bank[0].strip().rstrip(". !?")
    base_lower = base[0].lower() + base[1:] if base else replacement
    variants = [
        f"This image shows {base_lower}.",
        f"In the scene, {base_lower}.",
        f"A photograph shows {base_lower}.",
        f"The visible object is a {replacement}.",
    ]
    for value in variants:
        key = value.casefold()
        if len(bank) >= minimum_size:
            break
        if (
            value
            and key not in seen
            and (
                not (
                    source
                    and re.search(
                        f"\\b{re.escape(source)}\\b", key, flags=re.IGNORECASE
                    )
                )
            )
        ):
            seen.add(key)
            bank.append(value)
    return bank


def dynamic_contrast(
    adapter, image, target, box, categories, caption, topk=5, eta=0.25
):
    """Source plus current regional Top-K competitors; no gradient through selection."""
    normalize = lambda x: str(x).strip().lower()
    source, destination = (
        normalize(target["source_object_category"]),
        normalize(target["target_object_category"]),
    )
    v = adapter.encode_region_feature(image, box)
    positive = F.normalize(
        (1 - eta) * adapter.encode_category(destination)
        + eta * adapter.encode_text_target(caption),
        dim=-1,
    )
    resource = Path(__file__).resolve().parents[1] / "resources/mscoco_categories.json"
    coco = [
        normalize(c["name"]) for c in json.loads(resource.read_text())["categories"]
    ]
    pool = []
    for category in list(adapter.categories) + list(categories) + coco:
        category = normalize(category)
        if category and category != destination and (category not in pool):
            pool.append(category)
    candidates = [c for c in pool if c != source]
    embeddings = torch.cat([adapter.encode_category(c) for c in candidates], dim=0)
    indices = (
        (embeddings @ v.detach().squeeze(0))
        .topk(min(topk, len(candidates)))
        .indices.tolist()
    )
    negatives = [source] + [candidates[i] for i in indices]
    eneg = torch.cat([adapter.encode_category(c) for c in negatives], dim=0)
    pos = F.cosine_similarity(v, positive, dim=-1)
    neg = F.cosine_similarity(v.unsqueeze(1), eneg.unsqueeze(0), dim=-1)
    weights = torch.ones(len(negatives), device=eneg.device, dtype=neg.dtype).view(
        1, -1
    )
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
    loss = (F.relu(neg - pos.unsqueeze(1) + adapter.tau) * weights).sum(dim=1).mean()
    return (loss, negatives)


def _loc_tokens(box_xyxy: Sequence[float], image_size):
    width, height = image_size
    x1, y1, x2, y2 = box_xyxy

    def q(value, upper):
        return int(
            np.clip(round(float(value) / max(float(upper), 1.0) * 999.0), 0, 999)
        )

    return f"<loc_{q(x1, width)}><loc_{q(y1, height)}><loc_{q(x2, width)}><loc_{q(y2, height)}>"


def _xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def build_task_targets(meta: Dict, region_box_canvas, canvas_size: int):
    target = str(meta["target_object_category"]).strip()
    target_caption = str(meta.get("caption_label", "")).strip()
    if not target or not target_caption:
        raise ValueError("Target category and target caption are required")
    original_size = (int(meta["image_width"]), int(meta["image_height"]))
    source_boxes = [_xywh_to_xyxy(x) for x in meta["source_object_boxes"]]
    grouped = [(target, source_boxes)]
    for category, boxes in (meta.get("other_objects") or {}).items():
        grouped.append((str(category).strip(), [_xywh_to_xyxy(x) for x in boxes]))
    detection_label = "".join(
        (
            category + "".join((_loc_tokens(box, original_size) for box in boxes))
            for (category, boxes) in grouped
            if category and boxes
        )
    )
    localization_label = target + "".join(
        (_loc_tokens(box, original_size) for box in source_boxes)
    )
    region_prompt = "<REGION_TO_CATEGORY>" + _loc_tokens(
        region_box_canvas, (canvas_size, canvas_size)
    )
    return {
        "caption": ("<CAPTION>", target_caption, 96),
        "detection": ("<OD>", detection_label, 256),
        "region": (region_prompt, target, 16),
        "localization": (
            "<OPEN_VOCABULARY_DETECTION>" + target,
            localization_label,
            64,
        ),
    }


def sequence_task_loss(adapter, image, target, box, task):
    if task not in ("detection", "localization"):
        raise ValueError(task)
    prompt, label, _ = build_task_targets(target, box, adapter.image_size)[task]
    prompt_tokens = adapter.tokenizer(
        adapter._task_prompt(prompt), return_tensors="pt"
    ).to(adapter.device)
    labels = adapter.tokenizer(
        label, return_tensors="pt", truncation=False
    ).input_ids.to(adapter.device)
    if labels.shape[1] > 1024:
        raise ValueError(
            "OD/OL label exceeds decoder budget; do not silently truncate target boxes"
        )
    labels = labels.clone()
    labels[labels == adapter.tokenizer.pad_token_id] = -100
    result = adapter.model(
        input_ids=prompt_tokens.input_ids,
        attention_mask=prompt_tokens.attention_mask,
        pixel_values=adapter._normalize_pixel_values(image),
        labels=labels,
    )
    return result.loss.float()
