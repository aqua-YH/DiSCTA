from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Sequence
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from .data import clamp_xyxy
import math
from torch import optim
from .model import build_counterfactual_caption_bank, dynamic_contrast
from .model import sequence_task_loss


def _vae_device_dtype(model):
    p = next(model.vae.parameters())
    return (p.device, p.dtype)


def _module_device_dtype(module):
    p = next(module.parameters())
    return (p.device, p.dtype)


def preprocess(image, res=512, device=None, dtype=None):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :]
    if device is not None:
        image = image.to(device=device)
    else:
        image = image.cuda()
    if dtype is not None:
        image = image.to(dtype=dtype)
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    vae_device, vae_dtype = _vae_device_dtype(model)
    image = preprocess(image, res, device=vae_device, dtype=vae_dtype)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


def _amp_ctx(enabled):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def ddim_reverse_sample(
    image,
    prompt,
    model,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    res=512,
):
    batch_size = 1
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    model.scheduler.set_timesteps(num_inference_steps)
    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)
    all_latents = [latents]
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        next_timestep = (
            t
            + model.scheduler.config.num_train_timesteps
            // model.scheduler.num_inference_steps
        )
        alpha_bar_next = (
            model.scheduler.alphas_cumprod[next_timestep]
            if next_timestep <= model.scheduler.config.num_train_timesteps
            else torch.tensor(0.0, device=latents.device)
        )
        reverse_x0 = (
            1
            / torch.sqrt(model.scheduler.alphas_cumprod[t])
            * (latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t]))
        )
        latents = (
            reverse_x0 * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * noise_pred
        )
        all_latents.append(latents)
    return (latents, all_latents)


def diffusion_step_once(
    model, latents, context, t, guidance_scale, use_amp=True, force_fp32=False
):
    unet_device, unet_dtype = _module_device_dtype(model.unet)
    latents_cast = latents.to(device=unet_device, dtype=unet_dtype)
    context = context.to(device=unet_device, dtype=unet_dtype)
    latents_input = torch.cat([latents_cast] * 2)
    with _amp_ctx(use_amp and (not force_fp32)):
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    return model.scheduler.step(
        noise_pred.float(), t, latents.to(device=unet_device, dtype=torch.float32)
    )["prev_sample"]


def diffusion_step(model, latents, context, t, guidance_scale, use_amp=True):
    prev_sample = diffusion_step_once(
        model, latents, context, t, guidance_scale, use_amp=use_amp, force_fp32=False
    )
    if _is_finite_tensor(prev_sample) or not use_amp:
        return prev_sample
    return diffusion_step_once(
        model, latents, context, t, guidance_scale, use_amp=False, force_fp32=True
    )


def _is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def build_soft_mask(
    box_xyxy: Sequence[float],
    res: int,
    device: torch.device,
    expand_ratio: float = 0.1,
    blur_sigma: float = 4.0,
) -> torch.Tensor:
    x1, y1, x2, y2 = clamp_xyxy(box_xyxy, res)
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    expand_x = box_w * expand_ratio * 0.5
    expand_y = box_h * expand_ratio * 0.5
    x1 = max(0.0, x1 - expand_x)
    y1 = max(0.0, y1 - expand_y)
    x2 = min(float(res), x2 + expand_x)
    y2 = min(float(res), y2 + expand_y)
    mask_img = Image.new("L", (res, res), 0)
    drawer = ImageDraw.Draw(mask_img)
    drawer.rectangle([x1, y1, x2, y2], fill=255)
    if blur_sigma > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask_np)[None, None, :, :].to(device=device)
    return mask


def build_grabcut_soft_mask(
    image: Image.Image,
    box_xyxy: Sequence[float],
    res: int,
    device: torch.device,
    expand_ratio: float = 0.1,
    blur_sigma: float = 4.0,
    grabcut_iter: int = 5,
) -> torch.Tensor:
    try:
        import cv2
    except ImportError:
        return build_soft_mask(
            box_xyxy, res, device, expand_ratio=expand_ratio, blur_sigma=blur_sigma
        )
    x1, y1, x2, y2 = clamp_xyxy(box_xyxy, res)
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    rx = int(np.clip(round(x1), 0, res - 2))
    ry = int(np.clip(round(y1), 0, res - 2))
    rw = int(np.clip(round(box_w), 1, res - rx))
    rh = int(np.clip(round(box_h), 1, res - ry))
    if rw <= 1 or rh <= 1:
        return build_soft_mask(
            box_xyxy, res, device, expand_ratio=expand_ratio, blur_sigma=blur_sigma
        )
    image_np = np.array(
        image.resize((res, res), resample=Image.LANCZOS).convert("RGB"), dtype=np.uint8
    )
    grabcut_mask = np.zeros((res, res), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image_np,
            grabcut_mask,
            (rx, ry, rw, rh),
            bgd_model,
            fgd_model,
            max(1, int(grabcut_iter)),
            cv2.GC_INIT_WITH_RECT,
        )
    except Exception:
        return build_soft_mask(
            box_xyxy, res, device, expand_ratio=expand_ratio, blur_sigma=blur_sigma
        )
    object_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    object_mask[:ry, :] = 0
    object_mask[ry + rh :, :] = 0
    object_mask[:, :rx] = 0
    object_mask[:, rx + rw :] = 0
    if int(object_mask.sum()) <= 0:
        return build_soft_mask(
            box_xyxy, res, device, expand_ratio=expand_ratio, blur_sigma=blur_sigma
        )
    dilate_px = int(round(max(box_w, box_h) * max(0.0, float(expand_ratio)) * 0.25))
    if dilate_px > 0:
        kernel_size = 2 * dilate_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        object_mask = cv2.dilate(object_mask, kernel, iterations=1)
    mask_img = Image.fromarray(object_mask, mode="L")
    if blur_sigma > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    return torch.from_numpy(mask_np)[None, None, :, :].to(device=device)


def _hard_box_mask(
    box_xyxy: Sequence[float], res: int, device: torch.device
) -> torch.Tensor:
    x1, y1, x2, y2 = clamp_xyxy(box_xyxy, res)
    mask = torch.zeros((1, 1, res, res), device=device, dtype=torch.float32)
    ix1 = int(np.clip(np.floor(x1), 0, res - 1))
    iy1 = int(np.clip(np.floor(y1), 0, res - 1))
    ix2 = int(np.clip(np.ceil(x2), ix1 + 1, res))
    iy2 = int(np.clip(np.ceil(y2), iy1 + 1, res))
    mask[:, :, iy1:iy2, ix1:ix2] = 1.0
    return mask


def _grabcut_foreground_mask(
    image: Image.Image,
    box_xyxy: Sequence[float],
    res: int,
    device: torch.device,
    grabcut_iter: int = 5,
) -> torch.Tensor:
    mask_box = _hard_box_mask(box_xyxy, res, device)
    try:
        import cv2
    except ImportError:
        return mask_box
    x1, y1, x2, y2 = clamp_xyxy(box_xyxy, res)
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    rx = int(np.clip(round(x1), 0, res - 2))
    ry = int(np.clip(round(y1), 0, res - 2))
    rw = int(np.clip(round(box_w), 1, res - rx))
    rh = int(np.clip(round(box_h), 1, res - ry))
    if rw <= 1 or rh <= 1:
        return mask_box
    image_np = np.array(
        image.resize((res, res), resample=Image.LANCZOS).convert("RGB"), dtype=np.uint8
    )
    grabcut_mask = np.zeros((res, res), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image_np,
            grabcut_mask,
            (rx, ry, rw, rh),
            bgd_model,
            fgd_model,
            max(1, int(grabcut_iter)),
            cv2.GC_INIT_WITH_RECT,
        )
    except Exception:
        return mask_box
    object_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)
    object_mask[:ry, :] = 0.0
    object_mask[ry + rh :, :] = 0.0
    object_mask[:, :rx] = 0.0
    object_mask[:, rx + rw :] = 0.0
    if float(object_mask.sum()) <= 0.0:
        return mask_box
    mask_fg = torch.from_numpy(object_mask)[None, None, :, :].to(
        device=device, dtype=torch.float32
    )
    return mask_fg * mask_box


def _morph_boundary(mask: torch.Tensor, boundary_width: int) -> torch.Tensor:
    width = max(1, int(boundary_width))
    kernel_size = 2 * width + 1
    dilated = F.max_pool2d(
        mask.float(), kernel_size=kernel_size, stride=1, padding=width
    )
    eroded = -F.max_pool2d(
        -mask.float(), kernel_size=kernel_size, stride=1, padding=width
    )
    return torch.clamp(dilated - eroded, 0.0, 1.0)


def build_attack_mask_dict(
    image: Image.Image,
    box_xyxy: Sequence[float],
    res: int,
    device: torch.device,
    expand_ratio: float = 0.1,
    blur_sigma: float = 4.0,
    grabcut_iter: int = 5,
    boundary_width: int = 5,
) -> Dict[str, torch.Tensor]:
    mask_box = _hard_box_mask(box_xyxy, res, device)
    mask_fg = _grabcut_foreground_mask(
        image, box_xyxy, res, device, grabcut_iter=grabcut_iter
    )
    soft_mask = build_grabcut_soft_mask(
        image,
        box_xyxy,
        res,
        device,
        expand_ratio=expand_ratio,
        blur_sigma=blur_sigma,
        grabcut_iter=grabcut_iter,
    )
    mask_fg = torch.clamp(mask_fg * mask_box, 0.0, 1.0)
    mask_ring = torch.clamp(mask_box - mask_fg, 0.0, 1.0)
    mask_out = torch.clamp(1.0 - mask_box, 0.0, 1.0)
    mask_bnd = _morph_boundary(mask_fg, boundary_width=boundary_width)
    soft_mask = torch.clamp(soft_mask, 0.0, 1.0)
    return {
        "mask_box": mask_box,
        "mask_fg": mask_fg,
        "mask_ring": mask_ring,
        "mask_out": mask_out,
        "mask_bnd": mask_bnd,
        "soft_mask": soft_mask,
    }


def build_region_weights(
    mask_dict: Dict[str, torch.Tensor],
    alpha_ring: float = 0.5,
    beta_fg: float = 0.05,
    beta_ring: float = 0.3,
    beta_out: float = 1.0,
) -> Dict[str, torch.Tensor]:
    mask_fg = mask_dict["mask_fg"]
    mask_ring = mask_dict["mask_ring"]
    mask_out = mask_dict["mask_out"]
    soft_mask = mask_dict["soft_mask"]
    W_edit = mask_fg + float(alpha_ring) * mask_ring
    W_edit = torch.maximum(W_edit, soft_mask)
    W_edit = torch.clamp(W_edit, 0.0, 1.0)
    W_protect = (
        float(beta_fg) * mask_fg
        + float(beta_ring) * mask_ring
        + float(beta_out) * mask_out
    )
    W_protect = torch.clamp(W_protect, 0.0, 1.0)
    return {"W_edit": W_edit, "W_protect": W_protect}


def _total_variation(image_01: torch.Tensor) -> torch.Tensor:
    diff_h = image_01[:, :, :, 1:] - image_01[:, :, :, :-1]
    diff_v = image_01[:, :, 1:, :] - image_01[:, :, :-1, :]
    return diff_h.abs().mean() + diff_v.abs().mean()


def _mask_to_latent(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        mask, size=latent.shape[-2:], mode="bilinear", align_corners=False
    ).to(device=latent.device, dtype=latent.dtype)


def compute_region_latent_loss(latent_param, z_init, W_protect_z, eps=1e-08):
    loss = ((latent_param - z_init) ** 2 * W_protect_z).sum()
    denom = W_protect_z.sum() * latent_param.shape[1] + eps
    return loss / denom


def compute_latent_tv_loss(latent_param, z_init, W_edit_z):
    delta_z = latent_param - z_init
    masked_delta_z = delta_z * W_edit_z
    return _total_variation(masked_delta_z)


def compute_boundary_loss(adv_01, init_image_01, mask_bnd, eps=1e-08):
    mask_bnd = mask_bnd.to(device=adv_01.device, dtype=adv_01.dtype)
    loss = (torch.abs(adv_01 - init_image_01) * mask_bnd).sum()
    denom = mask_bnd.sum() * adv_01.shape[1] + eps
    return loss / denom


class RelativeBalance:
    """Four-task coefficients, temperature 1; reference is the first candidate."""

    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon
        self.initial = None

    def coefficients(self, losses):
        values = [float(torch.as_tensor(x).detach().float().item()) for x in losses]
        if len(values) != 4 or any((not math.isfinite(x) or x < 0 for x in values)):
            raise FloatingPointError("Expected four finite nonnegative losses")
        if self.initial is None:
            self.initial = tuple(values)
        ratios = [x / max(b, self.epsilon) for (x, b) in zip(values, self.initial)]
        offset = max(ratios)
        exps = [math.exp(x - offset) for x in ratios]
        weights = [4 * x / sum(exps) for x in exps]
        coefficients = [w / max(x, self.epsilon) for (w, x) in zip(weights, values)]
        return (
            coefficients,
            dict(
                tasks=["IC", "OD", "RC", "OL"],
                losses=values,
                initial_losses=list(self.initial),
                relative_losses=ratios,
                weights=weights,
                coefficients=coefficients,
            ),
        )


def extract_image_gradient(loss: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        loss, image, retain_graph=False, create_graph=False, allow_unused=True
    )[0]
    if gradient is None:
        gradient = torch.zeros_like(image)
    return gradient.detach()


def gradient_proxy(
    image: torch.Tensor, gradient: torch.Tensor, value: torch.Tensor | float
) -> torch.Tensor:
    detached_gradient = gradient.detach().to(device=image.device, dtype=image.dtype)
    detached_value = (
        value.detach().to(device=image.device, dtype=torch.float32)
        if torch.is_tensor(value)
        else torch.tensor(float(value), device=image.device, dtype=torch.float32)
    )
    dot = (image * detached_gradient).sum()
    return dot - dot.detach() + detached_value


def finite(*values):
    if any((not bool(torch.isfinite(x).all()) for x in values)):
        raise FloatingPointError(
            "Nonfinite attack tensor; sample is not marked successful"
        )


def initialize(model, image, target, cfg):
    """Original 20-step inversion and four denoising steps, with null-text tuning."""
    prompt = [target["caption_label"]] * 2
    _, trajectory = ddim_reverse_sample(
        image, prompt, model, cfg.diffusion_steps, 0, res=768
    )
    trajectory = trajectory[::-1]
    z_init = trajectory[cfg.start_step - 1].detach().float()
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(
        [prompt[0]],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text = model.text_encoder(text_input.input_ids.to(model.device))[0]
    latent = z_init.clone()
    uncond = uncond.float().detach().clone().requires_grad_(True)
    optimizer = optim.AdamW([uncond], lr=0.05)
    unconditional = []
    for index, t in enumerate(
        tqdm(model.scheduler.timesteps[cfg.start_step :], desc="Null-text optimization")
    ):
        for _ in range(max(1, 4 + 1 * index)):
            context = torch.cat([uncond, text.float()], dim=0)
            predicted = diffusion_step(
                model,
                latent,
                context.to(dtype=text.dtype),
                t,
                cfg.guidance,
                use_amp=True,
            )
            loss = torch.nn.functional.mse_loss(
                predicted.float(), trajectory[cfg.start_step + index].float()
            )
            if not bool(torch.isfinite(loss).all()):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([uncond], max_norm=1.0)
            optimizer.step()
        with torch.no_grad():
            context = torch.cat([uncond, text.float()], dim=0).to(dtype=text.dtype)
            latent = diffusion_step(
                model, latent, context, t, cfg.guidance, use_amp=True
            ).detach()
            unconditional.append(uncond.detach().clone().to(dtype=text.dtype))
    uncond.requires_grad_(False)
    return (z_init, text, unconditional)


def attack(model, adapter, image, target, box, categories, cfg):
    for component in (model.vae, model.text_encoder, model.unet):
        component.requires_grad_(False)
    z_init, text, unconditional = initialize(model, image, target, cfg)
    latent = torch.nn.Parameter(z_init.clone())
    optimizer = optim.AdamW([latent], lr=cfg.latent_lr, weight_decay=0.0)
    device, dtype = _vae_device_dtype(model)
    clean = (preprocess(image, 768, device=device, dtype=dtype) / 2 + 0.5).clamp(0, 1)
    masks = build_attack_mask_dict(
        image,
        box,
        768,
        clean.device,
        expand_ratio=0.02,
        blur_sigma=1.0,
        grabcut_iter=5,
        boundary_width=5,
    )
    weights = build_region_weights(
        masks, alpha_ring=0.5, beta_fg=0.05, beta_ring=0.3, beta_out=1.0
    )
    soft = masks["soft_mask"].to(clean)
    boundary = masks["mask_bnd"].to(clean)
    edit = _mask_to_latent(weights["W_edit"].to(clean), latent)
    protect = _mask_to_latent(weights["W_protect"].to(clean), latent)
    box_z = _mask_to_latent(masks["mask_box"], latent)
    bank = build_counterfactual_caption_bank(target, minimum_size=4)
    balance = RelativeBalance()
    best_score = float("inf")
    best_image = None
    best_generated = None
    best_step = -1
    history = []
    for step in tqdm(range(cfg.iterations + 1), desc="Attack"):
        terminal = step == cfg.iterations
        current = latent
        for index, t in enumerate(model.scheduler.timesteps[cfg.start_step :]):
            current = diffusion_step(
                model,
                current,
                torch.cat([unconditional[index], text], dim=0),
                t,
                cfg.guidance,
                use_amp=True,
            )
        finite(current)
        with _amp_ctx(True):
            generated = (
                model.vae.decode(1 / 0.18215 * current)["sample"] / 2 + 0.5
            ).clamp(0, 1)
        adv = soft * generated + (1 - soft) * clean
        finite(generated, adv)
        caption = bank[step % len(bank)]
        contrast, negatives = dynamic_contrast(
            adapter, adv, target, box, categories, caption, 5, 0.25
        )
        contrast_gradient = extract_image_gradient(contrast, adv)
        contrast_value = contrast.detach()
        del contrast
        lat_loss = compute_region_latent_loss(latent, z_init, protect)
        tv_loss = compute_latent_tv_loss(latent, z_init, edit)
        boundary_loss = compute_boundary_loss(adv, clean, boundary)
        regularizer = 0.05 * lat_loss + 0.02 * tv_loss + 0.05 * boundary_loss
        cap = adapter.caption_targeted_loss(
            adv,
            caption,
            target,
            margin_weight=1.0,
            source_unlikelihood_weight=0.25,
            margin=1.0,
        ).float()
        cap_gradient = extract_image_gradient(cap, adv)
        cap_value = cap.detach()
        del cap
        reg = adapter.region_targeted_loss(
            adv,
            target,
            box,
            margin_weight=1.0,
            source_unlikelihood_weight=0.25,
            margin=1.0,
        ).float()
        reg_gradient = extract_image_gradient(reg, adv)
        reg_value = reg.detach()
        del reg
        finite(contrast_value, lat_loss, tv_loss, boundary_loss, cap_value, reg_value)
        diagnostics = None
        if not terminal:
            det = sequence_task_loss(adapter, adv, target, box, "detection")
            det_gradient = extract_image_gradient(det, adv)
            det_value = det.detach()
            del det
            loc = sequence_task_loss(adapter, adv, target, box, "localization")
            loc_gradient = extract_image_gradient(loc, adv)
            loc_value = loc.detach()
            del loc
            coefficients, diagnostics = balance.coefficients(
                [cap_value, det_value, reg_value, loc_value]
            )
            auxiliary = torch.zeros_like(contrast_gradient)
            auxiliary = auxiliary + coefficients[0] * cap_gradient
            auxiliary = auxiliary + coefficients[2] * reg_gradient
            combined = cfg.lambda_contrast * contrast_gradient + auxiliary
            combined = (
                combined
                + coefficients[1] * det_gradient
                + coefficients[3] * loc_gradient
            )
            diagnostics["image_gradient_norms"] = [
                float(g.float().norm().item())
                for g in (cap_gradient, det_gradient, reg_gradient, loc_gradient)
            ]
            finite(combined)
            proxy_value = (
                cfg.lambda_contrast * contrast_value
                + coefficients[0] * cap_value
                + coefficients[2] * reg_value
                + coefficients[1] * det_value
                + coefficients[3] * loc_value
            )
            total = gradient_proxy(adv, combined, proxy_value) + regularizer
            optimizer.zero_grad()
            total.backward()
            latent.grad.mul_(edit)
            finite(latent.grad)
            diagnostics["latent_gradient_norm"] = float(
                latent.grad.float().norm().item()
            )
            torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                latent.data = box_z * latent.data + (1 - box_z) * (
                    (1 - 0.2) * latent.data + 0.2 * z_init
                )
            del det_gradient, loc_gradient, combined, auxiliary, total
        score = float(
            (cfg.lambda_contrast * contrast_value + regularizer.detach()).item()
        )
        if not terminal and score < best_score:
            best_score = score
            best_step = step
            best_image = adv.detach().clone()
            best_generated = generated.detach().clone()
        history.append(
            dict(
                iter=step,
                evaluation_only=terminal,
                contrast=float(contrast_value.item()),
                regularizer=float(regularizer.detach().item()),
                selection_score=score,
                negative_classes=negatives,
                caption=caption,
                joint=diagnostics,
            )
        )
        del contrast_gradient, cap_gradient, reg_gradient
    if best_image is None:
        raise RuntimeError("No finite attack candidate")
    return (
        best_image,
        best_generated,
        soft,
        dict(
            status="ok",
            nonfinite_steps=0,
            best_iter=best_step,
            best_selection_score=best_score,
            loss_history=history,
        ),
    )
