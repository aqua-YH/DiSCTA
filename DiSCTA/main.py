from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from discta.data import save_json, load_records, save_records, save_image

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(
        description="DiSCTA: four-task relative-progress local diffusion attack"
    )
    p.add_argument("mode", choices=["attack", "evaluate"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, help="Explicit sample IDs; default: all dataset images"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible completed records in the same output directory",
    )
    p.add_argument(
        "--check-config",
        action="store_true",
        help="Validate paths, manifest and config only; no model loading",
    )
    return p


def prepare(args):
    from discta.data import validate_config

    c = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(c)
    for key in ["images_root", "labels_root", "diffusion_model", "vlm_model"]:
        if not Path(c["paths"][key]).is_dir():
            raise FileNotFoundError(c["paths"][key])
    from discta.data import load_crossvlad_samples

    samples = load_crossvlad_samples(
        c["paths"]["images_root"], c["paths"]["labels_root"]
    )
    by_id = {x["image_id"]: x for x in samples}
    if args.mode == "evaluate":
        ids = json.loads((args.output / "selected_ids.json").read_text())
    elif args.manifest:
        ids = json.loads(args.manifest.read_text())
    else:
        ids = sorted(by_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be nonempty and unique")
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Missing data/labels for IDs: {sorted(missing)[:10]}")
    if args.mode == "attack":
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Invalid shard index/count")
        ids = ids[args.shard_index :: args.num_shards]
        if not ids:
            raise ValueError("Empty shard")
    return (c, samples, [by_id[i] for i in ids], ids)


def generate(args, c, samples, selected, ids):
    import random
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from discta.attack import attack
    from discta.model import FlorenceFeatureAdapter
    from discta.data import build_box_bundle, collect_category_vocab

    cfg = SimpleNamespace(**c["attack"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for generation")
    existing = args.output / "config.json"
    if existing.exists():
        if not args.resume:
            raise FileExistsError("Output already contains a run; use --resume")
        if json.loads(existing.read_text()) != c:
            raise ValueError("Resume config differs from saved config")
        if json.loads((args.output / "selected_ids.json").read_text()) != ids:
            raise ValueError("Resume sample manifest differs")
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["adv_images", "debug"]:
        (args.output / name).mkdir(exist_ok=True)
    save_json(existing, c)
    save_json(args.output / "selected_ids.json", ids)
    save_json(
        args.output / "execution.json",
        dict(
            device=args.device,
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            seed=c["seed"],
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            python=sys.version,
        ),
    )
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = load_records(args.output)
    if not set(records) <= set(ids):
        raise ValueError("Existing records contain IDs outside manifest")
    started = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        c["paths"]["diffusion_model"], torch_dtype=torch.float16, local_files_only=True
    ).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    pipe.unet.enable_gradient_checkpointing()
    pipe.unet.train()
    categories = collect_category_vocab(samples)
    adapter = FlorenceFeatureAdapter(
        c["paths"]["vlm_model"],
        image_size=768,
        use_fp16=True,
        categories=categories,
        tau=cfg.tau,
        device=args.device,
    )
    for sample in selected:
        image_id = sample["image_id"]
        prior = records.get(image_id, {})
        adv_path = args.output / "adv_images" / f"{image_id}.png"
        clean_path = args.output / f"{image_id}_clean.png"
        if prior.get("status") == "ok" and adv_path.is_file() and clean_path.is_file():
            print("Skipping completed sample", image_id, flush=True)
            continue
        image = Image.open(sample["image_path"]).convert("RGB")
        image.resize((768, 768), resample=Image.LANCZOS).save(clean_path)
        box = build_box_bundle(sample["meta"], canvas_size=768)["box_xyxy_canvas"]
        try:
            adv, generated, mask, info = attack(
                pipe, adapter, image, sample["meta"], box, categories, cfg
            )
            save_image(adv_path, adv)
            save_image(args.output / "debug" / f"{image_id}_gen.png", generated)
            save_image(args.output / "debug" / f"{image_id}_mask.png", mask)
            records[image_id] = dict(
                image_id=image_id,
                clean_path=str(clean_path.resolve()),
                adv_path=str(adv_path.resolve()),
                **info,
            )
            save_records(args.output, records)
            print(
                f"Completed {len([r for r in records.values() if r['status'] == 'ok'])}/{len(ids)}: {image_id}",
                flush=True,
            )
            del adv, generated, mask
            torch.cuda.empty_cache()
        except Exception as exc:
            records[image_id] = dict(
                image_id=image_id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
            save_records(args.output, records)
            raise
    save_json(
        args.output / "run_summary.json",
        dict(
            num_samples=len(ids),
            status_counts={"ok": len(ids), "failed": 0},
            elapsed_seconds=time.time() - started,
        ),
    )


def evaluate(args, c, selected, ids):
    from discta.evaluation import evaluate_crossvlad_metrics
    from discta.evaluation import compute_run_preservation

    records = []
    for sample in selected:
        i = sample["image_id"]
        clean = args.output / f"{i}_clean.png"
        adv = args.output / "adv_images" / f"{i}.png"
        for path in [clean, adv, args.output / "debug" / f"{i}_mask.png"]:
            if not path.is_file():
                raise FileNotFoundError(path)
        records.append(
            dict(id=i, clean_path=str(clean), adv_path=str(adv), meta=sample["meta"])
        )
    cfg = SimpleNamespace(
        florence_model_path=c["paths"]["vlm_model"],
        eval_device=args.device,
        save_dir=str(args.output),
        res=768,
        theta_box=c["evaluation"]["theta_box"],
        theta_loc=c["evaluation"]["theta_loc"],
        metrics_output="crossvlad_metrics.json",
    )
    start = time.time()
    metrics = evaluate_crossvlad_metrics(records, cfg)
    os.environ["DIFFATTACK_METRIC_DEVICE"] = args.device
    quality = compute_run_preservation(args.output)
    save_json(
        args.output / "all_metrics.json",
        dict(attack=metrics["summary"], preservation=quality),
    )
    save_json(
        args.output / "eval_summary.json",
        dict(num_samples=len(ids), elapsed_seconds=time.time() - start),
    )


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    c, samples, selected, ids = prepare(args)
    if args.check_config:
        print(f"Configuration valid; selected {len(ids)} images; no model loaded.")
        return
    if args.mode == "attack":
        generate(args, c, samples, selected, ids)
    else:
        evaluate(args, c, selected, ids)


if __name__ == "__main__":
    main()
