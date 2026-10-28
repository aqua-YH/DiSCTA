This repository contains the official implementation of the paper “Disrupting Multi-Task Semantic Coordination: A Diffusion-Based Cross-Task Attack for Fooling Unified Vision-Language Models.”

## Requirements

### Hardware Requirements

Attack generation requires an NVIDIA GPU with CUDA support. The experiments used an NVIDIA RTX 4090 with 24 GB VRAM.

### Software Requirements

| Software              | Version        |
| --------------------- | -------------- |
| Python                | 3.9            |
| PyTorch / torchvision | 2.8.0 / 0.23.0 |
| CUDA build            | cu128          |
| Transformers          | 4.57.6         |
| Diffusers             | 0.36.0         |
| Accelerate            | 1.10.1         |

## Preparation

Before running the project, you need to download the model and prepare the dataset.

### 1. Dataset

We use the [CrossVLAD](https://drive.google.com/file/d/1DPmSPSYH6iXtuEPwRKeM_0rojLoFdGQ7/view) data and annotations provided by the official CRAFT project. This benchmark comprises 3,000 images from MSCOCO train2017.
The final structure should look like this:

```text
CrossVLAD/
├── images/
│   ├── <image_id>.jpg
│   └── ...
└── labels/
    ├── <image_id>.json
    └── ...
```

Set dataset paths under `paths` in `configs/default.json`. By default, all images with matching annotations are selected. The `manifests/` directory is empty. To select a subset, create a JSON list of image IDs and pass `--manifest /path/to/ids.json`.

### 2. Pretrained Models

A complete local model directory should be prepared in advance. Stable Diffusion 2.1 requires Diffusers-format components, configurations, and weights, whereas vision-language models (VLMs) such as Florence-2 require the model, processor, tokenizer, and a matching custom Python implementation. The present code uses `local_files_only=True` and therefore does not automatically download models during the attack phase.

Update `paths` in `configs/default.json` to match your local files, for example:

```json
"paths": {
  "diffusion_model": "/path/to/stable-diffusion-2-1",
  "vlm_model": "/path/to/Florence-2-large",
  "images_root": "/path/to/CrossVLAD/images",
  "labels_root": "/path/to/CrossVLAD/labels"
}
```

### 3. Installing Additional Dependencies

```bash
python -m pip install -r requirements.txt
```

LPIPS and FID require additional metric-network weights, which may be downloaded on first use. For offline evaluation, cache the AlexNet/LPIPS and pytorch-fid Inception weights in advance.

## Quick Start

### Running the Attack

**1. Configure paths and attack parameters**

Edit `configs/default.json` to specify the dataset and model paths:

- `paths.images_root`: directory containing the clean images.
- `paths.labels_root`: directory containing the corresponding CrossVLAD annotations.
- `paths.diffusion_model`: local Stable Diffusion 2.1 model directory.
- `paths.vlm_model`: local Florence-2-large model directory.

The main attack parameters are:`diffusion_steps`、`start_step`、`iterations`、`guidance`、`latent_lr`...


**2. Check the configuration**

Verify the configured paths and sample selection before running the attack:

```
python main.py attack \
    --config configs/default.json \
    --output outputs/check \
    --check-config
```

This command does not load model weights or generate adversarial images.

**3. Generate adversarial images**

Run the attack on physical GPU0:

```
CUDA_VISIBLE_DEVICES=0 python main.py attack \
    --config configs/default.json \
    --output outputs/result
```

By default, the attack processes all images with matching annotations. `CUDA_VISIBLE_DEVICES` selects the physical GPU, which appears as `cuda:0` inside the process.

To attack a subset, create a JSON file containing image IDs, such as:

```
["000000000620", "000000001000"]
```

Replace these example IDs with IDs present in your dataset, then pass the file through `--manifest`:

```
CUDA_VISIBLE_DEVICES=0 python main.py attack \
    --config configs/default.json \
    --manifest /path/to/selected_ids.json \
    --output outputs/subset
```

**4. Optional: Generate images on multiple GPUs**

For three GPUs, run each command in a separate terminal:

```
CUDA_VISIBLE_DEVICES=0 python main.py attack \
    --output outputs/shard0 --num-shards 3 --shard-index 0
```

```
CUDA_VISIBLE_DEVICES=1 python main.py attack \
    --output outputs/shard1 --num-shards 3 --shard-index 1
```

```
CUDA_VISIBLE_DEVICES=2 python main.py attack \
    --output outputs/shard2 --num-shards 3 --shard-index 2
```

The sorted sample list is divided into three non-overlapping shards. Once all shards finish, merge their outputs:

```
python merge.py \
    --runs outputs/shard0 outputs/shard1 outputs/shard2 \
    --output outputs/merged
```

For subset experiments, use the same `--manifest` for all attack commands and the merge command.

### Evaluation

Evaluate the generated adversarial images across Image Captioning (IC), Object Detection (OD), Region Categorization (RC), and Object Localization (OL):

```
CUDA_VISIBLE_DEVICES=0 python main.py evaluate \
    --config configs/default.json \
    --output outputs/result
```

Set `--output` to the run directory containing `adv_images/` and `selected_ids.json`. Use the same configuration file as in attack generation.

For merged multi-GPU results, run:

```
CUDA_VISIBLE_DEVICES=0 python main.py evaluate \
    --config configs/default.json \
    --output outputs/merged
```

Evaluation reports per-task targeted attack success rates, CTSR-4 for success on all four tasks, and CTSR-3 for success on at least three tasks. It also computes LPIPS, SSIM, PSNR, MAE, FID, and outside-mask modification.