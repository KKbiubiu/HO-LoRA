# Core implementation

The public implementation isolates the method in two modules. It is intended for integration into an existing SDXL application. Model loading, tokenization, dataset management, checkpoint export and image generation are outside this module API.

## Installation and verification

Use Python 3.10 or 3.11 and the pinned versions in [`requirements-core.txt`](../requirements-core.txt). Follow the [CPU installation commands](../README.md#core-implementation), then run:

```bash
python -m unittest discover -s tests -p 'test_*_core.py' -v
```

These tests exercise adapter mathematics, phase-specific gradients, checkpoint recomputation, metric definitions and invalid-input handling. They do not download models or establish full-size SDXL convergence.

## Training

[`core/training.py`](../core/training.py) contains the three branch definitions, a frozen linear projection with trainable LoRA factors, phase-specific image transforms, cross-Gram regularization and one complete optimization step.

Prepare the following in the calling application:

- An adapter-free SDXL U-Net with ordinary dense Q/K/V/output projections, already on the target device and dtype.
- A frozen VAE in float32 and a DDPM-compatible training noise scheduler.
- RGB image batches shaped `B × 3 × H × W`, with values in `[-1, 1]`, on the model device.
- Detached, per-sample SDXL prompt embeddings, pooled text embeddings and time IDs, with matching batch dimensions and device/dtype.

The following fragment runs inside that application; the model components and batch tensors are supplied by the caller:

```python
import torch
from core.training import HOLoRA, training_step

model = HOLoRA(unet, content_rank=64, color_rank=32, texture_rank=128)
optimizer = torch.optim.AdamW(model.parameter_groups(base_lr=5e-5, boost=2.0))

metrics = training_step(
    model, vae, noise_scheduler, optimizer,
    images, prompt_embeds,
    {"text_embeds": pooled_text_embeds, "time_ids": time_ids},
    step=global_step,
    ortho_weight=0.15,
    ortho_warmup_steps=200,
    ct_weight=0.5,
)
```

`step % 4` selects joint reconstruction, color jitter without Color, texture destruction without Texture, or Texture-only reconstruction. The joint phase adds the orthogonal term. Noise offset, Min-SNR reconstruction weighting and gradient clipping follow the maintained training rule.

Each call clears gradients, performs backward and updates the optimizer. Branch switches remain active through checkpoint recomputation and are restored on exit. Supply the random seed and any learning-rate scheduler in the calling application. The step is single-process; it does not implement gradient accumulation or AMP/GradScaler.

The adapter update is `W(x) + up(down(x))`, with float32 trainable factors and no additional alpha/rank scaling. Do not cast the complete wrapped model to half precision: that would also cast the trainable factors. The wrapper uses its own state-dict layout and is not an adapter-file importer/exporter.

The cross-Gram term pairs matching attention projection paths for Content–Color and Content–Texture. Only Texture's first, same-width block participates in this term; the other three Texture blocks learn through reconstruction. `ct_weight=0.5` is explicit in this implementation. Historical configuration uncertainty remains documented in [METHOD.md](METHOD.md).

## Evaluation

[`core/evaluation.py`](../core/evaluation.py) provides standard metrics; the metrics themselves are not claimed as HO-LoRA innovations.

The pixel metrics can be exercised directly on a supplied result without loading a model:

```python
from core.evaluation import color_histogram_distance, edge_iou

image = "results/images/result_01.png"
assert color_histogram_distance(image, image) == 0.0
assert edge_iou(image, image) == 1.0
```

For paired evaluation, provide images generated from the same prompt and seed. The following paths are inputs from the calling experiment:

```python
from core.evaluation import VGGGram, evaluate_images, texture_delta

scores = evaluate_images(
    {
        "p00": {
            "full": "experiment/p00_full.png",
            "no_lora": "experiment/p00_no_lora.png",
            "texture_only": "experiment/p00_texture_only.png",
        }
    },
    gram=VGGGram(device="cpu"),
)
delta_t = texture_delta(scores["per_variant"])
```

`VGGGram` loads the official torchvision VGG19 `IMAGENET1K_V1` weights. Its constructor downloads the weights when needed, or uses the PyTorch cache. For offline evaluation, prepare that cache in advance; unavailable pretrained weights cause an error. Omitting `gram` computes only Color Histogram Distance and Edge IoU.

Inputs must be RGB PIL images or paths to RGB files, with equal dimensions. Every prompt must have the same set of variants, including `full` and at least one alternative. Prompt identity is supplied by the caller. The function returns per-prompt metrics and per-variant mean, population standard deviation and count; `texture_delta` requires both `no_lora` and `texture_only` Gram means.

The result gallery contains selected final outputs, not all comparison variants. It does not supply the paired inputs needed to recompute the full historical evaluation. Recorded metrics and their interpretation remain in [EVALUATION.md](EVALUATION.md).
