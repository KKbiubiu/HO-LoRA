"""HO-LoRA's SDXL branches and four-phase training rule, using PyTorch/torchvision.

Prepare an adapter-free SDXL U-Net on its intended device/dtype before wrapping
it in HOLoRA. The caller supplies a frozen fp32 VAE, a DDPM-compatible scheduler,
RGB batches in [-1, 1], and detached per-sample SDXL prompt embeddings plus
added_cond_kwargs (text_embeds and time_ids), on the model's device.

Create AdamW from model.parameter_groups(), then call training_step() for each
batch. That function performs backward and an optimizer update while the branch
switches remain active, including during gradient-checkpoint recomputation.
This module does not load models, tokenize text, export Diffusers adapters,
provide AMP/GradScaler, accumulate gradients, or coordinate distributed workers.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


BRANCH_BLOCKS = {
    "content": ("up_blocks.0.attentions.0",),
    "color": ("up_blocks.0.attentions.1",),
    "texture": (
        "up_blocks.0.attentions.2",  # 32x32 at 1024px: same width as content/color.
        "up_blocks.1.attentions.0",  # 64x64: finer texture, without cross-Gram pairing.
        "up_blocks.1.attentions.1",
        "up_blocks.1.attentions.2",
    ),
}
PHASE_BRANCHES = (
    frozenset(("content", "color", "texture")),
    frozenset(("content", "texture")),
    frozenset(("content", "color")),
    frozenset(("texture",)),
)


def branch_of(name):
    """Match a complete module-path prefix, not e.g. attentions.1 to .10."""
    return next((branch for branch, blocks in BRANCH_BLOCKS.items()
                 if any(name == block or name.startswith(block + ".") for block in blocks)), None)


class LoRAProjection(nn.Module):
    """Frozen Linear(x) + scale * up(down(x)); no alpha/rank rescaling."""

    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base, nn.Linear) or getattr(base, "lora_layer", None) is not None:
            raise ValueError("Expected an adapter-free torch.nn.Linear projection")
        if not isinstance(rank, int) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError("Rank must be positive and no larger than the projection dimensions")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.down = nn.Linear(self.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, self.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x, scale=1.0):
        output = self.base(x)
        if self.enabled:
            update = self.up(self.down(x.to(self.down.weight.dtype)))
            output = output + scale * update.to(output.dtype)
        return output


class HOLoRA(nn.Module):
    """Inject Q/K/V/output adapters into SDXL's six selected attention blocks.

    Injection mutates and freezes the supplied U-Net. Move it to its target
    device/dtype first; do not subsequently cast the fp32 trainable adapters.
    Only ordinary dense attention projections without other adapters are supported.
    """

    def __init__(self, unet, *, content_rank=64, color_rank=32, texture_rank=128):
        super().__init__()
        ranks = dict(content=content_rank, color=color_rank, texture=texture_rank)
        targets, found = [], set()
        for name, attention in list(unet.named_modules()):
            branch = branch_of(name)
            if branch is None or not all(hasattr(attention, key) for key in ("to_q", "to_k", "to_v", "to_out")):
                continue
            for key in ("to_q", "to_k", "to_v", "to_out.0"):
                projection = attention.get_submodule(key)
                rank = ranks[branch]
                if (not isinstance(projection, nn.Linear) or getattr(projection, "lora_layer", None) is not None
                        or not isinstance(rank, int) or not 0 < rank <= min(projection.in_features, projection.out_features)):
                    raise ValueError(f"Unsupported projection or rank: {name}.{key}")
                targets.append((branch, name + "." + key, attention, key, projection))
            found.update(block for block in BRANCH_BLOCKS[branch] if name.startswith(block + ".") or name == block)
        expected = {block for blocks in BRANCH_BLOCKS.values() for block in blocks}
        if found != expected:
            raise ValueError(f"Missing SDXL attention blocks: {sorted(expected - found)}")
        self.unet = unet.requires_grad_(False)
        self.adapters = {branch: {} for branch in BRANCH_BLOCKS}
        for branch, name, attention, key, projection in targets:
            adapter = LoRAProjection(projection, ranks[branch])
            if key == "to_out.0":
                attention.to_out[0] = adapter
            else:
                setattr(attention, key, adapter)
            self.adapters[branch][name] = adapter

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)

    def parameter_groups(self, base_lr=5e-5, boost=2.0):
        """Color uses base_lr; content uses base_lr*boost; texture uses 1.5x that."""
        multipliers = dict(content=boost, color=1.0, texture=boost * 1.5)
        return [{"params": [p for adapter in adapters.values() for p in
                            (adapter.down.weight, adapter.up.weight)],
                 "lr": base_lr * multipliers[branch]}
                for branch, adapters in self.adapters.items()]

    @contextmanager
    def phase(self, index):
        """Keep this context open through backward when using checkpointing."""
        if index not in range(4):
            raise ValueError("Phase must be 0, 1, 2 or 3")
        previous = [(adapter, adapter.enabled) for group in self.adapters.values() for adapter in group.values()]
        try:
            for branch, group in self.adapters.items():
                for adapter in group.values():
                    adapter.enabled = branch in PHASE_BRANCHES[index]
            yield
        finally:
            for adapter, enabled in previous:
                adapter.enabled = enabled

    def orthogonal_loss(self, ct_weight=0.5):
        """Mean squared cross-Gram of row-normalized down weights.

        Match identical projection suffixes in content/color and content/the
        first texture block. The three 64x64 texture blocks are not paired.
        Different branch ranks are supported; input feature widths must match.
        """
        if not math.isfinite(ct_weight) or ct_weight < 0:
            raise ValueError("ct_weight must be finite and nonnegative")

        def downs(branch, block):
            return {name[len(block):]: adapter.down.weight
                    for name, adapter in self.adapters[branch].items() if name.startswith(block + ".")}

        content = downs("content", BRANCH_BLOCKS["content"][0])
        loss = next(iter(content.values())).new_zeros(())
        for branch, weight in (("color", 1.0), ("texture", ct_weight)):
            other = downs(branch, BRANCH_BLOCKS[branch][0])
            if content.keys() != other.keys():
                raise ValueError("Paired blocks must have matching attention projection paths")
            terms = [(F.normalize(a, dim=1) @ F.normalize(other[key], dim=1).T).square().mean()
                     for key, a in content.items()]
            loss = loss + weight * torch.stack(terms).mean()
        return loss


def transform_images(images, phase, *, color_jitter=None, blur=None, noise_std=0.15):
    """Full/solo: original; phase 1: jitter; phase 2: blur plus clipped noise."""
    if phase not in range(4) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected a phase in [0, 3] and RGB images shaped [B, 3, H, W]")
    if phase in (0, 3):
        return images
    pixels = ((images + 1) / 2).clamp(0, 1)
    if phase == 1:
        jitter = color_jitter if color_jitter is not None else transforms.ColorJitter(0.4, 0.4, 0.7, 0.35)
        pixels = torch.stack([jitter(image) for image in pixels])
    else:
        blur = blur if blur is not None else transforms.GaussianBlur(25, (5.0, 12.0))
        pixels = torch.stack([blur(image) for image in pixels])
        pixels = (pixels + torch.randn_like(pixels) * noise_std).clamp(0, 1)
    return pixels * 2 - 1


def training_step(model, vae, scheduler, optimizer, images, prompt_embeds, added_cond_kwargs,
                  *, step, ortho_weight=0.15, ortho_warmup_steps=200, ct_weight=0.5,
                  snr_gamma=5.0, noise_offset=0.05, color_jitter=None, blur=None, texture_noise_std=0.15):
    """Execute one complete update and return detached losses plus phase index.

    Call with optimizer built from model.parameter_groups(). Each call clears
    gradients; this is a single-process step, without gradient accumulation or
    AMP scaling. The caller controls RNG seeding and any learning-rate scheduler.
    VAE.encode(...).latent_dist.sample(), VAE.config.scaling_factor, and the
    scheduler's add_noise/config/alphas_cumprod API follow SDXL/DDPM conventions.
    """
    if not isinstance(step, int) or step < 0 or ortho_warmup_steps < 0:
        raise ValueError("step and ortho_warmup_steps must be nonnegative")
    phase = step % 4
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with model.phase(phase):
        images = transform_images(images, phase, color_jitter=color_jitter, blur=blur, noise_std=texture_noise_std)
        with torch.no_grad():
            latents = vae.encode(images.float()).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.to(dtype=next(model.unet.parameters()).dtype)
        noise = torch.randn_like(latents)
        if noise_offset:
            noise = noise + noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=latents.device, dtype=latents.dtype)
        timesteps = torch.randint(scheduler.config.num_train_timesteps, (len(latents),), device=latents.device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        prediction = model(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError("Only epsilon and v_prediction targets are supported")
        per_image = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
        if snr_gamma is not None and snr_gamma > 0:
            alpha = scheduler.alphas_cumprod.to(latents.device)[timesteps]
            snr = alpha / (1 - alpha)
            denominator = snr + 1 if scheduler.config.prediction_type == "v_prediction" else snr
            per_image = per_image * snr.clamp(max=snr_gamma) / denominator
        reconstruction = per_image.mean()
        orthogonal = reconstruction.new_zeros(())
        if phase == 0:
            warmup = min(1.0, step / ortho_warmup_steps) if ortho_warmup_steps else 1.0
            orthogonal = model.orthogonal_loss(ct_weight) * ortho_weight * warmup
        loss = reconstruction + orthogonal
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; adapter update skipped")
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {"loss": loss.detach(), "reconstruction": reconstruction.detach(),
            "orthogonal": orthogonal.detach(), "phase": phase}
