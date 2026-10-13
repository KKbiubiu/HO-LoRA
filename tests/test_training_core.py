"""Small CPU checks for the training rule; no Diffusers models or downloads."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training import BRANCH_BLOCKS, HOLoRA, LoRAProjection, branch_of, training_step, transform_images


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = (nn.Linear(4, 4) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(4, 4)])
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out[0]):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.to_out[0](torch.tanh((self.to_q(x) + self.to_k(x) + self.to_v(x)) / 3))


class TinyUNet(nn.Module):
    def __init__(self, checkpointing=False):
        super().__init__()
        self.up_blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for block in self.up_blocks:
            block.attentions = nn.ModuleList([Attention() for _ in range(3)])
        self.checkpointing, self.fail = checkpointing, False
        self.observed = []

    def forward(self, sample, timesteps, prompt_embeds, added_cond_kwargs):
        if self.fail:
            raise RuntimeError("unet failed")
        x = sample.permute(0, 2, 3, 1)

        def attend(value, attention):
            self.observed.append(tuple(module.enabled for module in self.modules() if isinstance(module, LoRAProjection)))
            return attention(value)

        for block in self.up_blocks:
            for attention in block.attentions:
                if self.checkpointing:
                    x = checkpoint(attend, x, attention, use_reentrant=False)
                else:
                    x = attend(x, attention)
        return SimpleNamespace(sample=x.permute(0, 3, 1, 2))


class TinyVAE:
    config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, images):
        latent = torch.cat((images, images.mean(dim=1, keepdim=True)), dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))


class TinyScheduler:
    config = SimpleNamespace(num_train_timesteps=8, prediction_type="epsilon")
    alphas_cumprod = torch.linspace(0.95, 0.1, 8)

    def add_noise(self, latent, noise, timesteps):
        alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * latent + (1 - alpha).sqrt() * noise


class TrainingCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def model(self, checkpointing=False):
        return HOLoRA(TinyUNet(checkpointing), content_rank=1, color_rank=1, texture_rank=1)

    def run_step(self, model, optimizer, step):
        return training_step(
            model, TinyVAE(), TinyScheduler(), optimizer,
            torch.rand(1, 3, 16, 16) * 2 - 1, torch.zeros(1, 2, 4),
            {"text_embeds": torch.zeros(1, 4), "time_ids": torch.zeros(1, 6)},
            step=step, ortho_warmup_steps=0,
            color_jitter=lambda image: image.flip(0), blur=lambda image: image.mean().expand_as(image),
        )

    def test_branch_paths_and_complete_injection(self):
        self.assertEqual(branch_of("up_blocks.0.attentions.0.transformer_blocks.0.attn1"), "content")
        self.assertEqual(branch_of("up_blocks.0.attentions.1.transformer_blocks.0.attn2"), "color")
        for block in BRANCH_BLOCKS["texture"]:
            self.assertEqual(branch_of(block + ".transformer_blocks.0.attn1"), "texture")
        self.assertIsNone(branch_of("up_blocks.0.attentions.10.transformer_blocks.0.attn1"))
        self.assertIsNone(branch_of("down_blocks.0.attentions.0"))
        model = self.model()
        self.assertEqual({branch: len(group) for branch, group in model.adapters.items()},
                         {"content": 4, "color": 4, "texture": 16})
        with self.assertRaises(ValueError):
            HOLoRA(model.unet, content_rank=1, color_rank=1, texture_rank=1)

    def test_invalid_topology_does_not_partially_mutate_model(self):
        unet = TinyUNet()
        del unet.up_blocks[1].attentions[2]
        original = list(unet.parameters())
        with self.assertRaisesRegex(ValueError, "Missing SDXL"):
            HOLoRA(unet, content_rank=1, color_rank=1, texture_rank=1)
        self.assertTrue(all(parameter.requires_grad for parameter in original))
        self.assertFalse(any(isinstance(module, LoRAProjection) for module in unet.modules()))

    def test_adapter_matches_merged_linear_formula_and_freezes_base(self):
        base = nn.Linear(4, 5)
        adapter = LoRAProjection(base, rank=2)
        x = torch.randn(2, 3, 4)
        self.assertTrue(torch.equal(adapter(x), base(x)))
        with torch.no_grad():
            adapter.up.weight.normal_()
        scale = 0.7
        merged = base.weight + scale * (adapter.up.weight @ adapter.down.weight)
        torch.testing.assert_close(adapter(x, scale=scale), F.linear(x, merged, base.bias))
        adapter(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        self.assertGreater(adapter.down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0)

    def test_all_four_steps_route_gradients_with_checkpoint_recomputation(self):
        model = self.model(checkpointing=True)
        optimizer = torch.optim.AdamW(model.parameter_groups(base_lr=0.01))
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        expected = ({"content", "color", "texture"}, {"content", "texture"},
                    {"content", "color"}, {"texture"})
        for phase, active in enumerate(expected):
            before = {branch: [adapter.up.weight.detach().clone() for adapter in group.values()]
                      for branch, group in model.adapters.items()}
            model.unet.observed.clear()
            result = self.run_step(model, optimizer, phase)
            self.assertEqual(result["phase"], phase)
            self.assertTrue(torch.isfinite(result["loss"]))
            self.assertFalse(result["loss"].requires_grad)
            expected_switches = tuple(branch_of(name) in active for name, module in model.unet.named_modules()
                                      if isinstance(module, LoRAProjection))
            self.assertTrue(model.unet.observed)
            self.assertTrue(all(switches == expected_switches for switches in model.unet.observed))
            for branch, group in model.adapters.items():
                changed = any(not torch.equal(old, adapter.up.weight)
                              for old, adapter in zip(before[branch], group.values()))
                self.assertEqual(changed, branch in active)
                if branch not in active:
                    self.assertTrue(all(parameter.grad is None for adapter in group.values()
                                        for parameter in (adapter.down.weight, adapter.up.weight)))
            self.assertTrue(all(adapter.enabled for group in model.adapters.values() for adapter in group.values()))
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, frozen[name]))
                self.assertIsNone(parameter.grad)

    def test_failure_restores_switches_without_an_optimizer_update(self):
        model = self.model()
        next(iter(model.adapters["content"].values())).enabled = False
        switches = [adapter.enabled for group in model.adapters.values() for adapter in group.values()]
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        model.unet.fail = True
        with self.assertRaisesRegex(RuntimeError, "unet failed"):
            self.run_step(model, torch.optim.AdamW(model.parameter_groups()), 3)
        self.assertEqual(switches, [adapter.enabled for group in model.adapters.values() for adapter in group.values()])
        self.assertTrue(all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()))

    def test_cross_gram_pairs_only_the_compatible_texture_block_without_mutation(self):
        model = HOLoRA(TinyUNet(), content_rank=1, color_rank=2, texture_rank=3)
        with torch.no_grad():
            for branch, group in model.adapters.items():
                row = {"content": [1., 0., 0., 0.], "color": [0., 1., 0., 0.], "texture": [1., 1., 0., 0.]}[branch]
                for adapter in group.values():
                    adapter.down.weight.copy_(torch.tensor([row]))
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loss = model.orthogonal_loss(ct_weight=0.4)
        self.assertAlmostEqual(loss.item(), 0.2, places=6)
        self.assertTrue(all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()))
        loss.backward()
        for name, adapter in model.adapters["texture"].items():
            self.assertEqual(adapter.down.weight.grad is None, name.startswith("up_blocks.1."))
        with torch.no_grad():
            for name, adapter in model.adapters["texture"].items():
                if name.startswith("up_blocks.1."):
                    adapter.down.weight.fill_(100)
        self.assertAlmostEqual(model.orthogonal_loss(ct_weight=0.4).item(), 0.2, places=6)
        with self.assertRaises(ValueError):
            model.orthogonal_loss(ct_weight=-1)

    def test_nonfinite_loss_preserves_parameters_and_restores_switches(self):
        model = self.model()
        next(iter(model.adapters["color"].values())).enabled = False
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        switches = [adapter.enabled for group in model.adapters.values() for adapter in group.values()]
        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            training_step(
                model, TinyVAE(), TinyScheduler(), torch.optim.AdamW(model.parameter_groups()),
                torch.full((1, 3, 16, 16), float("nan")), torch.zeros(1, 2, 4), {}, step=3,
            )
        self.assertTrue(all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()))
        self.assertEqual(switches, [adapter.enabled for group in model.adapters.values() for adapter in group.values()])

    def test_image_transforms_preserve_input_and_full_solo_identity(self):
        images = torch.rand(2, 3, 16, 16) * 2 - 1
        before = images.clone()
        for phase in range(4):
            result = transform_images(images, phase)
            self.assertEqual(result.shape, images.shape)
            self.assertTrue((result >= -1).all() and (result <= 1).all())
            self.assertEqual(torch.equal(result, images), phase in (0, 3))
            self.assertTrue(torch.equal(images, before))


if __name__ == "__main__":
    unittest.main()
