"""Metric/formula checks with synthetic images; no pretrained downloads."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from core.evaluation import (
    VGGGram, color_histogram_distance, edge_iou, evaluate_images, gram_matrix, texture_delta,
)


def solid(value, size=(16, 16)):
    return Image.new("RGB", size, (value, value, value))


class EvaluationCoreTests(unittest.TestCase):
    def test_pixel_metrics_follow_histogram_and_edge_formulas(self):
        reference = np.zeros((8, 12, 3), dtype=np.uint8)
        reference[:, :8, 0], reference[:, 8:, 2] = 255, 255
        red = Image.new("RGB", (12, 8), "red")
        mixed = Image.fromarray(reference)
        a, b = 2 / math.sqrt(5), 1 / math.sqrt(5)
        self.assertAlmostEqual(color_histogram_distance(mixed, red), (a - 1) ** 2 / a + b, places=6)
        self.assertAlmostEqual(color_histogram_distance(red, mixed), (1 - a) ** 2, places=6)
        self.assertEqual(color_histogram_distance(red, red), 0)
        vertical, horizontal = np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16, 3), np.uint8)
        vertical[:, 8:], horizontal[8:, :] = 255, 255
        self.assertAlmostEqual(edge_iou(Image.fromarray(vertical), Image.fromarray(horizontal)), 1 / 31)
        self.assertEqual(edge_iou(solid(0), solid(255)), 1)

    def test_gram_matrix_normalization_and_invalid_features(self):
        features = torch.tensor([[[[1., 2.]], [[3., 4.]]]])
        expected = torch.tensor([[[1.25, 2.75], [2.75, 6.25]]])
        torch.testing.assert_close(gram_matrix(features), expected, rtol=0, atol=0)
        for invalid in (torch.ones(2, 2), torch.ones(1, 2, 0, 2), torch.ones(1, 1, 1, 1, dtype=torch.int64),
                        torch.full((1, 1, 1, 1), float("nan")), torch.full((1, 1, 1, 1), float("inf")),
                        torch.full((1, 1, 1, 1), 1e38)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                gram_matrix(invalid)

    def test_gram_distance_is_mean_frobenius_distance(self):
        scorer = VGGGram.__new__(VGGGram)
        scorer._extract = Mock(side_effect=[
            [torch.zeros(1, 2, 2) for _ in range(4)],
            [torch.ones(1, 2, 2) * scale for scale in (1, 2, 3, 4)],
        ])
        self.assertEqual(scorer.distance(solid(0), solid(1)), 5)

    def test_missing_pretrained_weights_fail_without_random_fallback(self):
        from torchvision import models
        with patch.object(models, "vgg19", side_effect=OSError("not cached")) as constructor:
            with self.assertRaisesRegex(RuntimeError, "Pretrained VGG19 IMAGENET1K_V1 weights are required"):
                VGGGram()
        constructor.assert_called_once_with(weights=models.VGG19_Weights.IMAGENET1K_V1)

    def test_same_prompt_reference_aggregation_and_texture_delta(self):
        images = {"p00": {"full": solid(0), "no_lora": solid(2), "texture_only": solid(1)},
                  "p01": {"full": solid(10), "no_lora": solid(14), "texture_only": solid(11)}}
        gram = SimpleNamespace(distance=lambda reference, image: image.getpixel((0, 0))[0] - reference.getpixel((0, 0))[0])
        result = evaluate_images(images, gram=gram)
        self.assertEqual(result["per_prompt"]["p01"]["no_lora"]["gram_dist"], 4)
        self.assertEqual(result["per_variant"]["no_lora"]["gram_dist"], {"mean": 3., "std": 1., "count": 2})
        self.assertEqual(texture_delta(result["per_variant"]), 2)
        with self.assertRaisesRegex(ValueError, "requires no_lora and texture_only"):
            texture_delta({})
        result["per_variant"]["no_lora"]["gram_dist"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            texture_delta(result["per_variant"])

    def test_incomplete_pairings_and_incompatible_images_are_rejected(self):
        invalid_sets = [
            {}, {"p00": {"no_lora": solid(0)}},
            {"p00": {"full": solid(0), "no_lora": solid(0)},
             "p01": {"full": solid(0), "texture_only": solid(0)}},
            {"p00": {"full": solid(0), "no_lora": solid(0, (8, 8))}},
            {"p00": {"full": solid(0), "no_lora": Image.new("F", (16, 16), float("nan"))}},
        ]
        for images in invalid_sets:
            with self.subTest(prompts=list(images)), self.assertRaises(ValueError):
                evaluate_images(images)
        for metric in (color_histogram_distance, edge_iou):
            with self.assertRaisesRegex(ValueError, "sizes differ"):
                metric(solid(0), solid(0, (8, 8)))
        with self.assertRaises(TypeError):
            color_histogram_distance(np.zeros((16, 16, 3)), solid(0))

    def test_pixel_only_path_input_never_loads_vgg_and_nonfinite_scores_fail(self):
        from torchvision import models
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            solid(10).save(path)
            images = {"p00": {"full": path, "no_lora": path}}
            with patch.object(models, "vgg19") as constructor:
                result = evaluate_images(images)
            constructor.assert_not_called()
            self.assertEqual(set(result["per_variant"]["no_lora"]), {"color_hist_dist", "edge_iou"})
            with self.assertRaisesRegex(ValueError, "finite"):
                evaluate_images(images, gram=SimpleNamespace(distance=lambda a, b: float("nan")))


if __name__ == "__main__":
    unittest.main()
