"""Standard image metrics used by HO-LoRA; these metrics are not method innovations.

Images are RGB PIL images or paths to RGB images. ``evaluate_images`` pairs each
variant with ``full`` under the same prompt key. Supply ``VGGGram()`` explicitly
to include pretrained feature distances; pixel-only evaluation loads no model.
"""

from collections.abc import Mapping
import math
from numbers import Real
from os import PathLike

import cv2
import numpy as np
from PIL import Image
import torch


def _image(value):
    if isinstance(value, (str, PathLike)):
        with Image.open(value) as opened:
            value = opened.copy()
    if not isinstance(value, Image.Image):
        raise TypeError("Expected an RGB PIL image or an image path")
    if value.mode != "RGB" or min(value.size) <= 0:
        raise ValueError("Images must be nonempty RGB images; convert their mode explicitly")
    return value


def _pair(reference, image):
    reference, image = _image(reference), _image(image)
    if reference.size != image.size:
        raise ValueError(f"Image sizes differ: {reference.size} versus {image.size}")
    return reference, image


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Metric must be a finite real number, received {value!r}")
    return float(value)


def color_histogram_distance(reference, image):
    """Asymmetric HSV chi-square distance: reference first, L2-normalized 16³ bins."""
    histograms = []
    for item in _pair(reference, image):
        hsv = cv2.cvtColor(np.asarray(item), cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16],
                                [0, 180, 0, 256, 0, 256])
        histograms.append(cv2.normalize(histogram, histogram).flatten())
    return _finite(cv2.compareHist(*histograms, cv2.HISTCMP_CHISQR))


def edge_iou(reference, image):
    """Canny(50, 150) edge overlap; an empty union has IoU 1."""
    edges = []
    for item in _pair(reference, image):
        gray = cv2.cvtColor(np.asarray(item), cv2.COLOR_RGB2GRAY)
        edges.append(cv2.Canny(gray, 50, 150) > 0)
    union = (edges[0] | edges[1]).sum()
    return float((edges[0] & edges[1]).sum() / union) if union else 1.0


def gram_matrix(features):
    """B×C×H×W features → B×C×C Gram matrices, normalized by C*H*W."""
    if not isinstance(features, torch.Tensor):
        raise TypeError("Features must be a torch.Tensor")
    if features.ndim != 4 or min(features.shape) <= 0 or not features.is_floating_point():
        raise ValueError("Features must be a nonempty floating B×C×H×W tensor")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain nonfinite values")
    _, channels, height, width = features.shape
    flattened = features.flatten(2)
    gram = torch.bmm(flattened, flattened.transpose(1, 2)) / (channels * height * width)
    if not torch.isfinite(gram).all():
        raise ValueError("Gram matrix contains nonfinite values")
    return gram


class VGGGram:
    """Pretrained VGG19 Gram distance at the four historical max-pool endpoints.

    Construction explicitly loads torchvision IMAGENET1K_V1 weights, downloading
    them if not cached. Any loading failure is fatal; random weights are never used.
    """

    layer_indices = (4, 9, 18, 27)

    def __init__(self, device="cpu"):
        from torchvision import models, transforms

        self.device = torch.device(device)
        try:
            # Torchvision's official weights loader uses check_hash=True.
            features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception as exc:
            raise RuntimeError("Pretrained VGG19 IMAGENET1K_V1 weights are required; "
                               "download/cache the official weights and retry") from exc
        self.features = features[:28].to(self.device).eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _extract(self, image):
        features = self.transform(image).unsqueeze(0).to(self.device)
        grams = []
        for index, layer in enumerate(self.features):
            features = layer(features)
            if index in self.layer_indices:
                grams.append(gram_matrix(features))
        return grams

    def distance(self, reference, image):
        """Mean Frobenius distance across the four normalized Gram matrices."""
        reference, image = _pair(reference, image)
        left, right = self._extract(reference), self._extract(image)
        return _finite(np.mean([torch.norm(a - b, p="fro").item()
                                for a, b in zip(left, right)]))


def evaluate_images(images_by_prompt, *, gram=None):
    """Evaluate ``{prompt_id: {variant: RGB_image_or_path}}`` against each ``full``.

    Every prompt must contain the same variants, including ``full`` and at least
    one alternative. Images must share a size. Identity/prompt labels are supplied
    by the caller; no filename parsing or image generation is performed here.
    ``gram=None`` computes only color/edge metrics; pass ``VGGGram(device)`` for all
    three. Returns per-prompt scores and per-variant mean/population-std/count.
    """
    if not isinstance(images_by_prompt, Mapping) or not images_by_prompt:
        raise ValueError("Provide a nonempty mapping of prompt IDs to image variants")
    expected, size, prepared = None, None, {}
    for prompt, variants in images_by_prompt.items():
        if not isinstance(prompt, str) or not prompt or not isinstance(variants, Mapping):
            raise ValueError("Each prompt must be a nonempty string mapped to image variants")
        if any(not isinstance(name, str) or not name for name in variants):
            raise ValueError("Variant names must be nonempty strings")
        names = set(variants)
        if "full" not in names or len(names) < 2:
            raise ValueError(f"{prompt}: include full and at least one alternative variant")
        if expected is None:
            expected = names
        if names != expected:
            raise ValueError(f"{prompt}: variant set differs; missing={sorted(expected - names)}, "
                             f"unexpected={sorted(names - expected)}")
        prepared[prompt] = {}
        for name, value in variants.items():
            image = _image(value)
            size = image.size if size is None else size
            if image.size != size:
                raise ValueError(f"{prompt}/{name}: image size {image.size} differs from {size}")
            prepared[prompt][name] = image

    per_prompt, grouped = {}, {name: {} for name in sorted(expected - {"full"})}
    for prompt, variants in prepared.items():
        per_prompt[prompt] = {}
        for name in grouped:
            reference, image = variants["full"], variants[name]
            metrics = {"color_hist_dist": color_histogram_distance(reference, image),
                       "edge_iou": edge_iou(reference, image)}
            if gram is not None:
                metrics["gram_dist"] = _finite(gram.distance(reference, image))
            per_prompt[prompt][name] = metrics
            for metric, value in metrics.items():
                grouped[name].setdefault(metric, []).append(value)
    per_variant = {name: {metric: {"mean": _finite(np.mean(values)),
                                  "std": _finite(np.std(values)), "count": len(values)}
                          for metric, values in scores.items()}
                   for name, scores in grouped.items()}
    return {"per_prompt": per_prompt, "per_variant": per_variant}


def texture_delta(per_variant):
    """ΔT = mean Gram(no_lora, full) − mean Gram(texture_only, full).

    Accepts evaluate_images(...)["per_variant"] or the same field from a recorded
    JSON. Both variants must have Gram means. This is a full-reference similarity
    difference, not an absolute image-quality or semantic-independence score.
    """
    try:
        baseline = _finite(per_variant["no_lora"]["gram_dist"]["mean"])
        texture = _finite(per_variant["texture_only"]["gram_dist"]["mean"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Delta T requires no_lora and texture_only Gram means") from exc
    return _finite(baseline - texture)
