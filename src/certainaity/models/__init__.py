"""Deep learning model interfaces and registry."""

from __future__ import annotations

from certainaity.models.base import ForensicModel, ModelName
from certainaity.models.ensemble import Ensemble, LocalizationResult
from certainaity.models.gandec import GANDetector
from certainaity.models.inpainting import InpaintingDetector
from certainaity.models.mantranet import MantraNet
from certainaity.models.patchforensic import PatchForensic
from certainaity.models.spsl import SPSL

__all__ = [
    "Ensemble",
    "ForensicModel",
    "GANDetector",
    "InpaintingDetector",
    "LocalizationResult",
    "MantraNet",
    "ModelName",
    "PatchForensic",
    "SPSL",
]
