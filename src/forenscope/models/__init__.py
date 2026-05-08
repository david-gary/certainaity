"""Deep learning model interfaces and registry."""

from __future__ import annotations

from forenscope.models.base import ForensicModel, ModelName
from forenscope.models.ensemble import Ensemble, LocalizationResult
from forenscope.models.gandec import GANDetector
from forenscope.models.inpainting import InpaintingDetector
from forenscope.models.mantranet import MantraNet
from forenscope.models.patchforensic import PatchForensic
from forenscope.models.spsl import SPSL

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
