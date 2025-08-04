"""Unit tests for model stubs: PatchForensic, MantraNet, SPSL, InpaintingDetector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forenscope.exceptions import WeightsNotFoundError
from forenscope.models.base import ModelName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def weights_dir(tmp_path: Path) -> Path:
    """Empty weights directory — no weight files present."""
    d = tmp_path / "weights"
    d.mkdir()
    return d


@pytest.fixture()
def dummy_rgb() -> np.ndarray:
    """(256, 256, 3) uint8 array suitable for model.predict()."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# PatchForensic
# ---------------------------------------------------------------------------

class TestPatchForensic:
    def test_weight_file_constant(self) -> None:
        from forenscope.models.patchforensic import PatchForensic
        assert PatchForensic.WEIGHT_FILE == "patchforensic_v2.pth"

    def test_model_name_constant(self) -> None:
        from forenscope.models.patchforensic import PatchForensic
        assert PatchForensic.MODEL_NAME == ModelName.PATCH_FORENSIC

    def test_instantiates_without_weights(self, weights_dir: Path) -> None:
        from forenscope.models.patchforensic import PatchForensic
        model = PatchForensic(weights_dir)
        assert model is not None

    def test_raises_weights_not_found_on_predict(
        self, weights_dir: Path, dummy_rgb: np.ndarray
    ) -> None:
        from forenscope.models.patchforensic import PatchForensic
        model = PatchForensic(weights_dir)
        with pytest.raises(WeightsNotFoundError):
            model.predict(dummy_rgb)

    def test_weight_path_property(self, weights_dir: Path) -> None:
        from forenscope.models.patchforensic import PatchForensic
        model = PatchForensic(weights_dir)
        assert model.weight_path == weights_dir / "patchforensic_v2.pth"

    def test_architecture_layer_names(self) -> None:
        pytest.importorskip("torch")
        from forenscope.models.patchforensic import _PatchForensicNet
        net = _PatchForensicNet()
        assert hasattr(net, "enc1")
        assert hasattr(net, "enc2")
        assert hasattr(net, "enc3")
        assert hasattr(net, "dec2")
        assert hasattr(net, "dec1")
        assert hasattr(net, "head")

    def test_forward_output_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from forenscope.models.patchforensic import _PatchForensicNet
        net = _PatchForensicNet()
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 1, 64, 64)

    def test_forward_output_in_range(self) -> None:
        torch = pytest.importorskip("torch")
        from forenscope.models.patchforensic import _PatchForensicNet
        rng = np.random.default_rng(1)
        net = _PatchForensicNet()
        x = torch.from_numpy(rng.random((1, 3, 128, 128)).astype(np.float32))
        with torch.no_grad():
            out = net(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# MantraNet
# ---------------------------------------------------------------------------

class TestMantraNet:
    def test_weight_file_constant(self) -> None:
        from forenscope.models.mantranet import MantraNet
        assert MantraNet.WEIGHT_FILE == "mantranet_finetuned.pth"

    def test_model_name_constant(self) -> None:
        from forenscope.models.mantranet import MantraNet
        assert MantraNet.MODEL_NAME == ModelName.MANTRA_NET

    def test_instantiates_without_weights(self, weights_dir: Path) -> None:
        from forenscope.models.mantranet import MantraNet
        model = MantraNet(weights_dir)
        assert model is not None

    def test_raises_weights_not_found_on_predict(
        self, weights_dir: Path, dummy_rgb: np.ndarray
    ) -> None:
        from forenscope.models.mantranet import MantraNet
        model = MantraNet(weights_dir)
        with pytest.raises(WeightsNotFoundError):
            model.predict(dummy_rgb)

    def test_architecture_has_features_and_anomaly(self) -> None:
        pytest.importorskip("torchvision")
        from forenscope.models.mantranet import _MantraNetModel
        net = _MantraNetModel()
        assert hasattr(net, "features")
        assert hasattr(net, "anomaly")

    def test_early_blocks_are_frozen(self) -> None:
        pytest.importorskip("torchvision")
        from forenscope.models.mantranet import _MantraNetModel
        net = _MantraNetModel()
        for i, layer in enumerate(net.features):
            for p in layer.parameters():
                if i < 20:
                    assert not p.requires_grad, f"Block {i} should be frozen"
                else:
                    assert p.requires_grad, f"Block {i} should be trainable"
                break  # check one param per layer


# ---------------------------------------------------------------------------
# SPSL
# ---------------------------------------------------------------------------

class TestSPSL:
    def test_weight_file_constant(self) -> None:
        from forenscope.models.spsl import SPSL
        assert SPSL.WEIGHT_FILE == "spsl_siamese.pth"

    def test_model_name_constant(self) -> None:
        from forenscope.models.spsl import SPSL
        assert SPSL.MODEL_NAME == ModelName.SPSL

    def test_instantiates_without_weights(self, weights_dir: Path) -> None:
        from forenscope.models.spsl import SPSL
        model = SPSL(weights_dir)
        assert model is not None

    def test_raises_weights_not_found_on_predict(
        self, weights_dir: Path, dummy_rgb: np.ndarray
    ) -> None:
        from forenscope.models.spsl import SPSL
        model = SPSL(weights_dir)
        with pytest.raises(WeightsNotFoundError):
            model.predict(dummy_rgb)

    def test_backbone_has_correct_layers(self) -> None:
        pytest.importorskip("torchvision")
        from forenscope.models.spsl import _SPSLBackbone
        bb = _SPSLBackbone()
        assert hasattr(bb, "stem")
        assert hasattr(bb, "layer1")
        assert hasattr(bb, "layer2")
        assert hasattr(bb, "layer3")
        assert hasattr(bb, "pool")
        assert hasattr(bb, "head")

    def test_projection_head_output_dim(self) -> None:
        torch = pytest.importorskip("torch")
        from forenscope.models.spsl import _ProjectionHead, _EMB_DIM
        head = _ProjectionHead()
        x = torch.zeros(4, 1024)
        out = head(x)
        assert out.shape == (4, _EMB_DIM)

    def test_projection_head_is_l2_normalised(self) -> None:
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        from forenscope.models.spsl import _ProjectionHead
        rng = np.random.default_rng(2)
        head = _ProjectionHead()
        x = torch.from_numpy(rng.random((8, 1024)).astype(np.float32))
        out = head(x)
        norms = out.norm(dim=1)
        np.testing.assert_allclose(norms.detach().numpy(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# InpaintingDetector
# ---------------------------------------------------------------------------

class TestInpaintingDetector:
    def test_weight_file_constant(self) -> None:
        from forenscope.models.inpainting import InpaintingDetector
        assert InpaintingDetector.WEIGHT_FILE == "inpainting_detector_clip.pth"

    def test_model_name_constant(self) -> None:
        from forenscope.models.inpainting import InpaintingDetector
        assert InpaintingDetector.MODEL_NAME == ModelName.INPAINTING_DETECTOR

    def test_instantiates_without_weights(self, weights_dir: Path) -> None:
        from forenscope.models.inpainting import InpaintingDetector
        model = InpaintingDetector(weights_dir)
        assert model is not None

    def test_raises_weights_not_found_on_predict(
        self, weights_dir: Path, dummy_rgb: np.ndarray
    ) -> None:
        from forenscope.models.inpainting import InpaintingDetector
        model = InpaintingDetector(weights_dir)
        with pytest.raises(WeightsNotFoundError):
            model.predict(dummy_rgb)

    def test_segmentation_head_layer_names(self) -> None:
        torch = pytest.importorskip("torch")
        from forenscope.models.inpainting import _SegmentationHead
        head = _SegmentationHead()
        assert hasattr(head, "proj")
