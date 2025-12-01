# tests/test_widget.py
import numpy as np
import pytest

# Adjust import if your package name is different
from napari_piscis import _widget as w


# ---------- Fixtures & Test Helpers ----------

@pytest.fixture(autouse=True)
def stub_piscis_and_utils(monkeypatch):
    """Stub out Piscis, pad_and_stack, rgb2gray so tests are fast & deterministic."""

    class DummyPiscis:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def predict(self, images_padded, threshold, intermediates, min_distance, stack):
            # Simple deterministic dummy output:
            # - coords: one fake spot per "image" in the batch
            # - features: 4-channel feature map
            batch = np.asarray(images_padded)
            if batch.ndim == 2:
                # (y, x) -> treat as 1-image batch
                batch = batch[None, ...]
            n_stack = batch.shape[0]
            y, x = batch.shape[-2], batch.shape[-1]

            coords = [(i, 0, 0) for i in range(n_stack)]  # fake coordinates
            features = np.zeros((n_stack, 4, y, x), dtype=float)
            return coords, features

    def dummy_pad_and_stack(imgs):
        # For tests we just convert to numpy without changing shape.
        return np.asarray(imgs)

    def dummy_rgb2gray(img):
        # Average over the last axis for a simple grayscale conversion.
        return img.mean(axis=-1)

    monkeypatch.setattr(w, "Piscis", DummyPiscis)
    monkeypatch.setattr(w, "pad_and_stack", dummy_pad_and_stack)
    monkeypatch.setattr(w, "rgb2gray", dummy_rgb2gray)
    monkeypatch.setattr(w, "DEPENDENCIES_INSTALLED", True)

    yield


class DummyViewer:
    """Minimal viewer stub for _display_features tests."""

    def __init__(self):
        self.images = []
        self.points = []

    def add_image(self, data, name, visible=True, colormap=None):
        self.images.append(
            {
                "data": np.asarray(data),
                "name": name,
                "visible": visible,
                "colormap": colormap,
            }
        )

    def add_points(self, data, name, size=1, face_color=None, symbol=None):
        self.points.append(
            {
                "data": np.asarray(data),
                "name": name,
                "size": size,
                "face_color": face_color,
                "symbol": symbol,
            }
        )


# ---------- infer_img_axes Tests ----------

def test_infer_img_axes_2d_yx():
    shape = (32, 64)
    axes = w.infer_img_axes(shape)
    assert axes == "yx"


def test_infer_img_axes_2d_color_yxc():
    shape = (32, 64, 3)
    axes = w.infer_img_axes(shape)
    assert axes == "yxc"


def test_infer_img_axes_3d_zyx():
    # smallest dim first -> treated as z
    shape = (5, 32, 32)
    axes = w.infer_img_axes(shape)
    assert axes == "zyx"


def test_infer_img_axes_unsupported_ndim():
    with pytest.raises(ValueError):
        w.infer_img_axes((4, 4, 4, 4))  # len(shape) == 4 is unsupported


# ---------- run_inference_logic Tests ----------

def test_run_inference_2d_greyscale():
    img = np.random.rand(16, 16).astype(np.float32)

    result = w.run_inference_logic(
        raw_image=img,
        model_name="20230905",
        threshold=0.5,
        min_distance=2,
        intermediates=True,
    )

    assert result["is_3d_stack"] is False
    assert result["was_color_converted"] is False
    assert result["processed_image"] is None

    coords = result["coords"]
    features = np.asarray(result["features"])
    padded_shape = result["padded_shape"]

    # From DummyPiscis: one coord, and features with shape (batch, 4, y, x)
    assert len(coords) == 1
    assert features.shape[0] == 1
    assert features.shape[1] == 4
    assert padded_shape[-2:] == img.shape


def test_run_inference_3d_greyscale():
    # Shape (z, y, x)
    img = np.random.rand(3, 16, 16).astype(np.float32)

    result = w.run_inference_logic(
        raw_image=img,
        model_name="20230905",
        threshold=0.5,
        min_distance=2,
        intermediates=True,
    )

    assert result["is_3d_stack"] is True
    assert result["was_color_converted"] is False
    assert result["processed_image"] is None

    coords = result["coords"]
    features = np.asarray(result["features"])
    padded_shape = result["padded_shape"]

    # 3 stack images
    assert len(coords) == img.shape[0]
    assert features.shape[0] == img.shape[0]
    assert features.shape[1] == 4
    assert padded_shape == img.shape


def test_run_inference_2d_color_converts_to_gray():
    # (y, x, c)
    img = np.random.rand(16, 16, 3).astype(np.float32)

    result = w.run_inference_logic(
        raw_image=img,
        model_name="20230905",
        threshold=0.5,
        min_distance=1,
        intermediates=True,
    )

    assert result["is_3d_stack"] is False
    assert result["was_color_converted"] is True
    processed = result["processed_image"]
    assert processed is not None
    assert processed.shape == img.shape[:2]  # grayscale

    coords = result["coords"]
    features = np.asarray(result["features"])
    padded_shape = result["padded_shape"]

    assert len(coords) == 1
    assert features.shape[0] == 1
    assert features.shape[1] == 4
    # pad_and_stack returns same shape in our stub
    assert padded_shape == processed.shape


def test_run_inference_3d_color_currently_unsupported():
    # With current infer_img_axes implementation, any 4D shape should fail.
    img = np.random.rand(3, 16, 16, 3).astype(np.float32)
    with pytest.raises(ValueError):
        w.run_inference_logic(
            raw_image=img,
            model_name="20230905",
            threshold=0.5,
            min_distance=1,
            intermediates=True,
        )


# ---------- _display_features Tests ----------

def test_display_features_2d_adds_expected_images():
    viewer = DummyViewer()
    layer_name = "Test2D"

    # features: (channels, y, x)
    feats = np.random.rand(4, 8, 8)

    w._display_features(viewer, feats, is_3d_stack=False, layer_name=layer_name)

    # Expect 4 images: Disp Y, Disp X, Labels, Pooled
    assert len(viewer.images) == 4

    names = {img["name"] for img in viewer.images}
    expected_names = {
        f"Disp Y ({layer_name})",
        f"Disp X ({layer_name})",
        f"Labels ({layer_name})",
        f"Pooled Labels ({layer_name})",
    }
    assert names == expected_names

    # All should be invisible by default
    assert all(not img["visible"] for img in viewer.images)


def test_display_features_3d_adds_expected_images():
    viewer = DummyViewer()
    layer_name = "Test3D"

    # Mimic (z, channel, y, x); _display_features will slice as features_np[:, 0, :, :]
    feats_3d = np.random.rand(3, 4, 8, 8)

    w._display_features(viewer, feats_3d, is_3d_stack=True, layer_name=layer_name)

    # For 3D we still expect 4 images total
    assert len(viewer.images) == 4

    names = {img["name"] for img in viewer.images}
    expected_names = {
        f"Disp Y ({layer_name})",
        f"Disp X ({layer_name})",
        f"Labels ({layer_name})",
        f"Pooled Labels ({layer_name})",
    }
    assert names == expected_names

    assert all(not img["visible"] for img in viewer.images)
