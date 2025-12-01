# tests/test_widget.py
import numpy as np
import pytest

import importlib
import magicgui

from napari_piscis import _widget as w


# ---------- Shared Test Helpers ----------

class DummyViewer:
    """Minimal viewer stub for _display_features & piscis_inference tests."""

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


class DummyImageLayer:
    def __init__(self, data, name="Layer"):
        self.data = np.asarray(data)
        self.name = name
        self.visible = True


class DummySignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def emit(self, *args, **kwargs):
        for cb in self._callbacks:
            cb(*args, **kwargs)


class DummyWorker:
    """Mimic a napari thread_worker-style worker."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.returned = DummySignal()
        self.errored = DummySignal()

    def start(self):
        if self._error is not None:
            self.errored.emit(self._error)
        else:
            self.returned.emit(self._result)


# ---------- Autouse fixture: stub PISCIS & utils for pure logic tests ----------

@pytest.fixture(autouse=True)
def stub_piscis_and_utils(monkeypatch):
    """
    Stub Piscis, pad_and_stack, rgb2gray for run_inference_logic tests.
    piscis_inference tests will override piscis_worker directly, so they
    don't depend on these.
    """

    class DummyPiscis:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def predict(self, images_padded, threshold, intermediates, min_distance, stack):
            batch = np.asarray(images_padded)
            if batch.ndim == 2:
                batch = batch[None, ...]
            n_stack = batch.shape[0]
            y, x = batch.shape[-2], batch.shape[-1]
            coords = [(i, 0, 0) for i in range(n_stack)]
            features = np.zeros((n_stack, 4, y, x), dtype=float)
            return coords, features

    def dummy_pad_and_stack(imgs):
        return np.asarray(imgs)

    def dummy_rgb2gray(img):
        return img.mean(axis=-1)

    monkeypatch.setattr(w, "Piscis", DummyPiscis)
    monkeypatch.setattr(w, "pad_and_stack", dummy_pad_and_stack)
    monkeypatch.setattr(w, "rgb2gray", dummy_rgb2gray)
    monkeypatch.setattr(w, "DEPENDENCIES_INSTALLED", True)

    yield


# ---------- infer_img_axes Tests ----------

def test_infer_img_axes_2d_yx():
    assert w.infer_img_axes((32, 64)) == "yx"


def test_infer_img_axes_2d_color_yxc():
    assert w.infer_img_axes((32, 64, 3)) == "yxc"


def test_infer_img_axes_3d_zyx():
    # z is smallest dim at index 0
    assert w.infer_img_axes((5, 32, 32)) == "zyx"


def test_infer_img_axes_3d_yxz():
    # z is smallest dim at index 2
    assert w.infer_img_axes((32, 32, 8)) == "yxz"


def test_infer_img_axes_unsupported_ndim():
    with pytest.raises(ValueError):
        w.infer_img_axes((4, 4, 4, 4))


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

    assert len(coords) == 1
    assert features.shape == (1, 4, 16, 16)
    assert padded_shape[-2:] == img.shape


def test_run_inference_3d_greyscale_zyx():
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

    assert len(coords) == img.shape[0]
    assert features.shape[0] == img.shape[0]
    assert features.shape[1] == 4
    assert padded_shape == img.shape


def test_run_inference_3d_greyscale_yxz():
    # Shape where infer_img_axes -> 'yxz' and we hit the np.moveaxis path.
    img = np.random.rand(16, 16, 8).astype(np.float32)

    result = w.run_inference_logic(
        raw_image=img,
        model_name="20230905",
        threshold=0.5,
        min_distance=2,
        intermediates=True,
    )

    assert result["is_3d_stack"] is True
    coords = result["coords"]
    features = np.asarray(result["features"])
    padded_shape = result["padded_shape"]

    # After moveaxis, batch size should be 8
    assert len(coords) == 8
    assert features.shape[0] == 8
    assert features.shape[1] == 4
    assert padded_shape == (8, 16, 16)


def test_run_inference_2d_color_converts_to_gray():
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
    assert processed.shape == img.shape[:2]

    coords = result["coords"]
    features = np.asarray(result["features"])
    padded_shape = result["padded_shape"]

    assert len(coords) == 1
    assert features.shape == (1, 4, 16, 16)
    assert padded_shape == processed.shape


def test_run_inference_3d_color_currently_unsupported():
    img = np.random.rand(3, 16, 16, 3).astype(np.float32)
    with pytest.raises(ValueError):
        w.run_inference_logic(
            raw_image=img,
            model_name="20230905",
            threshold=0.5,
            min_distance=1,
            intermediates=True,
        )


def test_run_inference_unsupported_3d_axes(monkeypatch):
    # Force an unsupported 3D axis pattern to hit error branch.
    def fake_infer(shape):
        return "xyz"  # not handled in 3D branch

    monkeypatch.setattr(w, "infer_img_axes", fake_infer)
    img = np.random.rand(3, 16, 16).astype(np.float32)

    with pytest.raises(ValueError):
        w.run_inference_logic(
            raw_image=img,
            model_name="20230905",
            threshold=0.5,
            min_distance=1,
            intermediates=True,
        )


def test_run_inference_unsupported_2d_axes(monkeypatch):
    # Force unsupported 2D axes pattern.
    def fake_infer(shape):
        return "xy"

    monkeypatch.setattr(w, "infer_img_axes", fake_infer)
    img = np.random.rand(16, 16).astype(np.float32)

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

    feats = np.random.rand(4, 8, 8)  # (channels, y, x)
    w._display_features(viewer, feats, is_3d_stack=False, layer_name=layer_name)

    assert len(viewer.images) == 4
    names = {img["name"] for img in viewer.images}
    expected = {
        f"Disp Y ({layer_name})",
        f"Disp X ({layer_name})",
        f"Labels ({layer_name})",
        f"Pooled Labels ({layer_name})",
    }
    assert names == expected
    assert all(not img["visible"] for img in viewer.images)


def test_display_features_3d_adds_expected_images_4d():
    viewer = DummyViewer()
    layer_name = "Test3D4D"

    # (z, channel, y, x)
    feats_3d = np.random.rand(3, 4, 8, 8)
    w._display_features(viewer, feats_3d, is_3d_stack=True, layer_name=layer_name)

    assert len(viewer.images) == 4
    names = {img["name"] for img in viewer.images}
    expected = {
        f"Disp Y ({layer_name})",
        f"Disp X ({layer_name})",
        f"Labels ({layer_name})",
        f"Pooled Labels ({layer_name})",
    }
    assert names == expected
    assert all(not img["visible"] for img in viewer.images)


def test_display_features_3d_5d_branch():
    viewer = DummyViewer()
    layer_name = "Test3D5D"

    # (batch, z, channel, y, x) or similar – triggers ndim==5 path
    feats_5d = np.random.rand(1, 4, 3, 8, 8)
    w._display_features(viewer, feats_5d, is_3d_stack=True, layer_name=layer_name)

    # Still should add 4 images
    assert len(viewer.images) == 4


def test_display_features_error_path(monkeypatch):
    viewer = DummyViewer()
    layer_name = "BadShape"

    # Shape that will make code raise inside and hit except block
    bad_feats = np.random.rand(3, 3, 3)  # invalid for current logic

    called = {"error_msg": None}

    def fake_show_error(msg):
        called["error_msg"] = msg

    monkeypatch.setattr(w, "show_error", fake_show_error)

    w._display_features(viewer, bad_feats, is_3d_stack=True, layer_name=layer_name)

    # Error message was shown and raw features were added once
    assert called["error_msg"] is not None
    assert len(viewer.images) == 1
    assert viewer.images[0]["name"] == f"Raw Features ({layer_name})"


# ---------- Helper to get a "plain" widget module ----------

def _get_plain_widget_module(monkeypatch):
    """
    Reload napari_piscis._widget with magic_factory replaced by an
    identity decorator, so piscis_inference is a normal function.
    """
    def identity_magic_factory(*d_args, **d_kwargs):
        def decorator(func):
            return func
        return decorator

    # Patch magicgui.magic_factory so the reload uses the identity decorator
    monkeypatch.setattr(magicgui, "magic_factory", identity_magic_factory)

    import napari_piscis._widget as widget_mod
    widget_mod = importlib.reload(widget_mod)
    return widget_mod


# ---------- piscis_inference Tests (using reloaded module) ----------

def test_piscis_inference_dependencies_missing(monkeypatch):
    w_plain = _get_plain_widget_module(monkeypatch)

    viewer = DummyViewer()
    img_layer = DummyImageLayer(np.zeros((4, 4)), name="Input")

    calls = {"error": None}

    def fake_show_error(msg):
        calls["error"] = msg

    # Dependencies missing branch
    monkeypatch.setattr(w_plain, "DEPENDENCIES_INSTALLED", False, raising=False)
    monkeypatch.setattr(w_plain, "show_error", fake_show_error, raising=False)

    # Now piscis_inference is a plain function
    w_plain.piscis_inference(
        viewer=viewer,
        image_layer=img_layer,
        model_name="20230905",
        threshold=1.0,
        min_distance=1,
        intermediates=False,
    )

    assert calls["error"] is not None
    assert viewer.images == []  # nothing was added


def test_piscis_inference_no_image_layer(monkeypatch):
    w_plain = _get_plain_widget_module(monkeypatch)

    viewer = DummyViewer()
    calls = {"warning": None}
    worker_called = {"called": False}

    def fake_show_warning(msg):
        calls["warning"] = msg

    def fake_worker(*args, **kwargs):
        worker_called["called"] = True
        return DummyWorker(result={}, error=None)

    # Ensure deps "installed" for this branch
    monkeypatch.setattr(w_plain, "DEPENDENCIES_INSTALLED", True, raising=False)
    monkeypatch.setattr(w_plain, "show_warning", fake_show_warning, raising=False)
    monkeypatch.setattr(w_plain, "piscis_worker", fake_worker, raising=False)

    w_plain.piscis_inference(
        viewer=viewer,
        image_layer=None,
        model_name="20230905",
        threshold=1.0,
        min_distance=1,
        intermediates=False,
    )

    assert calls["warning"] is not None
    assert worker_called["called"] is False  # early return, worker never used


def test_piscis_inference_success_color_image(monkeypatch):
    w_plain = _get_plain_widget_module(monkeypatch)

    viewer = DummyViewer()
    img_data = np.zeros((8, 8, 3), dtype=np.float32)
    img_layer = DummyImageLayer(img_data, name="RGBLayer")

    # Result mimics run_inference_logic w/ color conversion + intermediates
    fake_result = {
        "coords": np.array([[1, 2]]),
        "features": np.zeros((1, 4, 8, 8), dtype=float),
        "is_3d_stack": False,
        "padded_shape": (1, 8, 8),
        "processed_image": np.ones((8, 8), dtype=float),
        "was_color_converted": True,
    }

    def fake_worker(raw_image, model_name, threshold, min_distance, intermediates):
        return DummyWorker(result=fake_result, error=None)

    messages = {"warning": None, "info": []}

    def fake_show_warning(msg):
        messages["warning"] = msg

    def fake_show_info(msg):
        messages["info"].append(msg)

    monkeypatch.setattr(w_plain, "DEPENDENCIES_INSTALLED", True, raising=False)
    monkeypatch.setattr(w_plain, "piscis_worker", fake_worker, raising=False)
    monkeypatch.setattr(w_plain, "show_warning", fake_show_warning, raising=False)
    monkeypatch.setattr(w_plain, "show_info", fake_show_info, raising=False)

    w_plain.piscis_inference(
        viewer=viewer,
        image_layer=img_layer,
        model_name="20230905",
        threshold=1.0,
        min_distance=1,
        intermediates=True,  # ensures intermediates branch is taken
    )

    # Color conversion branch
    assert messages["warning"] is not None
    assert img_layer.visible is False
    # One grayscale image + 4 feature maps
    assert len(viewer.images) == 5
    assert len(viewer.points) == 1
    assert any("Detected" in msg for msg in messages["info"])


def test_piscis_inference_worker_error(monkeypatch):
    w_plain = _get_plain_widget_module(monkeypatch)

    viewer = DummyViewer()
    img_layer = DummyImageLayer(np.zeros((4, 4)), name="Input")

    error = RuntimeError("boom")

    def fake_worker(raw_image, model_name, threshold, min_distance, intermediates):
        return DummyWorker(result=None, error=error)

    calls = {"error": None}

    def fake_show_error(msg):
        calls["error"] = msg

    monkeypatch.setattr(w_plain, "DEPENDENCIES_INSTALLED", True, raising=False)
    monkeypatch.setattr(w_plain, "piscis_worker", fake_worker, raising=False)
    monkeypatch.setattr(w_plain, "show_error", fake_show_error, raising=False)

    # This exercises the on_error path in piscis_inference
    w_plain.piscis_inference(
        viewer=viewer,
        image_layer=img_layer,
        model_name="20230905",
        threshold=1.0,
        min_distance=1,
        intermediates=False,
    )

    assert calls["error"] is not None