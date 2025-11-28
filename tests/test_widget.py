import numpy as np
import pytest

from napari_piscis import _widget as w


# --------------------------
# infer_img_axes tests
# --------------------------

def test_infer_img_axes_2d_yx():
    """2D image should be interpreted as 'yx'."""
    assert w.infer_img_axes((10, 20)) == "yx"


def test_infer_img_axes_3d_color_yxc():
    """3D image with last dim 3/4 should be treated as color."""
    assert w.infer_img_axes((10, 20, 3)) == "yxc"
    assert w.infer_img_axes((32, 32, 4)) == "yxc"


def test_infer_img_axes_3d_stack_zyx_and_yxz():
    """Smallest dimension should be interpreted as z."""
    # z at the front
    assert w.infer_img_axes((5, 64, 64)) == "zyx"
    # z at the end
    assert w.infer_img_axes((64, 64, 5)) == "yxz"


def test_infer_img_axes_unsupported_shape():
    """Shapes with more than 3 dimensions should raise."""
    with pytest.raises(ValueError):
        w.infer_img_axes((2, 3, 4, 5))


# --------------------------
# run_inference_logic tests
# --------------------------

def test_run_inference_logic_2d_grayscale(monkeypatch):
    """
    For a simple 2D image:
    - axes should be 'yx'
    - is_3d_stack should be False
    - pad_and_stack should be called with a 3D array of shape (1, y, x)
    - return dict should echo coords/features from the fake model
    """

    # --- fake Piscis model ------------------------------------------
    class FakePiscis:
        last_call = None

        def __init__(self, model_name):
            self.model_name = model_name

        def predict(
            self, images_padded, threshold, intermediates, min_distance, stack
        ):
            # record what was passed in for later assertions
            FakePiscis.last_call = dict(
                images_shape=images_padded.shape,
                threshold=threshold,
                intermediates=intermediates,
                min_distance=min_distance,
                stack=stack,
            )
            coords = [[1, 2], [3, 4]]
            features = np.zeros((2, 10, 20), dtype=float)
            return coords, features

    # --- fake pad_and_stack -----------------------------------------
    def fake_pad_and_stack(images):
        fake_pad_and_stack.last_input_shape = images.shape
        # in this simple case, don't actually pad; just echo back
        return images

    # patch module globals
    monkeypatch.setattr(w, "Piscis", FakePiscis)
    monkeypatch.setattr(w, "pad_and_stack", fake_pad_and_stack)
    monkeypatch.setattr(w, "DEPENDENCIES_INSTALLED", True)

    # simple 2D input
    raw_image = np.ones((10, 20), dtype=float)

    result = w.run_inference_logic(
        raw_image=raw_image,
        model_name="test-model",
        threshold=0.5,
        min_distance=2,
        intermediates=True,
    )

    # --- assertions on result dict ----------------------------------
    assert result["coords"] == [[1, 2], [3, 4]]
    assert isinstance(result["features"], np.ndarray)
    assert result["features"].shape == (2, 10, 20)
    assert result["is_3d_stack"] is False
    # pad_and_stack simply echoes input in our fake, so shape matches
    assert result["padded_shape"] == (1, 10, 20)
    assert result["processed_image"] is None
    assert result["was_color_converted"] is False

    # --- assertions on how model was called -------------------------
    assert FakePiscis.last_call is not None
    assert FakePiscis.last_call["images_shape"] == (1, 10, 20)
    assert FakePiscis.last_call["threshold"] == 0.5
    assert FakePiscis.last_call["intermediates"] is True
    assert FakePiscis.last_call["min_distance"] == 2
    assert FakePiscis.last_call["stack"] is False

    # pad_and_stack should have seen a batch dimension
    assert fake_pad_and_stack.last_input_shape == (1, 10, 20)


# --------------------------
# _display_features tests
# --------------------------

class DummyViewer:
    """Minimal stand-in for napari.Viewer for testing _display_features."""

    def __init__(self):
        self.images = []

    def add_image(self, data, name=None, visible=True, colormap=None):
        # store a copy so later modifications don't affect the record
        self.images.append(
            {
                "data": np.array(data),
                "name": name,
                "visible": visible,
                "colormap": colormap,
            }
        )


def test_display_features_2d_adds_expected_images():
    """
    For a 2D feature map with 4 channels (disp_y, disp_x, labels, pooled),
    _display_features should:
    - compute a magnitude image
    - add five images with the expected names
    - mark them all as not visible by default
    """
    viewer = DummyViewer()
    layer_name = "TestLayer"

    # feats[0] = disp_y, feats[1] = disp_x, feats[2] = labels, feats[3] = pooled
    feats = np.random.rand(4, 5, 5)
    w._display_features(viewer, feats, is_3d_stack=False, layer_name=layer_name)

    # We expect 4 images:
    #  - Disp Y (TestLayer)
    #  - Disp X (TestLayer)
    #  - Labels (TestLayer)
    #  - Pooled Labels (TestLayer)
    assert len(viewer.images) == 4

    names = {img["name"] for img in viewer.images}
    expected_names = {
        f"Disp Y ({layer_name})",
        f"Disp X ({layer_name})",
        f"Labels ({layer_name})",
        f"Pooled Labels ({layer_name})",
    }

    # All expected names should be present, no extras
    assert names == expected_names

    # All images should be invisible by default
    assert all(img["visible"] is False for img in viewer.images)
