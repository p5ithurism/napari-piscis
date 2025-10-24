from typing import Tuple, Sequence
import numpy as np
from magicgui import magicgui
from napari.utils.notifications import show_info, show_warning
from piscis import Piscis

def _prepare_input(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(image)
    if arr.ndim == 2:          # (Y, X)
        return arr[None, ...], False
    if arr.ndim == 3:          # assume (Z, Y, X)
        return arr[None, ...], True
    raise ValueError(f"Unsupported ndim={arr.ndim}; use 2D or 3D images")

def _coords_to_points(coords_pred: Sequence[np.ndarray], is_stack: bool) -> np.ndarray:
    c = coords_pred[0]
    if c.size == 0:
        return np.empty((0, 3 if is_stack else 2))
    if is_stack:
        assert c.shape[1] == 3, "Expected (z,y,x)"
        return c[:, [0, 1, 2]]
    else:
        assert c.shape[1] == 2, "Expected (y,x)"
        return c

def make_inference_widget(viewer=None):
    @magicgui(
        call_button="Run Piscis",
        layout="vertical",
        model_name={"label": "Model name", "choices": ["20230905"]},
        threshold={"label": "Confidence threshold", "min": 0.0, "max": 2.0, "step": 0.05, "value": 1.0},
        show_intermediates={"text": "Show intermediates (diagnostics)", "value": False},
    )
    def widget(
        image: "napari.types.ImageData",
        model_name: str = "20230905",
        threshold: float = 1.0,
        show_intermediates: bool = False,
    ):
        if image is None:
            show_warning("Select an image layer in napari.")
            return
        try:
            batch, is_stack = _prepare_input(image)
            model = Piscis(model_name=model_name)
            coords_pred, y = model.predict(
                batch, threshold=threshold, stack=is_stack, intermediates=show_intermediates
            )
            pts = _coords_to_points(coords_pred, is_stack)
            name = f"Piscis spots [{model_name}]"
            if pts.size == 0:
                show_info("No spots detected at this threshold.")
            else:
                viewer.add_points(pts, name=name, size=3, ndim=(3 if is_stack else 2))
            if show_intermediates and y is not None:
                # Example: add first intermediate channel if present
                arr = y[0] if y.ndim == 4 else y[0, 0]
                viewer.add_image(arr, name="Piscis | intermediate")
        except Exception as e:
            show_warning(f"Piscis inference failed: {e}")
    return widget