from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning, show_error
from magicgui import magic_factory
from piscis import Piscis


def _prepare_input(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(image)
    if arr.ndim == 2:          # (Y, X)
        return arr[None, ...], False
    if arr.ndim == 3:          # assume (Z, Y, X)
        return arr[None, ...], True
    raise ValueError(f"Unsupported ndim={arr.ndim}; use 2D (Y,X) or 3D (Z,Y,X).")

def _coords_to_points(coords_pred: Sequence[np.ndarray], is_stack: bool) -> np.ndarray:
    c = coords_pred[0]
    if c.size == 0:
        return np.empty((0, 3 if is_stack else 2))
    if is_stack:
        if c.shape[1] != 3:
            raise ValueError(f"Expected (z,y,x) coords for stacks, got shape {c.shape}")
        return c[:, [0, 1, 2]]
    else:
        if c.shape[1] != 2:
            raise ValueError(f"Expected (y,x) coords for 2D, got shape {c.shape}")
        return c

@thread_worker(start_thread=False)
def _run_piscis_worker(
    image_batched: np.ndarray,
    is_stack: bool,
    model_name: str,
    threshold: float,
    intermediates: bool,
):
    # Load model once inside worker
    model = Piscis(model_name=model_name)

    # (Optional) warm-up tiny call so first real run compiles JAX and is faster
    tiny = np.zeros_like(image_batched[..., :1, :1]) if is_stack else np.zeros_like(image_batched[..., :1, :1])
    _ = model.predict(tiny, threshold=threshold, stack=is_stack, intermediates=False)

    # Real inference
    coords_pred, y = model.predict(
        image_batched, threshold=threshold, stack=is_stack, intermediates=intermediates
    )
    pts = _coords_to_points(coords_pred, is_stack)
    return pts, y, is_stack

@magic_factory(
    call_button="Run Piscis",
    layout="vertical",
    model_name={"label": "Model name", "choices": ["20230905"]},
    threshold={"label": "Confidence threshold", "min": 0.0, "max": 2.0, "step": 0.05, "value": 1.0},
    show_intermediates={"text": "Show intermediates (diagnostics)", "value": False},
)
def piscis_inference(
    viewer: napari.Viewer,                               # <-- napari injects this
    image: "napari.types.ImageData" = None,
    model_name: str = "20230905",
    threshold: float = 1.0,
    show_intermediates: bool = False,
):
    """Dock widget that runs Piscis spot detection."""
    if image is None:
        show_warning("Select an image layer in napari (2D YX or 3D ZYX).")
        return

    try:
        image_batched, is_stack = _prepare_input(image)
    except Exception as e:
        show_error(f"Input error: {e}")
        return

    # Disable the panel while running
    piscis_inference.native.setEnabled(False)
    if viewer is not None:
        viewer.status = "Piscis: running… (first run may JIT-compile JAX)"

    worker = _run_piscis_worker(
        image_batched, is_stack, model_name, threshold, show_intermediates
    )

    def _done(result):
        try:
            pts, y, is_stack_local = result
            layer_name = f"Piscis spots [{model_name}]"
            if pts.size == 0:
                show_info("No spots detected at this threshold.")
            else:
                viewer.add_points(
                    pts, name=layer_name, size=3, ndim=(3 if is_stack_local else 2)
                )
            if show_intermediates and y is not None:
                arr = y[0] if y.ndim == 4 else y[0, 0]
                viewer.add_image(arr, name="Piscis | intermediate")
        finally:
            piscis_inference.native.setEnabled(True)
            if viewer is not None:
                viewer.status = "Piscis: done."

    def _error(err: BaseException):
        show_error(f"Piscis inference failed: {err}")
        piscis_inference.native.setEnabled(True)
        if viewer is not None:
            viewer.status = ""

    worker.returned.connect(_done)
    worker.errored.connect(_error)
    worker.start()