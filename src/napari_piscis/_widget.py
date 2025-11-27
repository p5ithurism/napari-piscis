from __future__ import annotations
import time
import warnings
warnings.filterwarnings(
    "ignore",
    message="jax.lib.xla_bridge.get_backend is deprecated",
    category=DeprecationWarning,
)

from typing import Sequence, Tuple, Union, List
import numpy as np
import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning, show_error
from magicgui import magic_factory
from piscis import Piscis


# --------- helpers ---------
def _to_float01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32, copy=False)
        vmax = float(np.nanmax(a)) if a.size else 1.0
        if vmax > 1.5:  # looks like 0..255 floats
            a = a / 255.0
        return a
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        return (a.astype(np.float32) / info.max)
    return a.astype(np.float32)

def _rgb_to_gray(a: np.ndarray) -> np.ndarray:
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a.mean(axis=-1).astype(np.float32)

def _prepare_input(
    image: np.ndarray,
    assume_zstack: bool,
) -> Tuple[np.ndarray, bool, str]:
    """
    Returns (unbatched_array, is_stack, debug_string).

    Piscis expects:
      - 2D single: (Y, X) with stack=False
      - 3D Z-stack: (Z, Y, X) with stack=True
    """
    raw = np.asarray(image)
    dbg = [f"raw shape={raw.shape} dtype={raw.dtype}"]

    arr = _to_float01(raw)
    dbg.append(f"norm dtype={arr.dtype} min={arr.min():.3g} max={arr.max():.3g}")

    # If last axis looks like color, convert to gray
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = _rgb_to_gray(arr)
        dbg.append("detected RGB(A) → grayscale (Y,X)")

    if arr.ndim == 2:
        stack = False          # pass (Y,X)
        dbg.append("treat as 2D → (Y,X), stack=False")
        return arr, stack, " | ".join(dbg)

    if arr.ndim == 3:
        if assume_zstack:
            stack = True       # pass (Z,Y,X)
            dbg.append("treat as Z-stack → (Z,Y,X), stack=True")
            return arr, stack, " | ".join(dbg)
        else:
            # If not a Z-stack, treat as batch of 2D and collapse to one 2D if possible
            if arr.shape[0] == 1:
                dbg.append("3D but not zstack: squeezing first axis to 2D")
                return arr[0], False, " | ".join(dbg)
            # If true batch, pick the first for now (simplest UX)
            dbg.append("batch of 2D: using first slice; consider adding batch UI later")
            return arr[0], False, " | ".join(dbg)

    raise ValueError(f"Unsupported ndim={arr.ndim}; provide 2D (Y,X), RGB (Y,X,3|4), or 3D (Z,Y,X).")

def _coords_to_points(
    coords_pred: Union[np.ndarray, Sequence[np.ndarray], List[np.ndarray]],
    is_stack: bool
) -> np.ndarray:
    """
    Piscis may return:
      - a single ndarray of shape (N,2) or (N,3), OR
      - a list like [ndarray] when tiling/batching paths are used.
    Accept both.
    """
    if isinstance(coords_pred, list) or isinstance(coords_pred, tuple):
        if len(coords_pred) == 0:
            return np.empty((0, 3 if is_stack else 2))
        c = np.asarray(coords_pred[0])
    else:
        c = np.asarray(coords_pred)

    if c.size == 0:
        return np.empty((0, 3 if is_stack else 2))

    if is_stack:
        if c.shape[1] != 3:
            raise ValueError(f"Expected (z,y,x) for stacks; got {c.shape}")
        return c[:, [0, 1, 2]]
    else:
        if c.shape[1] != 2:
            raise ValueError(f"Expected (y,x) for 2D; got {c.shape}")
        return c


# --------- worker ---------
@thread_worker(start_thread=False)
def _run_piscis_worker(
    image_unbatched: np.ndarray,   # (Y,X) or (Z,Y,X)
    is_stack: bool,
    model_name: str,
    threshold: float,
    intermediates: bool,
):
    t0 = time.time()
    model = Piscis(model_name=model_name)

    # Warm-up ON THE REAL SHAPE (unbatched) to avoid second compile
    warm = np.zeros_like(image_unbatched)
    _ = model.predict(warm, threshold=threshold, stack=is_stack, intermediates=False)

    t1 = time.time()
    coords_pred, y = model.predict(
        image_unbatched, threshold=threshold, stack=is_stack, intermediates=intermediates
    )
    pts = _coords_to_points(coords_pred, is_stack)
    t2 = time.time()
    timings = (t1 - t0, t2 - t1, t2 - t0)
    return pts, y, is_stack, timings


# --------- widget ---------
@magic_factory(
    call_button="Run Piscis",
    layout="vertical",
    model_name={"label": "Model name", "choices": ["20230905"]},
    threshold={"label": "Confidence threshold", "min": 0.0, "max": 2.0, "step": 0.05, "value": 1.0},
    show_intermediates={"text": "Show intermediates (diagnostics)", "value": False},
    assume_zstack={"label": "Treat 3D as Z-stack", "value": True},
    dry_run={"text": "Dry-run only (no layers)", "value": False},
)
def piscis_inference(
    viewer: napari.Viewer,
    image: "napari.types.ImageData" = None,
    model_name: str = "20230905",
    threshold: float = 1.0,
    show_intermediates: bool = False,
    assume_zstack: bool = True,
    dry_run: bool = False,
):
    """Run Piscis with strict input shaping: 2D (Y,X) or 3D (Z,Y,X)."""
    if image is None:
        show_warning("Select an image layer (2D YX, RGB YXC, or 3D ZYX).")
        return

    try:
        img_unbatched, is_stack, dbg = _prepare_input(image, assume_zstack)
    except Exception as e:
        show_error(f"Input error: {e}")
        return

    viewer.status = f"Piscis: preparing… {dbg}"

    if dry_run:
        show_info(f"[Dry-run] predict(shape={img_unbatched.shape}, stack={is_stack})")
        viewer.status = "[Dry-run] Ready."
        return

    worker = _run_piscis_worker(
        img_unbatched, is_stack, model_name, threshold, show_intermediates
    )

    def _done(result):
        pts, y, is_stack_local, (twarm, tinfer, ttot) = result
        viewer.status = f"Piscis done. warmup={twarm:.2f}s, infer={tinfer:.2f}s, total={ttot:.2f}s"
        if pts.size == 0:
            show_info("No spots detected at this threshold.")
        else:
            viewer.add_points(
                pts,
                name=f"Piscis spots [{model_name}]",
                size=3,
                ndim=(3 if is_stack_local else 2),
            )
        if show_intermediates and y is not None:
            # y may be (C,Y,X) or (C,Z,Y,X); provide quick looks
            if y.ndim == 3:
                # simple magnitude over first two channels if present
                c = min(2, y.shape[0])
                viewer.add_image(np.linalg.norm(y[:c], axis=0), name="Piscis | ||disp||")
            elif y.ndim == 4:
                # max-projection for 3D
                if y.shape[0] > 2:
                    viewer.add_image(np.max(y[2], axis=0), name="Piscis | labels (maxZ)")
                else:
                    viewer.add_image(np.max(y[0], axis=0), name="Piscis | ch0 (maxZ)")

    def _error(err: BaseException):
        show_error(f"Piscis inference failed: {err}")
        viewer.status = ""

    worker.returned.connect(_done)
    worker.errored.connect(_error)
    viewer.status = "Piscis: running… (first call may JIT-compile)"
    worker.start()