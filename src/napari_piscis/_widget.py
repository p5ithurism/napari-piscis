from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Tuple, List, Union, Dict, Any, Optional

import numpy as np
import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning, show_error
from magicgui import magic_factory
from loguru import logger

try:
    from skimage.color import rgb2gray
    from piscis import Piscis
    from piscis.utils import pad_and_stack
    DEPENDENCIES_INSTALLED = True
except ImportError as e:
    DEPENDENCIES_INSTALLED = False
    Piscis = None
    pad_and_stack = None
    rgb2gray = None

if TYPE_CHECKING:
    import napari.layers
    import napari.types


# --- 1. Pure Scientific Logic ---

def infer_img_axes(shape: tuple) -> str:
    """Infers axes (e.g. 'yx', 'zyx') from shape."""
    if len(shape) == 2:
        return 'yx'
    elif len(shape) == 3:
        if shape[-1] in (3, 4):
            return 'yxc'
        min_dim_idx = shape.index(min(shape))
        low_dim_shape = list(shape)
        low_dim_shape.pop(min_dim_idx)
        low_dim_axes = infer_img_axes(tuple(low_dim_shape))
        return low_dim_axes[:min_dim_idx] + 'z' + low_dim_axes[min_dim_idx:]
    else:
        raise ValueError(f"Image shape {shape} is not supported.")

def run_inference_logic(
    raw_image: np.ndarray,
    model_name: str,
    threshold: float,
    min_distance: int,
    intermediates: bool
) -> Dict[str, Any]:
    """Pure python function to run PISCIS inference."""
    if not DEPENDENCIES_INSTALLED:
        raise ImportError("PISCIS or skimage not installed.")

    # 1. Analyze Input
    axes = infer_img_axes(raw_image.shape)
    logger.info(f"Input Shape {raw_image.shape}, Inferred axes {axes}")

    is_3d_stack = 'z' in axes
    
    # 2. Initialize Model
    model = Piscis(model_name=model_name)
    
    # 3. Preprocess
    images_batch = None
    processed_image_for_display = None 
    was_color_converted = False

    if is_3d_stack:
        if axes == 'zyx':
            images_batch = raw_image
        elif axes == 'yxz':
            images_batch = np.moveaxis(raw_image, -1, 0)
        else:
            raise ValueError(f"Unsupported 3D axis pattern: {axes}")
    else:
        if axes == 'yx':
            images_batch = raw_image
        elif axes == 'yxc':
            gray_img = rgb2gray(raw_image)
            images_batch = gray_img 
            processed_image_for_display = gray_img
            was_color_converted = True
        else:
            raise ValueError(f"Unsupported 2D axis pattern: {axes}")

    # 4. Pad and Stack
    images_padded = pad_and_stack(images_batch)

    # 5. Predict
    logger.info("PISCIS: Predicting...")
    coords_pred, features = model.predict(
        images_padded, 
        threshold=threshold, 
        intermediates=intermediates, 
        min_distance=min_distance,
        stack=is_3d_stack
    )
    
    return {
        'coords': coords_pred,
        'features': features,
        'is_3d_stack': is_3d_stack,
        'padded_shape': images_padded.shape,
        'processed_image': processed_image_for_display,
        'was_color_converted': was_color_converted
    }


# --- 2. Worker Thread ---

@thread_worker
def piscis_worker(
    raw_image: np.ndarray,
    model_name: str,
    threshold: float,
    min_distance: int,
    intermediates: bool
):
    try:
        return run_inference_logic(
            raw_image, model_name, threshold, min_distance, intermediates
        )
    except Exception as e:
        raise e


# --- 3. Visualization Helper ---

def _display_features(viewer: napari.Viewer, features: Any, is_3d_stack: bool, layer_name: str):
    """Parses PISCIS feature maps and adds them to the Napari viewer."""
    features_np = np.array(features)
    
    try:
        if is_3d_stack:
            # Handle 3D Stacks
            if features_np.ndim == 5:
                feats = features_np[0] 
            elif features_np.ndim == 4:
                feats = features_np
            else:
                raise ValueError(f"Unexpected 3D shape: {features_np.shape}")
            
            if feats.shape[0] >= 2:
                disp_y = features_np[:, 0, :, :]
                disp_x = features_np[:, 1, :, :]
                
                viewer.add_image(disp_y, name=f"Disp Y ({layer_name})", visible=False)
                viewer.add_image(disp_x, name=f"Disp X ({layer_name})", visible=False)

            if feats.shape[0] > 2:
                labels = features_np[:, 2, :, :]
                viewer.add_image(labels, name=f"Labels ({layer_name})", visible=False)
                pooled = features_np[:, 3, :, :]
                viewer.add_image(pooled, name=f"Pooled Labels ({layer_name})", visible=False)

        else:
            # Handle 2D Images
            if features_np.ndim == 4:
                feats = features_np[0]
            else:
                feats = features_np
            
            if feats.shape[0] >= 2:
                
                viewer.add_image(feats[0], name=f"Disp Y ({layer_name})", visible=False)
                viewer.add_image(feats[1], name=f"Disp X ({layer_name})", visible=False)

            if feats.shape[0] > 2:
                viewer.add_image(feats[2], name=f"Labels ({layer_name})", visible=False)
                viewer.add_image(feats[3], name=f"Pooled Labels ({layer_name})", visible=False)
                
    except Exception as e:
        logger.error(f"Error parsing features: {e}")
        show_error(f"Error displaying features: {e}")
        viewer.add_image(features_np, name=f"Raw Features ({layer_name})")


# --- 4. Main Widget ---

@magic_factory(
    call_button="Run PISCIS",
    layout="vertical",
    image_layer={"label": "Select Input"},
    model_name={"label": "Model Name", "choices": ["20230905"]},
    threshold={"label": "Threshold", "min": 0.0, "max": 1.0, "step": 0.1, "value": 1.0, "tooltip": "Minimum pixels for a spot."},
    min_distance={"label": "Min Distance", "min": 0, "max": 20, "step": 1, "value": 1},
    intermediates={"label": "Return Feature Maps", "value": False},
)
def piscis_inference(
    viewer: napari.Viewer,
    image_layer: "napari.layers.Image",
    model_name: str = "20230905",
    threshold: float = 1.0,
    min_distance: int = 1,
    intermediates: bool = False,
):
    if not DEPENDENCIES_INSTALLED:
        show_error("PISCIS dependencies not found. Please install: pip install piscis scikit-image")
        return
    if image_layer is None:
        show_warning("Please select an image layer.")
        return

    raw_image = np.asarray(image_layer.data)
    show_info(f"PISCIS: Preparing input...")

    worker = piscis_worker(
        raw_image,
        model_name,
        threshold,
        int(min_distance),
        intermediates
    )

    def on_success(result):
        coords_pred = result['coords']
        features = result['features']
        is_3d_stack = result['is_3d_stack']
        was_color_converted = result['was_color_converted']
        processed_image = result['processed_image']
        
        # 1. Handle Color Conversion
        if was_color_converted and processed_image is not None:
            show_warning(f"Color input detected. Converted to grayscale for processing.")
            
            viewer.add_image(
                processed_image, 
                name=f"Grayscale Input ({image_layer.name})",
                colormap='gray'
            )
            # Hide the original RGB layer
            image_layer.visible = False
        
        # 2. Handle Coordinates
        if len(coords_pred) > 0:
            viewer.add_points(
                np.array(coords_pred),
                name=f"Spots ({image_layer.name})",
                size=3,
                face_color='lime',
                symbol='disc',
            )
            show_info(f"PISCIS: Detected {len(coords_pred)} spots.")
        else:
            show_info("PISCIS: No spots detected.")

        # 3. Handle Feature Maps
        if intermediates and features is not None:
             _display_features(viewer, features, is_3d_stack, image_layer.name)

    def on_error(e):
        show_error(f"PISCIS failed: {str(e)}")
        # Loguru syntax to attach the exception object 'e' for a traceback
        logger.opt(exception=e).error("Worker failed")

    worker.returned.connect(on_success)
    worker.errored.connect(on_error)
    worker.start()