from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Tuple, List, Union

import numpy as np
import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning, show_error
from magicgui import magic_factory

try:
    from skimage.color import rgb2gray
    from piscis import Piscis
    from piscis.utils import pad_and_stack
except ImportError as e:
    # Handle missing dependencies gracefully in the GUI
    Piscis = None
    pad_and_stack = None
    rgb2gray = None

if TYPE_CHECKING:
    import napari.layers
    import napari.types


# --- 1. Utility Functions ---

def infer_img_axes(shape: tuple) -> str:
    """
    Recursively infers the axes of an image based on its shape.
    Returns strings like 'yx', 'zyx', or 'yxc'.
    """
    if len(shape) == 2:
        return 'yx'
    elif len(shape) == 3:
        # If last dim is 3 or 4, assume RGB/RGBA (Channels)
        if shape[-1] in (3, 4):
            return 'yxc'
        
        # Otherwise, assume the smallest dimension is Z or Time, 
        # but standard biological images are usually ZYX if not RGB.
        min_dim_idx = shape.index(min(shape))
        low_dim_shape = list(shape)
        low_dim_shape.pop(min_dim_idx)
        low_dim_axes = infer_img_axes(tuple(low_dim_shape))
        return low_dim_axes[:min_dim_idx] + 'z' + low_dim_axes[min_dim_idx:]
    else:
        raise ValueError(f"Image shape {shape} is not supported by the notebook logic.")


# --- 2. Worker Thread (Inference Logic) ---

@thread_worker
def piscis_worker(
    raw_image: np.ndarray,
    model_name: str,
    threshold: float,
    min_distance: int,
    intermediates: bool
):
    """
    Executes the 'Run PISCIS Inference' cell logic in a background thread.
    """
    # --- Load Image (Axes Inference) ---
    axes = infer_img_axes(raw_image.shape)
    show_info(f"Input Shape {raw_image.shape}")
    show_info(f"Inferred axes {axes}")

    is_3d_stack = 'z' in axes
    
    # 1. Initialize Model
    model = Piscis(model_name=model_name)
    
    # 2. Preprocess Image for Batching
    images_batch = None
    processed_image_for_display = None # Use this to store the grayscale version if created
    was_color_converted = False

    if is_3d_stack:
        if axes == 'zyx':
            images_batch = raw_image
        elif axes == 'yxz':
            images_batch = np.moveaxis(raw_image, -1, 0)
        else:
            raise ValueError(f"Unsupported 3D axis pattern: {axes}")
    else:
        # 2D processing
        if axes == 'yx':
            images_batch = np.expand_dims(raw_image, axis=0)
        elif axes == 'yxc':
            # Convert RGB to Grayscale for detection
            gray_img = rgb2gray(raw_image)
            images_batch = gray_img # piscis expects (N, Y, X) or (Y, X) depending on batching
            
            # Store specific flags to send back to GUI
            processed_image_for_display = gray_img
            was_color_converted = True
        else:
            raise ValueError(f"Unsupported 2D axis pattern: {axes}")

    # 3. Pad and Stack
    images_padded = pad_and_stack(images_batch)

    # 4. Predict
    show_info(f"PISCIS: Predicting...")
    coords_pred, features = model.predict(
        images_padded, 
        threshold=threshold, 
        intermediates=intermediates, 
        min_distance=min_distance,
        stack=is_3d_stack
    )
    
    # Return everything needed for the Export/View step
    return {
        'coords': coords_pred,
        'features': features,
        'is_3d_stack': is_3d_stack,
        'padded_shape': images_padded.shape,
        # Return the processed image so it can be added to the viewer
        'processed_image': processed_image_for_display,
        'was_color_converted': was_color_converted
    }


# --- 3. Widget Definition ---

@magic_factory(
    call_button="Run PISCIS",
    layout="vertical",
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
    """
    A Napari widget that faithfully implements the PISCIS notebook workflow.
    """
    # Basic Validation
    if Piscis is None:
        show_error("PISCIS not found. Please install: pip install piscis")
        return
    if image_layer is None:
        show_warning("Please select an image layer.")
        return

    # Prepare Data
    raw_image = np.asarray(image_layer.data)
    show_info(f"PISCIS: Preparing input...")

    min_distance_int = int(min_distance)    

    # Initialize Worker
    worker = piscis_worker(
        raw_image,
        model_name,
        threshold,
        min_distance_int,
        intermediates
    )

    # --- Callback: Handle Results ---
    def on_success(result):
        coords_pred = result['coords']
        features = result['features']
        is_3d_stack = result['is_3d_stack']
        was_color_converted = result['was_color_converted']
        processed_image = result['processed_image']
        
        # 0. Handle Color Conversion Warning and Display
        # Perform this FIRST to ensure the new layer exists as a reference for points
        if was_color_converted and processed_image is not None:
            show_warning(f"Color input detected. Converted to grayscale for processing.")
            
            # Add the grayscale image to the viewer
            # This ensures that when plotting 2D points, they sit on top of this 2D layer
            viewer.add_image(
                processed_image, 
                name=f"Grayscale Input ({image_layer.name})",
                colormap='gray'
            )
        
        # 1. Handle Coordinates
        if len(coords_pred) > 0:
            coords_array = np.array(coords_pred)
            
            viewer.add_points(
                coords_array,
                name=f"Spots ({image_layer.name})",
                size=3,
                face_color='blue',
                symbol='disc',
            )
            show_info(f"PISCIS: Detected {len(coords_pred)} spots.")
        else:
            show_info("PISCIS: No spots detected.")

        # 2. Handle Feature Maps
        if intermediates and features is not None:
            features_np = np.array(features)
            
            try:
                if is_3d_stack:
                    # Logic for 3D stacks
                    if features_np.ndim == 5:
                        feats = features_np[0] 
                    elif features_np.ndim == 4:
                        feats = features_np
                    else:
                        raise ValueError(f"Unexpected 3D shape: {features_np.shape}")
                    
                    if feats.shape[0] >= 2:
                        disp_y = features_np[:, 0, :, :]
                        disp_x = features_np[:, 1, :, :]
                        mag = np.linalg.norm(feats[:, 0:2, :, :], axis=1)
                        
                        viewer.add_image(mag, name=f"Magnitude ({image_layer.name})", visible=False)
                        viewer.add_image(disp_y, name=f"Disp Y ({image_layer.name})", visible=False)
                        viewer.add_image(disp_x, name=f"Disp X ({image_layer.name})", visible=False)

                    if feats.shape[0] > 2:
                        labels = features_np[:, 2, :, :]
                        viewer.add_image(labels, name=f"Labels ({image_layer.name})", visible=False)
                        pooled = features_np[:, 3, :, :]
                        viewer.add_image(pooled, name=f"Pooled Labels ({image_layer.name})", visible=False)


                else:
                    # Logic for 2D images
                    if features_np.ndim == 4:
                        feats = features_np[0]
                    else:
                        feats = features_np
                    
                    if feats.shape[0] >= 2:
                        mag = np.linalg.norm(feats[0:2], axis=0)
                        
                        viewer.add_image(mag, name=f"Magnitude ({image_layer.name})", visible=False)
                        viewer.add_image(feats[0], name=f"Disp Y ({image_layer.name})", visible=False)
                        viewer.add_image(feats[1], name=f"Disp X ({image_layer.name})", visible=False)

                    if feats.shape[0] > 2:
                        viewer.add_image(feats[2], name=f"Labels ({image_layer.name})", visible=False)
                        viewer.add_image(feats[3], name=f"Pooled Labels ({image_layer.name})", visible=False)
                        
            except Exception as e:
                show_error(f"Error displaying features: {e}")
                viewer.add_image(features_np, name=f"Raw Features ({image_layer.name})")

    def on_error(e):
        show_error(f"PISCIS failed: {str(e)}")

    worker.returned.connect(on_success)
    worker.errored.connect(on_error)
    worker.start()