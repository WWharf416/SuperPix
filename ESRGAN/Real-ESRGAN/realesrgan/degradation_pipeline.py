import cv2
import numpy as np
import random
import yaml # Added for config loading

# Use relative import within the same package
from .degradations import (
    gaussian_blur, motion_blur, gaussian_noise, poisson_noise, salt_pepper_noise,
    random_resize, jpeg_compression, random_grayscale # Add other imported degradations here
)

def apply_degradation_pipeline(img_hr, config):
    """
    Applies a sequence of degradations based on the config dictionary.
    Handles first-order and second-order degradations.

    Args:
        img_hr (np.ndarray): High-resolution input image (H, W, C), uint8, BGR.
        config (dict): Configuration dictionary for degradations, loaded from YAML.

    Returns:
        np.ndarray: Degraded low-resolution image (h, w, c), uint8, BGR.
    """
    # --- Ensure input is uint8 BGR numpy array ---
    if not isinstance(img_hr, np.ndarray):
        # If it's a PIL image or other format, convert it first
        # This depends on how images are loaded in the dataset
        # Assuming PIL Image for now, convert to BGR numpy
        try:
            img_hr = cv2.cvtColor(np.array(img_hr), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise TypeError(f"Input image type not supported or conversion failed: {e}")

    if img_hr.dtype != np.uint8:
         img_hr = np.clip(img_hr, 0, 255).astype(np.uint8)

    img_lq = img_hr.copy()
    final_scale = config.get('final_scale', 4)

    # --- Get Degradation Settings from Config --- 
    # Use get with default empty dict/list to avoid KeyErrors if sections are missing
    degradation_settings = config.get('degradations', {})
    first_order_deg_seq = degradation_settings.get('first_order_degradations', [])
    second_order_deg_seq = degradation_settings.get('second_order_degradations', [])

    # --- First Pass Degradations --- 
    if first_order_deg_seq:
        random.shuffle(first_order_deg_seq)
        for deg_name in first_order_deg_seq:
            deg_config = degradation_settings.get(deg_name, {})
            prob = deg_config.get('probability', 1.0)
            params = deg_config.get('params', {})

            if random.random() < prob:
                try:
                    # Select the correct function based on name
                    if deg_name == 'gaussian_blur':
                        img_lq = gaussian_blur(img_lq, **params)
                    elif deg_name == 'motion_blur':
                        img_lq = motion_blur(img_lq, **params)
                    elif deg_name == 'resize':
                        current_scale_approx = img_lq.shape[0] / img_hr.shape[0]
                        max_scale = params.get('scale_max', 1.5)
                        # Cap intermediate upscale if needed to avoid excessive growth before final downscale
                        cap_scale = max(1.0, (1.0 / final_scale) / current_scale_approx + 0.1)
                        params['scale_max'] = min(max_scale, cap_scale)
                        # Ensure min is not greater than max after capping
                        params['scale_min'] = min(params.get('scale_min', 0.15), params['scale_max'])
                        img_lq = random_resize(img_lq, **params)
                    elif deg_name == 'gaussian_noise':
                        img_lq = gaussian_noise(img_lq, **params)
                    elif deg_name == 'poisson_noise':
                        img_lq = poisson_noise(img_lq) # Add params if implemented
                    elif deg_name == 'salt_pepper_noise':
                        img_lq = salt_pepper_noise(img_lq, **params)
                    elif deg_name == 'jpeg_compression':
                        img_lq = jpeg_compression(img_lq, **params)
                    elif deg_name == 'random_grayscale':
                        # Pass probability from params if defined, else use default 0.1 from function
                        grayscale_prob = params.get('probability', 0.1) # Allow overriding default probability
                        img_lq = random_grayscale(img_lq, probability=grayscale_prob)
                    # --- Add calls to other implemented degradations here --- 

                except Exception as e:
                    print(f"Warning: Failed to apply first-order {deg_name}: {e}")
                    # Continue degradation even if one step fails

    # --- Final Downsampling to Target Scale --- 
    h_hr, w_hr = img_hr.shape[:2]
    h_lq_target, w_lq_target = h_hr // final_scale, w_hr // final_scale
    # Ensure final dimensions are not zero
    h_lq_target = max(1, h_lq_target)
    w_lq_target = max(1, w_lq_target)

    # Use a high-quality resize for the final downsampling
    # TODO: Consider implementing and using sinc filter here as an option via config
    final_resize_interpolation = degradation_settings.get('final_resize_interpolation', 'cubic') # Allow config override
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interpolation = interp_map.get(final_resize_interpolation.lower(), cv2.INTER_CUBIC)

    img_lq = cv2.resize(img_lq, (w_lq_target, h_lq_target), interpolation=interpolation)

    # --- Second Pass Degradations (Applied after final downsampling) --- 
    if second_order_deg_seq:
        random.shuffle(second_order_deg_seq)
        for deg_name in second_order_deg_seq:
            deg_config = degradation_settings.get(deg_name, {})
            prob = deg_config.get('probability', 1.0)
            params = deg_config.get('params', {})

            if random.random() < prob:
                try:
                    # Select the correct function based on name
                    if deg_name == 'gaussian_blur':
                        img_lq = gaussian_blur(img_lq, **params)
                    elif deg_name == 'motion_blur':
                         img_lq = motion_blur(img_lq, **params)
                    # Avoid resize in second pass unless specifically intended (e.g., add a 'resize_final' type)
                    elif deg_name == 'gaussian_noise':
                         img_lq = gaussian_noise(img_lq, **params)
                    elif deg_name == 'poisson_noise':
                         img_lq = poisson_noise(img_lq) # Add params if implemented
                    elif deg_name == 'salt_pepper_noise':
                         img_lq = salt_pepper_noise(img_lq, **params)
                    elif deg_name == 'jpeg_compression':
                         img_lq = jpeg_compression(img_lq, **params)
                    elif deg_name == 'random_grayscale':
                         grayscale_prob = params.get('probability', 0.1)
                         img_lq = random_grayscale(img_lq, probability=grayscale_prob)
                    # --- Add calls to other implemented second-order degradations here --- 

                except Exception as e:
                    print(f"Warning: Failed to apply second-order {deg_name}: {e}")
                    # Continue degradation

    return np.clip(img_lq, 0, 255).astype(np.uint8) # Final clamp and type cast 