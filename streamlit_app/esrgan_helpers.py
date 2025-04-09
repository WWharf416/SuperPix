import os
import cv2
import torch
import numpy as np
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Default model paths and URLs (can be adjusted)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ESRGAN', 'Real-ESRGAN', 'weights')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_INFO = {
    'realesr-general-x4v3': {
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_conv': 32, 'upscale': 4, 'act_type': 'prelu'},
        'scale': 4,
        'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'],
        'filename': 'realesr-general-x4v3.pth'
    },
    'realesr-animevideov3': {
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_conv': 16, 'upscale': 4, 'act_type': 'prelu'},
        'scale': 4,
        'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'],
        'filename': 'realesr-animevideov3.pth'
    },
    # Add more models here if needed
}

def get_model_path(model_name):
    """Checks if model exists, downloads if necessary, and returns the path."""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unsupported model name: {model_name}")

    info = MODEL_INFO[model_name]
    model_path = os.path.join(MODEL_DIR, info['filename'])

    if not os.path.isfile(model_path):
        print(f"Model weights for {model_name} not found. Downloading...")
        for url in info['file_url']:
            try:
                model_path = load_file_from_url(
                    url=url, model_dir=MODEL_DIR, progress=True, file_name=info['filename'])
                break # Stop after successful download from one URL
            except Exception as e:
                 print(f"  Error downloading model weights from {url}: {e}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Failed to download model weights for {model_name}. Please check connectivity or download manually to {MODEL_DIR}.")
        print(f"Model downloaded to: {model_path}")
    return model_path


def initialize_upsampler(model_name, enhancement_type='none', outscale=4, model_path=None, tile=0, tile_pad=10, pre_pad=0, half=True, gpu_id=None):
    """Initializes and returns a RealESRGANer instance."""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unsupported model name: {model_name}")

    info = MODEL_INFO[model_name]
    netscale = info['scale']

    # Allow overriding default model path
    if model_path is None:
        model_path = get_model_path(model_name)
    elif not os.path.isfile(model_path):
         raise FileNotFoundError(f"Provided model path {model_path} not found.")

    # Adjust model parameters for the selected enhancement type
    model_params = info['params'].copy()
    model_params['upscale'] = int(outscale) # Use desired outscale, not necessarily model's native scale
    try:
        model = SRVGGNetCompact(**model_params, enhancement_type=enhancement_type)
    except ImportError as e:
        if enhancement_type == 'clahe':
             print(f"Warning: Cannot initialize CLAHE model - {e}. Scikit-image might be missing.")
             return None # Indicate failure for CLAHE
        else:
             raise e # Re-raise other import errors


    # Initialize upsampler
    try:
        upsampler = RealESRGANer(
            scale=netscale, # Use model's native scale here
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            gpu_id=gpu_id
        )
        print(f"Initialized {enhancement_type} upsampler for {model_name} (outscale={outscale}). FP16: {half}")
        return upsampler
    except Exception as e:
        print(f"Error initializing {enhancement_type} upsampler for {model_name}: {e}")
        # Consider more specific error handling (e.g., CUDA issues)
        return None


def upscale_image(upsampler, img_cv2, outscale):
    """Upscales a single image using the provided upsampler."""
    if upsampler is None:
        print("Upsampler not initialized, skipping upscale.")
        return None

    print(f"  Input dimensions: {img_cv2.shape[1]}x{img_cv2.shape[0]}")
    try:
        # The RealESRGANer.enhance method handles the actual upscaling factor with outscale
        output_img, _ = upsampler.enhance(img_cv2, outscale=outscale)
        print(f"  Output dimensions: {output_img.shape[1]}x{output_img.shape[0]}")
        return output_img
    except RuntimeError as error:
        print(f'  Error during inference: {error}')
        print('  Try reducing tile size or disabling FP16 (half=False) if encountering CUDA OOM.')
        return None
    except Exception as error:
        print(f'  Unexpected error during inference: {error}')
        return None


def downscale_image(img_cv2, scale_factor):
    """Downscales an image using cv2.resize."""
    if scale_factor >= 1.0:
        print("Warning: scale_factor for downscaling should be < 1.0")
        return img_cv2 # Or raise error?

    h_hr, w_hr = img_cv2.shape[:2]
    img_lr = cv2.resize(img_cv2, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    h_lr, w_lr = img_lr.shape[:2]

    if h_lr == 0 or w_lr == 0:
         print(f"Warning: Downsampling resulted in zero dimension for image shape {img_cv2.shape}. Returning original.")
         return img_cv2 # Return original to avoid errors downstream

    print(f"  Downscaled from {w_hr}x{h_hr} to {w_lr}x{h_lr} (factor {scale_factor:.2f})")
    return img_lr


def calculate_metrics(img_gt, img_pred):
    """Calculates PSNR and SSIM between ground truth and prediction."""
    if img_gt is None or img_pred is None:
        return None, None

    if img_gt.shape != img_pred.shape:
        print(f"  Warning: Shape mismatch GT {img_gt.shape} vs Pred {img_pred.shape}. Attempting resize.")
        height, width = img_gt.shape[:2]
        try:
            img_pred_resized = cv2.resize(img_pred, (width, height), interpolation=cv2.INTER_AREA)
            if img_gt.shape != img_pred_resized.shape:
                print(f"  Error: Cannot align shapes for comparison after resize: GT {img_gt.shape} vs Pred {img_pred_resized.shape}")
                return None, None
            img_pred = img_pred_resized
        except Exception as e:
            print(f"  Error resizing prediction for metric calculation: {e}")
            return None, None


    # Ensure images are uint8
    if img_gt.dtype != np.uint8:
        img_gt = (img_gt * 255).clip(0, 255).astype(np.uint8)
    if img_pred.dtype != np.uint8:
        img_pred = (img_pred * 255).clip(0, 255).astype(np.uint8)

    try:
        psnr_val = psnr(img_gt, img_pred, data_range=255)
    except Exception as e:
        print(f"  Error calculating PSNR: {e}")
        psnr_val = None

    ssim_val = None
    try:
        is_multichannel = len(img_gt.shape) == 3 and img_gt.shape[2] > 1
        # Ensure win_size is appropriate and odd, and <= smallest dimension
        win_size = min(7, img_gt.shape[0], img_gt.shape[1])
        if win_size % 2 == 0: win_size -= 1

        if win_size >= 3: # SSIM requires win_size >= 3
            if is_multichannel:
                # Use channel_axis=-1 for newer scikit-image versions
                ssim_val = ssim(img_gt, img_pred, data_range=255, channel_axis=-1 if hasattr(ssim, 'channel_axis') else 'multichannel', win_size=win_size, multichannel=True) # Provide multichannel for older versions
            else: # Grayscale
                 ssim_val = ssim(img_gt, img_pred, data_range=255, win_size=win_size)
        else:
            print(f"  Warning: Cannot compute SSIM for image size {img_gt.shape} with win_size {win_size}.")

    except Exception as e:
        print(f"  Error calculating SSIM: {e}")
        # Might fail if win_size is too large for the image, etc.
        ssim_val = None


    print(f"  Metrics: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}" if psnr_val is not None and ssim_val is not None else f" Metrics: PSNR={psnr_val}, SSIM={ssim_val}")
    return psnr_val, ssim_val 