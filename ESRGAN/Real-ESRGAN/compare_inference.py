import argparse
import cv2
import glob
import os
import torch
import numpy as np
from basicsr.utils.download_util import load_file_from_url
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Assuming realesrgan and its dependencies are installed or in the PYTHONPATH
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def calculate_metrics(img_gt, img_pred):
    """Calculates PSNR and SSIM between ground truth and prediction."""
    if img_gt.shape != img_pred.shape:
        # Attempt to resize prediction to match GT if dimensions mismatch slightly
        # This might happen due to floating point issues in down/upscaling
        print(f"  Warning: Shape mismatch GT {img_gt.shape} vs Pred {img_pred.shape}. Attempting resize.")
        height, width = img_gt.shape[:2]
        img_pred = cv2.resize(img_pred, (width, height), interpolation=cv2.INTER_AREA) # Use INTER_AREA for potential downscaling
        if img_gt.shape != img_pred.shape: # Check again after resize
             raise ValueError(f"Cannot align shapes for comparison: GT {img_gt.shape} vs Pred {img_pred.shape}")


    # Ensure images are uint8
    if img_gt.dtype != np.uint8:
        img_gt = (img_gt * 255).clip(0, 255).astype(np.uint8)
    if img_pred.dtype != np.uint8:
        img_pred = (img_pred * 255).clip(0, 255).astype(np.uint8)

    psnr_val = psnr(img_gt, img_pred, data_range=255)

    # Calculate SSIM. Use multichannel=True for color images.
    is_multichannel = len(img_gt.shape) == 3 and img_gt.shape[2] > 1
    # Ensure win_size is appropriate and odd
    win_size = min(7, img_gt.shape[0], img_gt.shape[1])
    if win_size % 2 == 0: win_size -= 1

    if win_size >= 3:
        if is_multichannel:
            ssim_val = ssim(img_gt, img_pred, data_range=255, channel_axis=-1, win_size=win_size)
        else: # Grayscale
             ssim_val = ssim(img_gt, img_pred, data_range=255, win_size=win_size)
    else:
        print(f"  Warning: Cannot compute SSIM for image size {img_gt.shape} with win_size >= 3.")
        ssim_val = None

    return psnr_val, ssim_val


def main():
    """Comparison demo for Real-ESRGAN with/without hist eq, including downsampling and evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs_hr', help='Input folder containing HIGH-RESOLUTION images')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-general-x4v3', # Defaulting to a SRVGG model
        help=('SRVGG Model names: realesr-animevideov3 | realesr-general-x4v3')
    )
    parser.add_argument('-o', '--output', type=str, default='results_compare_eval', help='Output folder for upscaled images')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The downsampling/upsampling scale factor')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    # parser.add_argument( # Output extension is less critical now, defaults to PNG for consistency
    #     '--ext',
    #     type=str,
    #     default='png',
    #     help='Image extension. Options: png | jpg')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # --- Model Setup ---
    args.model_name = args.model_name.split('.')[0]
    try:
        if args.model_name == 'realesr-animevideov3':
            model_params = {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_conv': 16, 'upscale': int(args.outscale), 'act_type': 'prelu'}
            netscale = int(args.outscale)
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif args.model_name == 'realesr-general-x4v3':
            if int(args.outscale) != 4:
                 print(f"Warning: Model 'realesr-general-x4v3' is natively x4. Using outscale={args.outscale} might require specific model weights or adjustments.")
            model_params = {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_conv': 32, 'upscale': int(args.outscale), 'act_type': 'prelu'}
            netscale = int(args.outscale)
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
        else:
            raise ValueError(f"Model {args.model_name} is not a supported SRVGGNetCompact model for this script.")

        # Instantiate models with different enhancement types
        model_orig = SRVGGNetCompact(**model_params, enhancement_type='none')
        model_hist_eq = SRVGGNetCompact(**model_params, enhancement_type='he')
        # Try to instantiate CLAHE model, might fail if scikit-image not installed
        model_clahe = None
        try:
            model_clahe = SRVGGNetCompact(**model_params, enhancement_type='clahe')
        except ImportError as e:
            print(f"Warning: Cannot initialize CLAHE model - {e}")

    except ValueError as e:
        print(f"Error setting up models: {e}")
        return
    except ImportError as e: # Catch import error from SRVGGNetCompact itself
        print(f"Error setting up CLAHE model: {e}")
        # Decide if we should continue without CLAHE or exit
        # For now, we'll print a warning and continue if model_clahe remains None
        pass

    # Determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found. Please download the weights.")

    # --- Upsampler Initialization ---
    common_args = dict(
        scale=netscale,
        model_path=model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id
    )

    print("Initializing Original Upsampler...")
    upsampler_orig = RealESRGANer(model=model_orig, **common_args)

    print("Initializing Histogram Equalization Upsampler...")
    upsampler_hist_eq = RealESRGANer(model=model_hist_eq, **common_args)

    upsampler_clahe = None
    if model_clahe is not None:
        try:
            print("Initializing CLAHE Upsampler...")
            upsampler_clahe = RealESRGANer(model=model_clahe, **common_args)
        except Exception as e:
            print(f"Error initializing CLAHE upsampler: {e}. CLAHE processing will be skipped.")

    # --- Processing Loop ---
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    if not paths:
        print(f"Error: No input images found in {args.input}")
        return

    results_metrics = {'orig': {'psnr': [], 'ssim': []}, 'hist_eq': {'psnr': [], 'ssim': []}, 'clahe': {'psnr': [], 'ssim': []}}
    processed_count = 0

    print("\n" + "-" * 50)
    print(f"Processing HR images from: {args.input}")
    print(f"Saving SR results to:    {args.output}")
    print(f"Upscale factor:          {args.outscale}")
    print(f"Using FP16:              {not args.fp32}")
    print("-" * 50 + "\n")


    for idx, path in enumerate(paths):
        imgname, input_extension = os.path.splitext(os.path.basename(path))
        print(f"Processing {idx + 1}/{len(paths)}: {imgname}{input_extension}")

        # --- Load HR Image ---
        img_hr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_hr is None:
            print(f"  Warning: Skipping invalid image: {path}")
            continue

        h_hr, w_hr = img_hr.shape[:2]
        print(f"  HR dimensions: {w_hr}x{h_hr}")

        # --- Downsample HR to get LR ---
        scale = 1 / args.outscale
        img_lr = cv2.resize(img_hr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h_lr, w_lr = img_lr.shape[:2]
        print(f"  LR dimensions: {w_lr}x{h_lr}")

        # Check if LR image is valid
        if h_lr == 0 or w_lr == 0:
             print(f"  Warning: Downsampling resulted in zero dimension for {path}. Skipping.")
             continue


        # --- Inference: Upsample LR to SR ---
        output_orig, output_hist_eq, output_clahe = None, None, None

        # Original Model
        try:
            output_orig, _ = upsampler_orig.enhance(img_lr, outscale=args.outscale)
        except RuntimeError as error:
            print(f'  Error during Original inference: {error}')
            print('  If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        except Exception as error:
            print(f'  Error during Original inference: {error}')

        # Histogram Equalization Model
        try:
            output_hist_eq, _ = upsampler_hist_eq.enhance(img_lr, outscale=args.outscale)
        except RuntimeError as error:
            print(f'  Error during Hist Eq inference: {error}')
            print('  If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        except Exception as error:
            print(f'  Error during Hist Eq inference: {error}')

        # CLAHE Model
        if upsampler_clahe:
            try:
                output_clahe, _ = upsampler_clahe.enhance(img_lr, outscale=args.outscale)
            except RuntimeError as error:
                print(f'  Error during CLAHE inference: {error}')
                print('  If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            except Exception as error:
                print(f'  Error during CLAHE inference: {error}')
        else:
            print("  CLAHE upsampler not available. Skipping.")

        # --- Save SR outputs ---
        output_extension = 'png' # Use PNG for potentially better quality saving

        if output_orig is not None:
            save_path_orig = os.path.join(args.output, f'{imgname}_sr_orig.{output_extension}')
            cv2.imwrite(save_path_orig, output_orig)
            # print(f"  Saved: {save_path_orig}")
        else:
            print("  Failed to generate Original SR output.")

        if output_hist_eq is not None:
            save_path_hist_eq = os.path.join(args.output, f'{imgname}_sr_hist_eq.{output_extension}')
            cv2.imwrite(save_path_hist_eq, output_hist_eq)
            # print(f"  Saved: {save_path_hist_eq}")
        else:
             print("  Failed to generate Hist Eq SR output.")

        if output_clahe is not None:
            save_path_clahe = os.path.join(args.output, f'{imgname}_sr_clahe.{output_extension}')
            cv2.imwrite(save_path_clahe, output_clahe)
            # print(f"  Saved: {save_path_clahe}")
        elif upsampler_clahe:
             print("  Failed to generate CLAHE SR output.")

        # --- Evaluate SR against HR ---
        evaluated = False
        # Compare Original SR vs HR
        if output_orig is not None:
            try:
                psnr_orig, ssim_orig = calculate_metrics(img_hr, output_orig)
                if ssim_orig is not None: # Check if SSIM calculation was successful
                    results_metrics['orig']['psnr'].append(psnr_orig)
                    results_metrics['orig']['ssim'].append(ssim_orig)
                    print(f"  Metrics Orig    vs HR -> PSNR: {psnr_orig:.4f}, SSIM: {ssim_orig:.4f}")
                    evaluated = True
                else:
                    results_metrics['orig']['psnr'].append(psnr_orig) # Still record PSNR
                    print(f"  Metrics Orig    vs HR -> PSNR: {psnr_orig:.4f}, SSIM: N/A")
                    evaluated = True # Counted as partially evaluated
            except ValueError as e:
                 print(f"  Error comparing Orig vs HR: {e}")
            except Exception as e:
                print(f"  Unexpected error during Orig vs HR comparison: {e}")

        # Compare Hist Eq SR vs HR
        if output_hist_eq is not None:
            try:
                psnr_he, ssim_he = calculate_metrics(img_hr, output_hist_eq)
                if ssim_he is not None: # Check if SSIM calculation was successful
                    results_metrics['hist_eq']['psnr'].append(psnr_he)
                    results_metrics['hist_eq']['ssim'].append(ssim_he)
                    print(f"  Metrics Hist Eq vs HR -> PSNR: {psnr_he:.4f}, SSIM: {ssim_he:.4f}")
                    evaluated = True
                else:
                    results_metrics['hist_eq']['psnr'].append(psnr_he) # Still record PSNR
                    print(f"  Metrics Hist Eq vs HR -> PSNR: {psnr_he:.4f}, SSIM: N/A")
                    evaluated = True # Counted as partially evaluated
            except ValueError as e:
                print(f"  Error comparing Hist Eq vs HR: {e}")
            except Exception as e:
                print(f"  Unexpected error during Hist Eq vs HR comparison: {e}")

        # Compare CLAHE SR vs HR
        if output_clahe is not None:
            try:
                psnr_cl, ssim_cl = calculate_metrics(img_hr, output_clahe)
                if ssim_cl is not None:
                    results_metrics['clahe']['psnr'].append(psnr_cl)
                    results_metrics['clahe']['ssim'].append(ssim_cl)
                    print(f"  Metrics CLAHE   vs HR -> PSNR: {psnr_cl:.4f}, SSIM: {ssim_cl:.4f}")
                    evaluated = True # Count even if only one model works
                else:
                    results_metrics['clahe']['psnr'].append(psnr_cl)
                    print(f"  Metrics CLAHE   vs HR -> PSNR: {psnr_cl:.4f}, SSIM: N/A")
                    evaluated = True
            except ValueError as e:
                 print(f"  Error comparing CLAHE vs HR: {e}")
            except Exception as e:
                print(f"  Unexpected error during CLAHE vs HR comparison: {e}")

        if evaluated:
            processed_count += 1
        print("-" * 20) # Separator between images


    # --- Calculate and Print Average Metrics ---
    print("\n" + "-" * 50)
    print("Average Metrics:")
    print("-" * 50)

    if processed_count > 0:
        # Calculate averages safely
        avg_psnr_orig = np.mean(results_metrics['orig']['psnr']) if results_metrics['orig']['psnr'] else 0
        avg_ssim_orig = np.mean(results_metrics['orig']['ssim']) if results_metrics['orig']['ssim'] else 0 # Excludes None/NaN

        avg_psnr_he = np.mean(results_metrics['hist_eq']['psnr']) if results_metrics['hist_eq']['psnr'] else 0
        avg_ssim_he = np.mean(results_metrics['hist_eq']['ssim']) if results_metrics['hist_eq']['ssim'] else 0 # Excludes None/NaN

        avg_psnr_clahe = np.mean(results_metrics['clahe']['psnr']) if results_metrics['clahe']['psnr'] else 0
        avg_ssim_clahe = np.mean(results_metrics['clahe']['ssim']) if results_metrics['clahe']['ssim'] else 0 # Excludes None/NaN

        print(f"Original Upscaler vs HR ({processed_count} images):")
        print(f"  Average PSNR: {avg_psnr_orig:.4f}")
        print(f"  Average SSIM: {avg_ssim_orig:.4f}")
        print("-" * 20)
        print(f"Hist Eq Upscaler vs HR ({processed_count} images):")
        print(f"  Average PSNR: {avg_psnr_he:.4f}")
        print(f"  Average SSIM: {avg_ssim_he:.4f}")
        print("-" * 20)
        print(f"CLAHE Upscaler vs HR ({processed_count} images):")
        print(f"  Average PSNR: {avg_psnr_clahe:.4f}")
        print(f"  Average SSIM: {avg_ssim_clahe:.4f}")
    else:
        print("No images were successfully processed and evaluated.")

    print("-" * 50)
    print("Processing finished.")


if __name__ == '__main__':
    # Removed sys.path manipulation assuming standard package structure or running from correct dir
    main() 