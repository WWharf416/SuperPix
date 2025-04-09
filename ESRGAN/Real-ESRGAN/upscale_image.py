import argparse
import cv2
import os
import glob
import torch
from basicsr.utils.download_util import load_file_from_url

# Assuming realesrgan and its dependencies are installed or in the PYTHONPATH
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def main():
    """Upscales images in a folder using both standard and hist_eq Real-ESRGAN models."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image folder path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-general-x4v3', # Defaulting to a SRVGG model
        help=('SRVGG Model names: realesr-animevideov3 | realesr-general-x4v3')
    )
    parser.add_argument('-o', '--output_dir', type=str, default='results_upscaled', help='Base output directory to save results')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--ext',
        type=str,
        default='png',
        help='Output image extension. Options: png | jpg')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # --- Validate Input ---
    if not os.path.isdir(args.input):
        print(f"Error: Input folder not found: {args.input}")
        return

    # --- Create Output Directories ---
    output_dir_orig = os.path.join(args.output_dir, 'original')
    output_dir_hist_eq = os.path.join(args.output_dir, 'hist_eq')
    output_dir_clahe = os.path.join(args.output_dir, 'clahe')
    os.makedirs(output_dir_orig, exist_ok=True)
    os.makedirs(output_dir_hist_eq, exist_ok=True)
    os.makedirs(output_dir_clahe, exist_ok=True)


    # --- Model Setup ---
    args.model_name = args.model_name.split('.')[0]
    model_clahe = None
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

        # Instantiate models
        model_orig = SRVGGNetCompact(**model_params, enhancement_type='none')
        model_hist_eq = SRVGGNetCompact(**model_params, enhancement_type='he')
        try:
            model_clahe = SRVGGNetCompact(**model_params, enhancement_type='clahe')
        except ImportError as e:
            print(f"Warning: Cannot initialize CLAHE model - {e}. CLAHE processing will be skipped.")

    except ValueError as e:
        print(f"Error setting up models: {e}")
        return
    except ImportError as e:
        print(f"Error setting up CLAHE model: {e}. CLAHE processing will be skipped.")
        pass

    # Determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                try:
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
                except Exception as e:
                     print(f"Error downloading model weights from {url}: {e}")
                     print("Please ensure you have internet connectivity or manually download the weights.")
                     return

    if not os.path.isfile(model_path):
        print(f"Error: Model path {model_path} not found. Please download the weights.")
        return

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

    try:
        print("Initializing Original Upsampler...")
        upsampler_orig = RealESRGANer(model=model_orig, **common_args)
        
        print("Initializing Histogram Equalization Upsampler...")
        upsampler_hist_eq = RealESRGANer(model=model_hist_eq, **common_args)

        upsampler_clahe = None
        if model_clahe is not None:
            print("Initializing CLAHE Upsampler...")
            upsampler_clahe = RealESRGANer(model=model_clahe, **common_args)

    except Exception as e:
        print(f"Error initializing upsamplers: {e}")
        # Potential causes: CUDA issues, missing dependencies
        return

    # --- Get list of images --- 
    img_list = sorted(glob.glob(os.path.join(args.input, '*')))
    if not img_list:
        print(f"Error: No images found in input folder: {args.input}")
        return
        
    print(f"Found {len(img_list)} images to process.")

    # --- Processing Loop ---
    total_processed = 0
    for idx, img_path in enumerate(img_list):
        # --- Load Input Image ---
        imgname, input_extension = os.path.splitext(os.path.basename(img_path))
        print(f"\nProcessing {idx + 1}/{len(img_list)}: {imgname}{input_extension}")
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Warning: Could not read image: {img_path}. Skipping.")
            continue

        print(f"  Input dimensions: {img.shape[1]}x{img.shape[0]}")

        # --- Inference and Saving ---
        output_orig, output_hist_eq, output_clahe = None, None, None

        # Original Model
        print("  Upscaling with Original model...")
        try:
            output_orig, _ = upsampler_orig.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print(f'    Error during Original inference: {error}')
            print('    If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        except Exception as error:
            print(f'    Unexpected error during Original inference: {error}')

        # Histogram Equalization Model
        print("  Upscaling with Histogram Equalization model...")
        try:
            # Use a copy if the enhance method potentially modifies the input in-place
            img_copy = img.copy()
            output_hist_eq, _ = upsampler_hist_eq.enhance(img_copy, outscale=args.outscale)
            del img_copy # Free memory if needed
        except RuntimeError as error:
            print(f'    Error during Hist Eq inference: {error}')
            print('    If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        except Exception as error:
            print(f'    Unexpected error during Hist Eq inference: {error}')

        # CLAHE Model
        if upsampler_clahe:
            print("  Upscaling with CLAHE model...")
            try:
                img_copy_cl = img.copy()
                output_clahe, _ = upsampler_clahe.enhance(img_copy_cl, outscale=args.outscale)
                del img_copy_cl
            except RuntimeError as error:
                print(f'    Error during CLAHE inference: {error}')
                print('    If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            except Exception as error:
                print(f'    Unexpected error during CLAHE inference: {error}')
        elif model_clahe is not None:
            print("  CLAHE upsampler failed to initialize. Skipping CLAHE inference.")

        # --- Save outputs ---
        output_extension = args.ext.lower()
        if output_extension not in ['png', 'jpg']:
            print(f"  Warning: Invalid extension '{args.ext}'. Defaulting to png.")
            output_extension = 'png'
        
        # Adjust extension for RGBA if needed (force PNG)
        if len(img.shape) == 3 and img.shape[2] == 4 and output_extension != 'png':
            print("  Input image has alpha channel. Forcing output to PNG format.")
            output_extension = 'png'

        saved_count = 0
        if output_orig is not None:
            save_path_orig = os.path.join(output_dir_orig, f'{imgname}_sr_orig.{output_extension}')
            try:
                 cv2.imwrite(save_path_orig, output_orig)
                 print(f"  Saved Original SR: {save_path_orig}")
                 saved_count += 1
            except Exception as e:
                 print(f"  Error saving Original SR image to {save_path_orig}: {e}")
        else:
            print("  Failed to generate Original SR output. Skipping save.")

        if output_hist_eq is not None:
            save_path_hist_eq = os.path.join(output_dir_hist_eq, f'{imgname}_sr_hist_eq.{output_extension}')
            try:
                cv2.imwrite(save_path_hist_eq, output_hist_eq)
                print(f"  Saved Hist Eq SR:  {save_path_hist_eq}")
                saved_count += 1
            except Exception as e:
                print(f"  Error saving Hist Eq SR image to {save_path_hist_eq}: {e}")
        else:
             print("  Failed to generate Hist Eq SR output. Skipping save.")

        if output_clahe is not None:
            save_path_clahe = os.path.join(output_dir_clahe, f'{imgname}_sr_clahe.{output_extension}')
            try:
                cv2.imwrite(save_path_clahe, output_clahe)
                print(f"  Saved CLAHE SR:    {save_path_clahe}")
                saved_count += 1
            except Exception as e:
                print(f"  Error saving CLAHE SR image to {save_path_clahe}: {e}")
        elif upsampler_clahe:
            print("  Failed to generate CLAHE SR output. Skipping save.")

        if saved_count > 0:
            total_processed += 1 # Count image as processed if at least one output was saved

    print(f"\nProcessing finished. Successfully processed {total_processed} out of {len(img_list)} images.")


if __name__ == '__main__':
    main() 