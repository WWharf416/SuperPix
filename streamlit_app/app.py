import streamlit as st
from PIL import Image
import numpy as np
import cv2
from srcnn import process_srcnn
# from skimage.metrics import structural_similarity as ssim # No longer needed
import esrgan_helpers # Import the new helper module
import time
import os

# Page config
st.set_page_config(page_title="SuperPix", layout="wide")

# --- Model Cache ---
# Cache the upsampler initialization to avoid reloading models repeatedly
@st.cache_resource
def get_upsampler(model_name, enhancement_type, outscale, half, gpu_id):
    print(f"Attempting to load model: {model_name}, Enhancement: {enhancement_type}, FP16: {half}")
    start_time = time.time()
    upsampler = esrgan_helpers.initialize_upsampler(
        model_name=model_name,
        enhancement_type=enhancement_type,
        outscale=outscale,
        half=half,
        gpu_id=gpu_id
    )
    end_time = time.time()
    if upsampler:
        print(f"Model loaded in {end_time - start_time:.2f} seconds.")
    else:
        print(f"Failed to load model {model_name} with enhancement {enhancement_type}.")
    return upsampler

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #001f3f; /* Dark Navy */
        color: #ffffff; /* White text */
    }
    .stApp {
        background-color: #001f3f;
        padding: 20px;
    }
    .title {
        text-align: center;
        font-size: 2.7em;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 25px;
    }
    .result-box {
        background-color: #003366; /* Slightly lighter Navy */
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2); /* Darker shadow */
        margin-top: 20px;
        color: #ffffff;
        text-align: center; /* Center align content */
    }
    .result-box img { /* Ensure images don't overflow */
        max-width: 100%;
        height: auto;
        border-radius: 10px; /* Rounded corners for images */
    }
    .metric-label { /* Kept for potential future use, but not displayed now */
        font-size: 1.2em;
        font-weight: bold;
        margin-top: 10px;
        color: #FFD700; /* Gold for labels */
    }
    .stProgress > div > div > div > div { /* Style progress bar */
        background-color: #FFD700; /* Gold */
    }
    .stButton>button { /* Style buttons */
        background-color: #FFD700; /* Gold */
        color: #001f3f; /* Dark Navy Text */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        display: block; /* Make button block */
        margin: 20px auto 0 auto; /* Center button */
    }
    .stButton>button:hover {
        background-color: #FFEC8B; /* Lighter Gold on hover */
        color: #001f3f;
    }
    .stSelectbox div[data-baseweb="select"] > div { /* Style selectbox */
       background-color: #003366;
       color: white;
    }
    .stSelectbox div[role="listbox"] div { /* Style dropdown items */
        background-color: #003366;
        color: white;
    }
    .stSelectbox div[role="listbox"] div:hover {
        background-color: #004080; /* Slightly lighter on hover */
    }
    .uploadedFileName { /* Ensure file name is visible */
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">✨ SuperPix: Image Enhancer ✨</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Processing Options")
uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg", "bmp"])

# Method selection - simplified ESRGAN option
sr_method = st.sidebar.selectbox(
    "Select Super-Resolution Method",
    [
        "SRCNN",
        "ESRGAN (General x4)", # Single option for ESRGAN general model
        "Histogram Equalization (Simple)",
        "Interpolation+Bilinear (Simple)",
        # "Codeformer" # Keep commented if not implemented
    ]
)

upscale_factor = 4 # Keep for potential future use, hardcoded for ESRGAN x4
# use_fp16 = st.sidebar.checkbox("⚡ Use FP16 (Faster, Requires GPU, Default: On)", value=True)
use_fp16 = False
gpu_id_input = st.sidebar.text_input("GPU ID (Optional, e.g., 0)", value="")

if uploaded_file:
    input_image_pil = Image.open(uploaded_file).convert("RGB")  # Ensure RGB PIL
    input_image_np = np.array(input_image_pil) # Convert to NumPy array (RGB)

    # Display original image prominently at the top
    st.markdown("### Original Uploaded Image")
    st.image(input_image_np, use_container_width=True)
    st.markdown("--- Chip Off the Old Block ---")

    # Center the button
    button_placeholder = st.empty()
    if button_placeholder.button("🚀 Enhance Image!"):
        button_placeholder.empty() # Remove button after click
        start_process_time = time.time()
        output_image_std = None # For standard ESRGAN or other methods
        output_image_he = None  # For HE ESRGAN
        output_image_clahe = None # For CLAHE ESRGAN

        # Common setup
        input_image_bgr = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR) # Convert to BGR once for helpers
        gpu_id = None
        if gpu_id_input.strip().isdigit():
            gpu_id = int(gpu_id_input.strip())

        try:
            # Process image based on selected method
            if sr_method == "Histogram Equalization (Simple)":
                st.info("Processing using Simple Histogram Equalization...")
                ycrcb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                y_eq = cv2.equalizeHist(y)
                ycrcb_eq = cv2.merge([y_eq, cr, cb])
                enhanced_bgr = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
                output_image_bgr = cv2.resize(enhanced_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                output_image_std = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB) # Store in std output

            elif sr_method == "Interpolation+Bilinear (Simple)":
                st.info("Processing using Simple Interpolation + Sharpening...")
                upscaled_bgr = cv2.resize(input_image_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                output_image_bgr = cv2.filter2D(upscaled_bgr, -1, sharpening_kernel)
                output_image_std = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB) # Store in std output

            elif sr_method == "SRCNN":
                st.info("Processing using SRCNN...")
                output_image_std = process_srcnn(input_image_np) # Store in std output

            elif sr_method == "ESRGAN (General x4)":
                st.info(f"Processing using {sr_method}. This will enhance using Standard, Histogram Equalization, and CLAHE variants...")
                progress_bar = st.progress(0, text="Initializing...")
                model_name = "realesr-general-x4v3"

                # --- ESRGAN Workflow (Initialize -> Upscale x3) ---
                # 1. Initialize Upsamplers (Cached)
                progress_bar.progress(10, text=f"Loading Standard model...")
                upsampler_std = get_upsampler(model_name, 'none', upscale_factor, half=use_fp16, gpu_id=gpu_id)
                progress_bar.progress(25, text=f"Loading Hist Eq model...")
                upsampler_he = get_upsampler(model_name, 'he', upscale_factor, half=use_fp16, gpu_id=gpu_id)
                progress_bar.progress(40, text=f"Loading CLAHE model...")
                upsampler_clahe = get_upsampler(model_name, 'clahe', upscale_factor, half=use_fp16, gpu_id=gpu_id)

                # Check if any upsampler failed (especially CLAHE if scikit-image is missing)
                if upsampler_clahe is None:
                    st.warning("CLAHE model failed to load (likely missing 'scikit-image'). Skipping CLAHE enhancement.", icon="⚠️")
                if upsampler_std is None or upsampler_he is None:
                     st.error(f"Failed to initialize Standard or Hist Eq ESRGAN upsampler. Check logs or model files.")
                     st.stop()

                # 2. Upscale the Input image with each model
                progress_bar.progress(55, text="Upscaling (Standard)...")
                if upsampler_std:
                    output_std_bgr = esrgan_helpers.upscale_image(upsampler_std, input_image_bgr, outscale=upscale_factor)
                    if output_std_bgr is not None:
                         output_image_std = cv2.cvtColor(output_std_bgr, cv2.COLOR_BGR2RGB)
                    else:
                         st.warning("Standard ESRGAN upscaling failed.", icon="⚠️")

                progress_bar.progress(70, text="Upscaling (Hist Eq)...")
                if upsampler_he:
                     output_he_bgr = esrgan_helpers.upscale_image(upsampler_he, input_image_bgr, outscale=upscale_factor)
                     if output_he_bgr is not None:
                          output_image_he = cv2.cvtColor(output_he_bgr, cv2.COLOR_BGR2RGB)
                     else:
                          st.warning("Hist Eq ESRGAN upscaling failed.", icon="⚠️")

                progress_bar.progress(85, text="Upscaling (CLAHE)...")
                if upsampler_clahe:
                     output_clahe_bgr = esrgan_helpers.upscale_image(upsampler_clahe, input_image_bgr, outscale=upscale_factor)
                     if output_clahe_bgr is not None:
                          output_image_clahe = cv2.cvtColor(output_clahe_bgr, cv2.COLOR_BGR2RGB)
                     else:
                          st.warning("CLAHE ESRGAN upscaling failed.", icon="⚠️")

                progress_bar.progress(100, text="Processing Complete!")
                time.sleep(1) # Keep complete message visible briefly
                progress_bar.empty() # Remove progress bar

            else:
                st.warning(f"Method '{sr_method}' is selected but not yet implemented.")
                output_image_std = input_image_np # Display original if method not implemented

            end_process_time = time.time()
            st.success(f"Processing finished in {end_process_time - start_process_time:.2f} seconds.")

            # --- Display Results --- Adjusted Layout
            st.markdown("### Enhanced Results")

            if sr_method == "ESRGAN (General x4)":
                # Display Original + 3 ESRGAN outputs in 2x2 grid style
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("**Original Input**")
                    st.image(input_image_np, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("**Enhanced (Hist Eq)**")
                    if output_image_he is not None:
                         st.image(output_image_he, use_container_width=True)
                    else:
                         st.warning("Hist Eq output not generated.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("**Enhanced (Standard)**")
                    if output_image_std is not None:
                         st.image(output_image_std, use_container_width=True)
                    else:
                         st.warning("Standard output not generated.")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("**Enhanced (CLAHE)**")
                    if output_image_clahe is not None:
                         st.image(output_image_clahe, use_container_width=True)
                    elif upsampler_clahe is not None:
                         st.warning("CLAHE output not generated.")
                    else:
                         st.info("CLAHE enhancement was skipped (model not loaded).") # Info if skipped due to load fail
                    st.markdown('</div>', unsafe_allow_html=True)

            elif output_image_std is not None: # For non-ESRGAN methods that produced output
                # Display Original vs Enhanced (single output)
                col1, col2 = st.columns(2)
                with col1:
                     st.markdown('<div class="result-box">', unsafe_allow_html=True)
                     st.markdown("**Original Input**")
                     st.image(input_image_np, use_container_width=True)
                     st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    caption = f"**Enhanced ({sr_method})**"
                    st.markdown(caption)
                    st.image(output_image_std, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                 # Handle case where no output was generated for non-ESRGAN methods
                 st.error("Image processing failed to produce an output.")

        except FileNotFoundError as e:
             st.error(f"Error: Model file not found. {e} Ensure weights are downloaded or path is correct.")
        except ImportError as e:
             st.error(f"Error: Missing dependency. {e}. Please check required libraries (e.g., basicsr, scikit-image for CLAHE)." )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            import traceback
            st.code(traceback.format_exc()) # Show full traceback for debugging

# else: # No file uploaded
#     st.info("Please upload an image using the sidebar to get started.")
