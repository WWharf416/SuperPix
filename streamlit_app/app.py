import streamlit as st
from PIL import Image
import numpy as np
import cv2
from srcnn import process_srcnn
from skimage.metrics import structural_similarity as ssim

# Page config
st.set_page_config(page_title="SuperPix", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #001f3f;
        color: #ffffff;
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
    .result-box, .metric-box {
        background-color: #003366;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.1);
        margin-top: 20px;
        color: #ffffff;
    }
    .metric-label {
        font-size: 1.2em;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">SuperPix: Image Enhancer</div>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg", "bmp"])
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB
    input_image_np = np.array(input_image)

    # Method selection
    sr_method = st.selectbox(
        "Select Super-Resolution Method",
        ["Histogram Equalization", "Interpolation+Bilinear", "SRCNN", "ESRGAN", "Codeformer"]
    )

    if st.button("Process Image"):
        # Process image based on selected method
        if sr_method == "Histogram Equalization":
            ycrcb = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            ycrcb_eq = cv2.merge([y_eq, cr, cb])
            enhanced_image = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
            output_image = cv2.resize(enhanced_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        elif sr_method == "Interpolation+Bilinear":
            upscaled = cv2.resize(input_image_np, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            sharpening_kernel = np.array([[0, -1, 0],
                                          [-1, 5, -1],
                                          [0, -1, 0]])
            output_image = cv2.filter2D(upscaled, -1, sharpening_kernel)

        elif sr_method == "SRCNN":
            output_image = process_srcnn(input_image_np)

        else:
            st.warning(f"Method '{sr_method}' is selected but not yet implemented.")
            output_image = input_image_np

        # Convert for metrics
        input_bgr = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR) if output_image.shape[2] == 3 else output_image

        original_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)

        # Resize if mismatched
        if original_gray.shape != output_gray.shape:
            output_gray = cv2.resize(output_gray, (original_gray.shape[1], original_gray.shape[0]))

        # Calculate metrics
        psnr_value = cv2.PSNR(original_gray, output_gray)
        ssim_value = ssim(original_gray, output_gray)

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.image(input_image_np, caption="🖼️ Original Image", use_container_width=False, width=450)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.image(output_image, caption="📈 Enhanced Image", use_container_width=False, width=450)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display metrics
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown('<h4>📊 Quality Metrics</h4>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">🔍 PSNR:</div> <p>{psnr_value:.2f} dB</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">🔍 SSIM:</div> <p>{ssim_value:.4f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
