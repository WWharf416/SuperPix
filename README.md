# SuperPix
Image Super Resolution using traditional Computer Vision and Deep Learning Techniques
# SuperPix: Image Super Resolution Tool

SuperPix is an advanced image super-resolution application that enhances low-resolution images using both traditional computer vision techniques and state-of-the-art deep learning models.

## Features
    
- **Multiple Super Resolution Models**:
  - SRCNN (Super-Resolution Convolutional Neural Network)
  - Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)
    - Standard upscaling without preprocessing
    - Histogram Equalization (HE) preprocessing for improved contrast
    - Contrast Limited Adaptive Histogram Equalization (CLAHE) for balanced enhancement
  - Histogram Equalization (HE) for contrast enhancement
  - Interpolation with Bilinear filtering for smoother upscaling

- **Interactive Web Interface**:
  - Built with Streamlit for an intuitive user experience
  - Side-by-side comparison of original and enhanced images
  - Quality metrics to evaluate enhancement performance

- **Enhancement Options**:
  - Adjustable upscaling factors
  - Face enhancement capabilities
  - GPU acceleration support

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SuperPix.git
   cd SuperPix
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   ```
   cd ESRGAN/Real-ESRGAN
   pip install -e .
   cd ../../
   ```

3. Run the application:
   ```
   cd streamlit_app
   streamlit run app.py
   ```

## Requirements

- Python 3.7+
- PyTorch
- TensorFlow
- OpenCV
- Streamlit
- Other dependencies listed in requirements.txt

## Project Structure

- `streamlit_app/`: Contains the web application code
  - `app.py`: Main Streamlit application
  - `srcnn.py`: SRCNN model implementation
  - `esrgan_helpers.py`: Helper functions for Real-ESRGAN models
- `ESRGAN/`: Contains ESRGAN model files and weights
- `SRCNN/`: Contains SRCNN model files and weights

## How It Works

SuperPix uses deep learning models trained to reconstruct high-resolution details from low-resolution images. The application automatically downloads model weights when first used and provides a user-friendly interface for image enhancement.

Apart from this the repository also gives options to explore traditional Computer Vision solutions to image super resolution like Histogram Equalization and Interpolation.

## Acknowledgments

- Real-ESRGAN: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- SRCNN implementation based on the paper "Image Super-Resolution Using Deep Convolutional Networks" by Dong et al.
