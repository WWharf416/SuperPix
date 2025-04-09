import cv2
import numpy as np
from scipy import ndimage
from io import BytesIO
from PIL import Image
import random

# --- Helper Functions ---

def random_float(min_val, max_val):
    return random.uniform(min_val, max_val)

def random_int(min_val, max_val):
    # Ensure max_val is not less than min_val
    if max_val < min_val:
        max_val = min_val
    return random.randint(min_val, max_val)

# --- Blur Degradations ---

def gaussian_blur(img, kernel_min=5, kernel_max=21, sigma_min=0.1, sigma_max=5.0):
    """Applies Gaussian Blur with random kernel size and sigma."""
    kernel_size = random_int(kernel_min // 2, kernel_max // 2) * 2 + 1 # Ensure odd kernel size
    sigma = random_float(sigma_min, sigma_max)
    # Ensure kernel size is positive
    kernel_size = max(1, kernel_size)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def motion_blur(img, size_min=5, size_max=25, angle_min=-45, angle_max=45):
    """Applies Motion Blur with random size and angle."""
    size = random_int(size_min, size_max)
    angle = random_float(angle_min, angle_max)

    # Create motion blur kernel
    kernel = np.zeros((size, size))
    # Ensure center calculation is correct, especially for size=1
    if size == 1:
        center = 0
    else:
        center = (size - 1) / 2

    radian = np.deg2rad(angle)
    dx = np.cos(radian)
    dy = np.sin(radian)

    # Simplified line drawing for kernel - ensure integer coordinates
    pt1_x = int(round(center - dx * center))
    pt1_y = int(round(center - dy * center))
    pt2_x = int(round(center + dx * center))
    pt2_y = int(round(center + dy * center))

    cv2.line(kernel, (pt1_x, pt1_y), (pt2_x, pt2_y), 1.0, thickness=1)

    # Normalize the kernel, avoid division by zero
    kernel_sum = kernel.sum()
    if kernel_sum == 0:
        kernel[int(center), int(center)] = 1.0 # Use identity kernel if sum is zero
    else:
        kernel /= kernel_sum

    return cv2.filter2D(img, -1, kernel)

# --- Noise Degradations ---

def gaussian_noise(img, sigma_min=1, sigma_max=30):
    """Adds Gaussian noise with random standard deviation."""
    sigma = random_float(sigma_min, sigma_max)
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = img.astype(np.float32) + noise # Work with float32 for noise addition
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def poisson_noise(img):
    """Adds Poisson noise (approximated). Assumes input is uint8."""
    # Approximation: Convert image intensity to simulate photon counts, apply Poisson, scale back.
    # Higher intensity pixels get more noise variance.
    # This is a simplified model.
    # Ensure img is float for calculations
    img_float = img.astype(np.float32)
    # Determine scaling factor, avoiding potential log2(0)
    max_val = np.max(img_float)
    if max_val <= 0: # Handle black images
        return img # No noise to add
    # Scale factor based on max value to simulate 'counts'
    scale = 100 / max_val # Adjust 100 for noise level
    noisy_img_scaled = np.random.poisson(img_float * scale)
    noisy_img = noisy_img_scaled / scale
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    """Adds Salt and Pepper noise."""
    noisy_img = np.copy(img)
    h, w = img.shape[:2]
    num_pixels = img.size // img.shape[2] if len(img.shape) == 3 else img.size

    # Salt noise (white pixels)
    num_salt = int(salt_prob * num_pixels)
    salt_coords_h = np.random.randint(0, h, num_salt)
    salt_coords_w = np.random.randint(0, w, num_salt)
    if len(img.shape) == 3:
        noisy_img[salt_coords_h, salt_coords_w, :] = 255
    else:
        noisy_img[salt_coords_h, salt_coords_w] = 255

    # Pepper noise (black pixels)
    num_pepper = int(pepper_prob * num_pixels)
    pepper_coords_h = np.random.randint(0, h, num_pepper)
    pepper_coords_w = np.random.randint(0, w, num_pepper)
    if len(img.shape) == 3:
        noisy_img[pepper_coords_h, pepper_coords_w, :] = 0
    else:
        noisy_img[pepper_coords_h, pepper_coords_w] = 0

    return noisy_img.astype(img.dtype)


# --- Resize Degradations ---

def random_resize(img, scale_min=0.15, scale_max=1.5):
    """Resizes image with a random scale factor using random interpolation."""
    scale_factor = random_float(scale_min, scale_max)
    h, w = img.shape[:2]
    new_h, new_w = int(round(h * scale_factor)), int(round(w * scale_factor))

    # Prevent zero dimensions
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    interp_methods = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4
    ]
    interpolation = random.choice(interp_methods)

    # Handle potential edge case where new dims are same as old
    if new_h == h and new_w == w:
        return img.copy() # No resize needed

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return resized_img.astype(img.dtype)

# TODO: Add Sinc filter resize here (more complex, involves FFT or specific kernel construction)
# def sinc_resize(...)

# --- Compression Degradations ---

def jpeg_compression(img, quality_min=30, quality_max=95):
    """Applies JPEG compression with random quality factor."""
    quality = random_int(quality_min, quality_max)
    # Ensure image is in BGR uint8 format for Pillow
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Assume input is BGR, convert to RGB for Pillow
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    buffer = BytesIO()
    img_pil.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    img_jpeg_pil = Image.open(buffer)
    img_jpeg_rgb = np.array(img_jpeg_pil)
    # Convert back to BGR
    img_jpeg_bgr = cv2.cvtColor(img_jpeg_rgb, cv2.COLOR_RGB2BGR)
    return img_jpeg_bgr.astype(img.dtype)

# --- Color Degradations ---

def random_grayscale(img, probability=0.1):
     """ Randomly convert image to grayscale with a given probability """
     if random.random() < probability and len(img.shape) == 3 and img.shape[2] == 3:
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         # Stack grayscale channel to mimic 3 channels for consistency
         return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(img.dtype)
     return img

# --- Other potential additions ---
# Chromatic Aberration
# Lens Distortion
# Vignetting
# Haze/Fog
# Film Grain
# ... etc 