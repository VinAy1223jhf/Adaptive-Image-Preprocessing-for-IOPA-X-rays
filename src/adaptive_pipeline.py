from .quality_metrics import get_brightness, get_contrast, get_sharpness, get_noise

# adaptive 1 logic using individual functions
import pydicom
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------------------------
# Load and normalize DICOM
# ---------------------------
def load_dicom_image(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)

    # Normalize to 8-bit [0,255]
    image -= np.min(image)
    image /= np.max(image)
    image *= 255.0

    return image.astype(np.uint8)

# ---------------------------
# Image Quality Metrics
# ---------------------------
def get_brightness(image):
    return np.mean(image)

def get_contrast(image):
    return np.std(image)

def get_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_noise(image):
    h, w = image.shape
    roi = image[h//2 - h//20:h//2 + h//20, w//2 - w//20:w//2 + w//20]
    return np.std(roi)

# ---------------------------
# Adaptive Enhancement Steps
# ---------------------------
def adaptive_denoise(image, noise_level):
    if noise_level > 30:
        return cv2.fastNlMeansDenoising(image, h=10)
    elif noise_level > 15:
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return cv2.GaussianBlur(image, (3, 3), 0.5)

def adaptive_contrast(image, contrast_level):
    if contrast_level < 30:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    elif contrast_level < 50:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        return clahe.apply(image)
    else:
        return image

def adaptive_sharpen(image, sharpness_level):
    if sharpness_level > 100:
        return image
    elif sharpness_level > 50:
        amount = 1.0
    else:
        amount = 1.5

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened

# ---------------------------
# Complete Adaptive Pipeline
# ---------------------------
def adaptive_preprocess(image):
    brightness = get_brightness(image)
    contrast = get_contrast(image)
    sharpness = get_sharpness(image)
    noise = get_noise(image)

    print(f"Brightness: {brightness:.2f}, Contrast: {contrast:.2f}, Sharpness: {sharpness:.2f}, Noise: {noise:.2f}")

    denoised = adaptive_denoise(image, noise)
    contrasted = adaptive_contrast(denoised, contrast)
    final = adaptive_sharpen(contrasted, sharpness)

    return final


# adaptive 2 logic
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
from scipy.ndimage import gaussian_filter

def dynamic_adaptive_preprocessing(img, brightness, contrast, sharpness, noise_level):
    """
    Apply preprocessing strategy based on combined image quality metrics.
    """
    img = img.astype(np.uint8)
    processed = img.copy()

    if contrast < 40 and sharpness < 100:
        # Case A: Low contrast, low sharpness
        print("Strategy: Strong CLAHE + Strong Sharpening")
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        processed = unsharp_mask(processed, strength=2.0)

    elif contrast < 40 and noise_level > 20:
        # Case B: Low contrast, high noise
        print("Strategy: Mild CLAHE + Strong Denoising")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        processed = clahe.apply(processed)
        processed = denoise_image(processed, method="nlm")

    elif sharpness < 100 and noise_level < 10:
        # Case C: Low sharpness, low noise
        print("Strategy: Strong Sharpening Only")
        processed = unsharp_mask(processed, strength=2.5)

    else:
        # Case D: Normal/Moderate
        print("Strategy: Mild CLAHE + Gaussian Blur")
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        processed = clahe.apply(processed)
        processed = cv2.GaussianBlur(processed, (3, 3), 0.5)

    return processed

def unsharp_mask(image, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

def denoise_image(image, method="gaussian"):
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 1.0)
    elif method == "median":
        return cv2.medianBlur(image, 3)
    elif method == "nlm":
        image_float = img_as_float(image)
        sigma_est = np.mean(estimate_sigma(image_float, channel_axis=None))
        patch_kw = dict(patch_size=5, patch_distance=6, multichannel=False)
        denoised = denoise_nl_means(image_float, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
        return (denoised * 255).astype(np.uint8)
    return image

