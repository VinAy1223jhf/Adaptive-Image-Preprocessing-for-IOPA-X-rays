from skimage import exposure

def apply_histogram_equalization(image):
    eq_img = exposure.equalize_hist(image)  # Returns float image [0,1]
    eq_img = (eq_img * 255).astype(np.uint8)  # Convert back to uint8
    return eq_img


def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened

def apply_basic_denoising(image):
    median = cv2.medianBlur(image, 3)
    gaussian = cv2.GaussianBlur(median, (3, 3), 0)
    return gaussian

def static_preprocessing_pipeline(image):
    print("Applying static pipeline...")
    
    eq_img = apply_histogram_equalization(image)
    sharp_img = apply_unsharp_mask(eq_img)
    denoised_img = apply_basic_denoising(sharp_img)
    
    return denoised_img
