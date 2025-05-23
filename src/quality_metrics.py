
def get_brightness(image):
    return np.mean(image)

def get_contrast(image):
    return np.std(image)

def get_sharpness(image):
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.Laplacian(image_uint8, cv2.CV_64F).var()

def get_noise(image):
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(image_uint8, (3, 3), 0)
    noise = image_uint8 - blur
    return np.var(noise)
