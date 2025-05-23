# DISPLAY NORMALIZED DICOM IMAGES
import pydicom
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load DICOM
dicom_path = "/kaggle/input/dicom-data/Images/IS20250115_171841_9465_61003253.dcm"
dicom_data = pydicom.dcmread(dicom_path)
pixel_array = dicom_data.pixel_array

# Normalize image
norm_img = cv2.normalize(pixel_array.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Show image
plt.figure(figsize=(6,6))
plt.imshow(norm_img, cmap='gray')
plt.title("DICOM Image")
plt.axis("off")
plt.show()

# Print key metadata
print({
    "Patient ID": dicom_data.get("PatientID", "N/A"),
    "Modality": dicom_data.get("Modality", "N/A"),
    "Dimensions": pixel_array.shape,
    "Bits Stored": dicom_data.get("BitsStored", "N/A"),
    "Photometric Interpretation": dicom_data.get("PhotometricInterpretation", "N/A")
})


# FUNCTION TO COMPARE RAW STATIC PROCESSED ADAPTIVE APPROACH 1 AND ADAPTIVE APPROACH 2 IMAGES
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def display_preprocessing_comparison(dicom_path):
    """
    Given a DICOM file path, displays:
    - Raw image
    - Static preprocessed image
    - Adaptive preprocessed image (Approach 1)
    - Adaptive preprocessed image (Approach 2)
    """
    # Step 1: Load DICOM image
    dicom_data = pydicom.dcmread(dicom_path)
    raw_image = dicom_data.pixel_array.astype(np.float32)

    # Step 2: Preprocessing
    static_image = static_preprocessing_pipeline(raw_image)

    # Adaptive 1
    adaptive1_image = adaptive_preprocess(raw_image)

    # Adaptive 2
    brightness = get_brightness(raw_image)
    contrast = get_contrast(raw_image)
    sharpness = get_sharpness(raw_image)
    noise_level = get_noise(raw_image)

    adaptive2_image = dynamic_adaptive_preprocessing(raw_image, brightness, contrast, sharpness, noise_level)

    # Step 3: Visualization
    images = [raw_image, static_image, adaptive1_image, adaptive2_image]
    titles = ["Raw Image", "Static Preprocessing", "Adaptive 1", "Adaptive 2"]

    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

