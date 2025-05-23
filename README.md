# Adaptive Image Preprocessing for IOPA X-rays

This repository contains my submission for the Dobble AI Internship Assignment. The core objective was to develop an **adaptive preprocessing pipeline** for Intraoral Periapical (IOPA) dental X-rays in DICOM/RVG format, tailored to handle the inherent variability in medical imaging.

Rather than relying on fixed filters or blanket enhancement strategies, the focus here was to **analyze the quality of each image and dynamically adjust preprocessing steps** to normalize the images for better performance in downstream AI pipelines (like lesion detection, bone analysis, etc.).

---

## 🧠 Problem Understanding

Dental X-rays, particularly IOPA scans, are highly susceptible to inconsistencies introduced by:

* Varying exposure conditions
* Operator-dependent factors
* Machine types and manufacturers

This results in fluctuations in **brightness, contrast, sharpness**, and **noise levels**, all of which can **negatively impact AI model generalizability** if not handled.

A fixed preprocessing approach may under-process some images while over-processing others. Thus, an **adaptive strategy**—that inspects image quality and chooses preprocessing steps accordingly—was critical for robust performance.

---

## 🗃️ Dataset

Since the focus is on **image preprocessing rather than model training**, only a **small prototype dataset** of IOPA scans in `.dcm` format was used.

* **Format**: DICOM (`.dcm`)
* **Source**: Provided/uploaded manually
* **Size**: Small (\~10–20 images) — sufficient for validating the pipeline logic

All images were:

1. Parsed using `pydicom`
2. Extracted as pixel arrays
3. Normalized and converted to `uint8` for compatibility with OpenCV and matplotlib

This format alignment was necessary for applying conventional enhancement methods and evaluating them visually and quantitatively.

---

## ⚙️ Methodology

### 📊 Image Quality Metrics: The Core of Adaptiveness

To determine the preprocessing needs of an image, four **key image quality metrics** were computed:

| Metric         | Why it Matters                                      | How it was Computed                      |
| -------------- | --------------------------------------------------- | ---------------------------------------- |
| **Brightness** | Affects feature visibility                          | Mean of pixel values                     |
| **Contrast**   | Influences differentiation of anatomical structures | Standard deviation of pixel intensities  |
| **Sharpness**  | Detects focus and edge clarity                      | Variance of Laplacian                    |
| **Noise**      | Impacts downstream segmentation accuracy            | Difference-of-Gaussian residual analysis |

> 📌 These metrics form the **decision logic** of the adaptive pipeline. High noise → more denoising. Low contrast → contrast enhancement. And so on.

---

### 📌 Static Preprocessing (Baseline)

As a control experiment, a **static pipeline** was implemented to assess what a fixed set of enhancements might do. This included:

* **Histogram Equalization** — Good for general brightness redistribution.
* **CLAHE** — Local contrast improvement while avoiding over-amplification.
* **Gaussian Denoising** — To reduce high-frequency noise.

While effective in some cases, this pipeline sometimes:

* Was too aggressive (over-enhanced well-exposed images)
* Missed subtle improvements needed in darker images
* Applied denoising even when sharpness was already low

This made the case for **conditional**, metric-based processing stronger.

---

### 🤖 Adaptive Preprocessing Approaches

#### 🔹 Adaptive Approach 1: Heuristic Rule-Based

This approach defines **rules based on thresholds** derived from quality metrics:

```python
if brightness < 80:
    apply_gamma_correction()
if contrast < 25:
    apply_CLAHE()
if noise > 10:
    apply_denoising()
if sharpness < 150:
    apply_sharpening()
```

**Why heuristics?**

* Easy to interpret and tune
* Quick to prototype
* Avoids overfitting on small data

This method gives fine-grained control while keeping transparency in preprocessing logic.

---

#### 🔹 Adaptive Approach 2: Composite Adjustment

The second adaptive method (`dynamic_adaptive_preprocessing`) focuses on balancing multiple metrics rather than acting on one trigger at a time.

This is more realistic for edge cases like:

* Low contrast + high noise
* Moderate brightness but low sharpness

The logic here considers cross-dependencies among metrics and assigns composite enhancement strategies using fuzzy thresholds and weighted decisions.

---

## ✅ Results & Evaluation

### 🔬 Quantitative Evaluation Metrics

* **PSNR (Peak Signal-to-Noise Ratio)**:

  * Measures reconstruction quality (higher is better)
  * Useful to detect over-denoising or compression artifacts

* **SSIM (Structural Similarity Index)**:

  * Measures perceived visual similarity
  * Closer to 1 indicates better preservation of structure

These metrics were calculated for:

* Raw vs. Static Preprocessed
* Raw vs. Adaptive 1
* Raw vs. Adaptive 2

### 🖼️ Visual Comparison

Representative outputs below show side-by-side differences between the static and adaptive results.

Visual clarity and contrast were notably improved in the adaptive pipelines while preserving anatomical features like root canals and bone lines.

![download](https://github.com/user-attachments/assets/b040701c-499d-4928-bbf9-2188925fe9f8)

---

## 🔍 Discussion

### 🔧 Key Challenges

* **DICOM Normalization**:

  * Some pixel arrays had non-standard value ranges (e.g., 0–4000), needing rescaling.

* **OpenCV Compatibility**:

  * Most OpenCV functions expect 8-bit images, so careful normalization was required.

* **Metric Interference**:

  * For example, denoising reduces sharpness; so applying both requires balancing trade-offs.

### 💡 Lessons & Successes

* Rule-based adaptiveness is a good first step before ML-based pipelines.
* CLAHE consistently outperformed basic histogram equalization.
* Dynamic preprocessing produced more visually stable and artifact-free images.

---

## 🚀 Future Work

* Replace rule-based logic with a lightweight ML model trained to predict preprocessing needs.
* Use real-world datasets and evaluate downstream AI performance post-preprocessing.
* Add optional:

  * Gamma correction
  * Illumination normalization (e.g., Homomorphic filtering)
* Batch processing for entire study directories
* Build an interactive UI (e.g., Streamlit or Gradio) to allow dental practitioners to explore adjustments.

---

## 🏥 Clinical Relevance

Poor preprocessing leads to:

* **False negatives**: Missed pathologies due to low contrast or over-smoothing
* **False positives**: Artifacts enhanced to look like caries or lesions
* **Misdiagnosis**: Inconsistent input → Inconsistent model behavior

This pipeline ensures input uniformity, helping AI models focus on clinical patterns rather than low-level noise or exposure artifacts.

---

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/adaptive-iopa-preprocessing.git
cd adaptive-iopa-preprocessing
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
# Open in Jupyter or VS Code
jupyter notebook adaptive_pipeline.ipynb
```

Or convert to script and run:

```bash
jupyter nbconvert --to script adaptive_pipeline.ipynb
python adaptive_pipeline.py --dicom_path path/to/image.dcm
```

---

## 📁 Repository Structure

```bash
adaptive-iopa-preprocessing/
│
├── adaptive_pipeline.ipynb           # Main notebook with all preprocessing logic
├── sample_dicom/                     # DICOM test images
├── outputs/                          # Enhanced outputs and comparison images
├── images/                           # Visuals used in README/report
├── requirements.txt
└── README.md                         # This file (project report)
```

---

## 🗓️ Submission Notes

* ✅ **Assignment Deadline**: May 24, 2025
* 📦 All components self-contained in this notebook
* 💬 Open to feedback/discussion in follow-up stages
