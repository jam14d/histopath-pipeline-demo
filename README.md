# Histopath Pipeline — Digital Pathology Preprocessing & Stain Augmentation

This pipeline provides an end-to-end preprocessing and segmentation workflow for digital pathology images such as H&E-stained histology slides. It performs realistic stain perturbation augmentation, grayscale preprocessing, and adaptive threshold-based tissue segmentation.

---

## Overview

1. Loads whole-slide or patch-level RGB images.  
2. Optionally generates stain-perturbed versions of each image using HistomicsTK to simulate realistic color variation.  
3. Converts images to grayscale and applies Gaussian blur and optional histogram equalization.  
4. Segments tissue using adaptive or Otsu thresholding to separate tissue from background.  
5. Saves grayscale, mask, overlay, and augmented versions for further analysis or machine learning.

---

## Example Outputs

Example visualizations are stored in `images/`:

| Description | Image |
|--------------|-------|
| Local threshold result | ![Local Threshold](images/local_thresh.png) |
| Stain-augmented RGB | ![Stain Augment](images/stain_augment_rgb.png) |

---

Stain augmentation introduces controlled random variation in color intensity and channel composition.  
This mimics the natural differences caused by slide preparation, staining concentration, and scanner hardware.  
By exposing models to these variations during training, augmentation improves robustness and generalization across datasets and institutions.

Local adaptive (Otsu-like) thresholding adapts the segmentation decision to regional intensity differences instead of relying on one global threshold.  This is useful when slides have uneven illumination, scanner shading, or variable staining.

---

## Command-line usage

Run the script as a Python module from the project root:

```bash
cd histopath-pipeline-demo

python -m histopath_pipeline.tissue_pipeline \
    --image '/Users/jamieannemortel/Downloads/BreaKHis 400X/test/benign/SOB_B_A-14-22549AB-400-001.png' \
    --n_stain_aug 3 \
    --save_dir '/Users/jamieannemortel/Documents/Projects/histopath-pipeline-demo/output' \
    --show
```
To process a directory:

```bash
python -m histopath_pipeline.tissue_pipeline \
    --dir 'PATH' \
    --n_stain_aug 5 \
    --save_dir './output'
```