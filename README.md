# Histopath Pipeline — Automated Ground Truth Generation for Digital Pathology

This repository provides a modular framework for preprocessing, stain augmentation, and automated ground truth mask generation from histopathology images.  It is designed to support the development of machine learning and deep learning models in digital pathology, particularly for H&E-stained tissue sections.

---

## Usage

### Process a Single Image

Run the example script to generate tissue and cell masks for a single image:

```bash
PYTHONPATH=src python examples/groundtruth_generator.py --image images/sample.png
```

Outputs will be written to the `out/` directory.

### Process a Directory of Images

```bash
PYTHONPATH=src python examples/groundtruth_generator.py --dir /path/to/patches
```

Each image will be processed independently, and results will be saved to `out/`.

---

## Workflow Summary Example

| Step | Description | Output |
|------|--------------|---------|
| 1 | Load image | Input RGB histology image |
| 2 | Preprocess | Grayscale conversion, denoising, equalization |
| 3 | Tissue segmentation | Binary tissue/background mask |
| 4 | Stain augmentation | Optional color perturbations |
| 5 | Cell mask generation | HED + binary cell mask |
| 6 | Export results | Final masks and overlays saved to `out/` |

---

## Notes

Example scripts in `examples/` demonstrate how to compose and run complete workflows.  

This project makes use of open-source libraries developed by the digital pathology community, including:

- [HistomicsTK](https://digitalslidearchive.github.io/HistomicsTK/)
- [scikit-image](https://scikit-image.org/)
- [OpenCV](https://opencv.org/)
