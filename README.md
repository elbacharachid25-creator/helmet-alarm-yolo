# Alarm-level evaluation of YOLO detectors for safety helmet non-wearing

## Title

**From reactive to proactive workplace safety: alarm-level evaluation of deep learning object detectors for real-time detection of safety helmet non-wearing**

Code, data split and raw results accompanying the article submitted to *PeerJ Computer Science* (author: Rachid EL-BACHA, Sultan Moulay Slimane University, Beni-Mellal, Morocco).

Archived version: https://doi.org/10.5281/zenodo.24377426

## Description

Deep learning detectors of safety helmets are usually evaluated with box-level metrics (precision, recall, mAP). A deployed monitoring system, however, takes a different decision: for each scene, it raises an alarm or not. This repository contains everything needed to reproduce a study that evaluates helmet monitoring at the level of this decision:

- a public dataset of 5,000 construction-site images is relabelled into two classes, `helmet` and `head` (bare head = violation);
- three detectors, YOLOv5s (Ultralytics YOLOv5su), YOLOv8s and YOLO11s, are trained under identical settings on a fixed 70/15/15 training/validation/test split;
- the detectors are evaluated at box level on the test set;
- an image-level alarm rule is defined, the alarm threshold of each detector is selected on the validation set (maximum alarm F1-score) and applied unchanged to the test set.

Main result: the detector ranked first by mAP (YOLO11s) produced the weakest alarm, whereas YOLOv8s detected 128 of 139 violation images with 16 false alarms among 611 compliant images (alarm F1-score 0.905).

## Dataset Information

- **Source:** Hard Hat Detection dataset, Kaggle, https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection (third-party dataset; see the Kaggle page for its licence terms).
- **Content:** 5,000 images of workers, mainly on construction sites, annotated with bounding boxes in Pascal VOC (XML) format. Original classes: `helmet`, `head`, `person`.
- **Classes used in this study:** `0: helmet`, `1: head` (visible head without a helmet). The `person` class is discarded.
- **Split:** 70% training (3,500 images), 15% validation (750), 15% test (750), random shuffle with seed 42. The assignment of every image is given in `split.csv`.
- **Composition after relabelling:**

| Set | Images | Helmet boxes | Bare-head boxes | Images with ≥1 bare head |
|---|---|---|---|---|
| Training | 3,500 | 13,422 | 4,055 | 638 |
| Validation | 750 | 2,763 | 831 | 143 |
| Test | 750 | 2,781 | 899 | 139 |
| Total | 5,000 | 18,966 | 5,785 | 920 |

- **Redistribution:** the images are **not** redistributed in this repository. The notebook downloads them from Kaggle, and `split.csv` reproduces the exact sets.

## Code Information

| Path | Description |
|---|---|
| `helmet_v2_experiments.ipynb` | Complete pipeline as a Google Colab notebook (sections below) |
| `split.csv` | Columns `stem`, `split`: assignment of each image (file name without extension) to `train`, `val` or `test` |
| `requirements.txt` | Python dependencies |
| `results/table_dataset.csv` | Number of images and boxes per split |
| `results/table_test_metrics.csv` | Box-level test metrics: P, R, mAP50, mAP50-95, per-class P/R/AP, parameters, epochs, FPS on GPU and CPU |
| `results/table_alarm_test.csv` | Alarm-level test metrics with the threshold selected on the validation set |
| `results/alarm_sweep_val_<model>.csv` | Alarm metrics on the validation set for thresholds 0.05 to 0.95 |
| `results/training/<model>/results.csv` | Per-epoch training and validation losses and metrics |
| `results/training/<model>/args.yaml` | Full training configuration used by Ultralytics |
| `figures/` | Figures of the article (300 dpi PNG) |
| `LICENSE` | MIT licence for the code |

Notebook sections:

1. **Setup** – installs the libraries, mounts Google Drive, defines paths, seed (42), image size (416) and the three models.
2. **Download dataset** – downloads the Kaggle dataset with `kagglehub`.
3. **Convert VOC → YOLO and split** – conversion, class filtering, 70/15/15 split (reused from `split.csv` if present), `data.yaml`, dataset table.
4. **Train the three models** – identical settings, early stopping, automatic resume after a Colab disconnection.
5. **Test-set evaluation** – box-level metrics per class, parameters, GPU and CPU speed.
6. **Alarm-level evaluation** – threshold sweep on the validation set, selection of the threshold maximizing alarm F1, evaluation on the test set.
7. **Figures** – threshold curves and model comparison.
8. **(Optional) Alarm on a video** – applies the alarm rule (bare head in at least K of the last N frames) to a user video.
9. **Summary** – prints the tables and zips the results.

## Usage Instructions

1. Open `helmet_v2_experiments.ipynb` in Google Colab (File → Upload notebook) and select **Runtime → Change runtime type → T4 GPU**.
2. To reuse the exact split of the article, create the folder `MyDrive/helmet_v2/` in Google Drive and copy `split.csv` into it.
3. Run the cells in order. If the Kaggle download asks for credentials, add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab Secrets (key icon).
4. Outputs are written to `MyDrive/helmet_v2/`. If Colab disconnects, re-run the setup, download and conversion cells, then the training cell: training resumes from the last saved epoch.
5. Total run time is about 3–5 hours on a T4 GPU.

To load a trained model and run the alarm rule on one image:

```python
from ultralytics import YOLO
model = YOLO('MyDrive/helmet_v2/runs/YOLOv8s/weights/best.pt')
r = model.predict('image.jpg', imgsz=416, conf=0.35, verbose=False)[0]
alarm = bool((r.boxes.cls == 1).any())   # class 1 = bare head
```

## Requirements

- Python 3 (Google Colab runtime, Ubuntu Linux)
- `ultralytics==8.4.215` (installs PyTorch, OpenCV, NumPy, etc.)
- `kagglehub`, `pandas`, `matplotlib`, `Pillow`
- GPU: NVIDIA T4 (Google Colab) for training; CPU speed measured on the Colab CPU runtime

Install outside Colab with:

```bash
pip install -r requirements.txt
```

## Methodology

1. **Preprocessing:** Pascal VOC boxes converted to YOLO format (class, normalized centre x, centre y, width, height); only `helmet` and `head` kept; box coordinates clipped to the image and boxes smaller than 1 pixel discarded; images without kept objects receive an empty label file; images resized by the framework to 416 × 416 pixels (letterbox) with pixel values scaled to [0, 1].
2. **Split:** 70/15/15 with seed 42, saved in `split.csv` and reused for all models.
3. **Training:** COCO-pretrained YOLOv5su, YOLOv8s and YOLO11s; batch 16; up to 100 epochs with early stopping (patience 20); AdamW selected automatically (initial learning rate 1.67 × 10⁻³); mosaic, HSV, translation, scaling and horizontal-flip augmentation on the training set only; checkpoint with the best validation fitness retained.
4. **Box-level evaluation:** precision, recall, AP50 and AP50-95 per class on the test set (IoU ≥ 0.5 for a true positive); FPS one image at a time on GPU and CPU.
5. **Alarm-level evaluation:** an image is a violation if it contains at least one annotated bare head; an alarm is raised if at least one bare head is predicted with confidence ≥ t. The threshold t is varied from 0.05 to 0.95 (step 0.05) on the validation set; the value maximizing the alarm F1-score is applied unchanged to the test set, where TP, missed violations, false alarms, precision, recall, F1 and false alarm rate are reported.

## Citations

If you use this code or these results, please cite the article (citation to be updated after publication) and the archive:

> EL-BACHA, R. (2026). From reactive to proactive workplace safety: alarm-level evaluation of deep learning object detectors for real-time detection of safety helmet non-wearing — code and results. Zenodo. https://doi.org/10.5281/zenodo.24377426

Dataset: Larxel (andrewmvd). Hard Hat Detection. Kaggle. https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

Software: Jocher, G., Chaurasia, A., & Qiu, J. Ultralytics YOLO. https://github.com/ultralytics/ultralytics

## License & Contribution Guidelines

- Code: MIT License (see `LICENSE`).
- Results and figures: CC BY 4.0.
- The images belong to the original dataset and are subject to its own licence.

Contributions are welcome: please open an issue to report a problem or propose a change, or submit a pull request with a short description of the modification.

## Contact

Rachid EL-BACHA, Department of Chemistry and Environment, Faculty of Science and Technology, Sultan Moulay Slimane University, Beni-Mellal, Morocco. rachid.elbacha@usms.ac.ma
