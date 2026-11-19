# Alarm-level evaluation of YOLO detectors for safety helmet non-wearing

Code, data split and raw results for the article:

> EL-BACHA R. *From Reactive to Proactive Workplace Safety: Alarm-Level Evaluation of YOLO Detectors for Real-Time Detection of Safety Helmet Non-Wearing.* Submitted to PeerJ Computer Science.

## Contents

| Path | Description |
|---|---|
| `helmet_v2_experiments.ipynb` | Google Colab notebook: dataset download, VOC → YOLO conversion (classes `helmet`, `head`), 70/15/15 split, training of YOLOv5s(u), YOLOv8s and YOLO11s, test evaluation, alarm-level evaluation, figures, optional video test |
| `split.csv` | Assignment of each of the 5,000 images (`stem`) to `train`, `val` or `test` (seed 42). Reused for all models |
| `results/table_dataset.csv` | Number of images, helmet boxes and bare-head boxes per split |
| `results/table_test_metrics.csv` | Box-level test metrics (P, R, mAP50, mAP50-95, per class), parameters, epochs, FPS on GPU and CPU |
| `results/table_alarm_test.csv` | Alarm-level test metrics with the threshold selected on the validation set |
| `results/alarm_sweep_val_<model>.csv` | Alarm metrics on the validation set for thresholds 0.05–0.95 |
| `results/training/<model>/results.csv` | Per-epoch training and validation logs |
| `results/training/<model>/args.yaml` | Full training configuration |
| `figures/` | Figures of the article (300 dpi) |

## Data

The images are not redistributed here. They come from the public Kaggle dataset
[Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) (5,000 images, Pascal VOC).
The notebook downloads them automatically, and `split.csv` reproduces the exact split.

## Reproducing the results

1. Open `helmet_v2_experiments.ipynb` in Google Colab and select a T4 GPU runtime.
2. Run the cells in order. Outputs are written to `MyDrive/helmet_v2` and training resumes automatically after a disconnection.
3. To reuse the published split, copy `split.csv` into `MyDrive/helmet_v2/` before running the conversion cell.

Environment: Ultralytics 8.4.215, PyTorch, Google Colab (NVIDIA T4).

## Alarm definition

An image is a violation if it contains at least one annotated bare head. An alarm is raised if the detector predicts at least one bare head with confidence ≥ *t*. The threshold *t* maximizing the alarm F1-score is selected on the validation set and applied unchanged to the test set.

## License

Code: MIT License. Results and figures: CC BY 4.0.

## Contact

Rachid EL-BACHA, Faculty of Science and Technology, Sultan Moulay Slimane University, Beni-Mellal, Morocco. rachid.elbacha@usms.ac.ma
