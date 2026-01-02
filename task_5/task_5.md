## Task 5 
Polyp Detection in Colonoscopy Videos

## Objective: 
Detecting Polyps from Colonoscopy Videos using YOLOv111n-seg

## Datasets Used:
1. https://arxiv.org/pdf/2409.01437
2. https://www.kaggle.com/datasets/debeshjha1/polypgen-video-sequence

## Methodology:
- Uses `yolo11n-seg.pt` model for object detection and segmentation. 
- The Kvasir Dataset [1] was used to train the YOLO model. 
- The dataset was first converted into YOLO supported format
- Then the YOLO model was trained.
- The frames of the video were split and joined at 25 frames per second for detecting polyps. 

## Output Videos 1:
[![Watch the video](outputs/0.jpg)](outputs/colonoscopy_polyp_detection.mp4)

## Output Video:
[![Watch the video](outputs/1.jpg)](outputs/output1.mp4)

## Validation:
### Batch 1:
| Prediction  | Labels |
|----------|--------|
| ![Alt text](validation/val_batch0_pred.jpg) | ![Alt text](validation/val_batch0_labels.jpg)  |

### Batch 2:
| Prediction  | Labels |
|----------|--------|
| ![Alt text](validation/val_batch1_pred.jpg) | ![Alt text](validation/val_batch1_labels.jpg)  |


## Evaluation Metrics:

| F1 Confidence Curve  | BoxP Curve |
|----------|--------|
| ![Alt Text](metrics/BoxF1_curve.png) | ![Alt Text](metrics/BoxP_curve.png)  |


| BoxPR Curve  | BoxR Curve |
|----------|--------|
| ![Alt Text](metrics/BoxPR_curve.png) | ![Alt Text](metrics/BoxR_curve.png)  |


| Confusion Matrix  | Normalized Confusion MAtrix |
|----------|--------|
| ![Alt Text](metrics/confusion_matrix.png) | ![Alt Text](metrics/confusion_matrix_normalized.png)  |

| Mask F1 Curve  | MaskP Curve |
|----------|--------|
| ![Alt Text](metrics/MaskF1_curve.png) | ![Alt Text](metrics/MaskP_curve.png)  |

| Mask PR Curve | MaskP Curve |
|----------|--------|
| ![Alt Text](metrics/MaskPR_curve.png) | ![Alt Text](metrics/MaskR_curve.png)  |

| Results |
|---------|
| ![Alt Text](metrics/results.png) | 


## Dependencies

- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics
- natsort

Install:
```bash
pip install torch opencv-python ultralytics natsort

```

## References
1. [1] S. Gautam, A. Storås, C. Midoglu, S. A. Hicks,
    V. Thambawita, P. Halvorsen, and M. A. Riegler,
    "Kvasir-VQA: A Text-Image Pair GI Tract Dataset,"
    arXiv:2409.01437, 2024.

---


