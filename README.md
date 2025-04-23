# Brick Kiln Detection using Vision-Language and Object Detection Models

## Introduction

This repository presents a comprehensive framework for detecting brick kilns across South Asia using state-of-the-art object detection and vision-language models. The project integrates multi-source satellite imagery and diverse model architectures for robust detection across multiple regions.

## Related Work

### Comparison to Existing Datasets

Existing datasets for environmental monitoring often focus on specific geographies or lack scalability. Our dataset expands the scope by covering multiple states and includes varied brick kiln types, offering a large-scale, diverse, and validated benchmark for object detection.

## Dataset & Framework

### Satellite Datasets
- **Sentinel-2**: Free, multispectral satellite data with 10–20m resolution.
- **ESRI Basemap**: Used for annotation reference.

### Manual labeling Data Coverage
- Delhi-NCR
- Lucknow-Airshed
- Dhaka-Airshed
- WB-small region

### Predict Data Coverage

- India - IGP
- Bangladesh
- Pakisthan
- Afganisthan

### Data Downloading and Processing
All tiles were downloaded using a custom bulk downloader. Each tile was:
- Normalized and aligned
- Cropped to manageable sizes
- Annotated and validated

### Annotation Process
Annotations were generated using:
- Weak supervision (e.g., pre-trained detection models)
- Manual validation with regional experts
- Final verification using high-resolution maps

Annotation categories:
- Zigzag
- FCBK 
- CFCBK 

## Modelling – Choosing the Best Model

We tested and benchmarked the following models:

### Two-Stage Detectors
- Faster R-CNN
- Oriented RCNN

### One-Stage Detectors
- YOLOv5, YOLOv8
- YOLO-World
- YOLO-OBB (Oriented Bounding Box variant)

### Transformer-Based VLMs
- Florence-2
- DETA
- RHINO
- Grounding DINO
- OWLv2

### Anchor-Free Methods
- R3Det-KLD

## Out-of-Region Generalization

Models were tested on unseen regions to measure their generalization. Evaluation was based on:
- Mean Average Precision (mAP)
- Recall and Precision
- Confusion matrix over kiln types

## Multi-State Brick Kiln Dataset

### Verification Process
- Expert validation
- Cross-check with previous studies
- Manual map matching

### Final Dataset
- Balanced and cleaned
- Sub-region-wise directory structure
- Labelled in both COCO and YOLO format

### External Validation
- Used external maps and field data for evaluation

### Dataset Format
Each region contains:


## Benchmarking of Models on Complete Dataset

| Model         | Type            | AP@0.5 | Notes                           |
|---------------|------------------|--------|--------------------------------|
| YOLOv8        | One-stage        | xx.xx  | Baseline                       |
| Florence-2    | Vision-Language  | xx.xx  | Best zero-shot performance     |
| R3Det-KLD     | Anchor-free      | xx.xx  | Strong on rotated bounding box |
| DETA          | Transformer      | xx.xx  | Lightweight, good mAP          |
| ...           | ....             | ...    | ....                           |

*(Values are placeholders; replace with real results)*

## Applications

- Air quality impact studies
- Compliance enforcement for brick kiln conversion
- Urban planning and policy making
- Seasonal and temporal monitoring

## Limitations and Future Work

- Improve Zigzag subtype classification
- Introduce temporal change detection
- Optimize for low-resource inference

## Conclusion

