# Custom YOLO-style Object Detection Model

This project implements a YOLO-style object detection model trained from scratch using a lightweight CNN backbone. The model detects 20 VOC-style object classes, including people, animals, and household objects.

## Features

- Trained for **90 epochs** on a custom dataset with 20 object classes.
- Performs bounding box regression and object classification.
- Lightweight architecture suitable for moderate compute environments.
- Demo script included for testing on images via URLs or local paths.
- Uses Non-Maximum Suppression (NMS) to reduce duplicate detections.

## Dataset & Classes

The model was trained on Pascal VOC classes:

`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor`

## Training Details

- Number of epochs: 90  
- Input image size: 224x224  
- Optimized bounding box coordinates and class probabilities  
- Implemented in PyTorch  

## Usage

### Requirements

- Python 3.x  
- PyTorch  
- torchvision  
- pillow  
- matplotlib  
- requests  
- numpy

Detected objects:
 - tvmonitor: 0.87
 - vase: 0.84
 - keyboard: 0.69
 - pottedplant: 0.58
 - book: 0.35


Install dependencies with:

```bash
pip install torch torchvision pillow matplotlib requests numpy


