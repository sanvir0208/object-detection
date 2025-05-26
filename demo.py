
import torch
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import numpy as np
from model import YOLOModel
import torchvision.ops as ops

# Config
GRID_SIZE = 7
NUM_CLASSES = 20
IMG_SIZE = 224
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_image(image_path_or_url):
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path_or_url).convert('RGB')
        return image
    except Exception as e:
        print(f"Failed to load image: {image_path_or_url} -> {e}")
        return None

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image_tensor.unsqueeze(0)

def decode_predictions(preds, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    preds = preds.squeeze(0)
    boxes = []
    confidences = []
    class_ids = []

    cell_size = 1.0 / GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell_pred = preds[i, j]
            objectness = cell_pred[0].item()
            if objectness < conf_threshold:
                continue

            x_cell, y_cell, w, h = cell_pred[1:5].tolist()
            class_probs = cell_pred[5:]
            class_id = torch.argmax(class_probs).item()
            class_conf = class_probs[class_id].item()
            conf = objectness * class_conf
            if conf < conf_threshold:
                continue

            cx = (j + x_cell) * cell_size
            cy = (i + y_cell) * cell_size
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(class_id)

    if len(boxes) == 0:
        return []

    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(confidences)
    keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    results = []
    for idx in keep:
        results.append({
            'class_id': class_ids[idx],
            'class_name': VOC_CLASSES[class_ids[idx]],
            'confidence': confidences[idx]
        })

    # Sort results by confidence in descending order
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results

def plot_boxes(image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    img_w, img_h = image.size

    for box in boxes:
        x1, y1, x2, y2 = [coord * img_w if i % 2 == 0 else coord * img_h for i, coord in enumerate(box['bbox'])]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{box['class_name']} {box['confidence']:.2f}",
                color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLOModel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('weights/yolo_best.pth', map_location=device))
    model.eval()

    for path in image_paths:
        print(f"\nRunning object detection on: {path}")
        image = load_image(path)
        if image is None:
            continue

        input_tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

        detections = decode_predictions(preds.cpu())
        if detections:
            print("Detected objects:")
            for detection in detections:
                print(f" - {detection['class_name']}: {detection['confidence']:.2f}")
        else:
            print("No objects detected.")

if __name__ == "__main__":
    main()
