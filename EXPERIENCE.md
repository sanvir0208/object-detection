# Experience Report

## 1. Objective
The primary goal was to create an object detection model inspired by YOLO, starting from scratch and without relying on any pretrained weights. The aim was to dive deep into the essential concepts of object detection, including bounding box regression, objectness prediction, and multi-class classification in images.

## 2. Approach
This model employs a lightweight convolutional neural network as its backbone to pull features from images. It segments the image into a 7x7 grid, predicting bounding boxes and class probabilities for each cell, much like YOLO does. We utilized the Pascal VOC dataset, which includes 20 different object classes, and crafted a custom loss function that merges localization, confidence, and classification losses to enhance training.

## 3. Training
Training spanned 90 epochs with a batch size of 8, using the Adam optimizer alongside a cosine annealing learning rate schedule. The dataset was divided into training and validation sets to keep an eye on performance. We saved the best model weights based on the lowest validation loss to avoid overfitting and to promote better generalization.

## 4. Outcome
The model we trained showed decent detection accuracy on test images, successfully recognizing objects like people, dogs, and indoor plants. This project offered hands-on experience in designing custom detection heads, encoding ground truth labels for object detection, and applying non-max suppression to refine results. It also deepened our understanding of the training dynamics involved in object detection tasks.

## 5. Trained Model Weights
The final trained model weights are stored in the file `weights/yolo_best.pth`. These weights encapsulate the learned parameters after 90 epochs of training and can be loaded to make predictions on new images without needing to retrain. The saved weights ensure reproducibility and can be directly used for deployment or further fine-tuning.
