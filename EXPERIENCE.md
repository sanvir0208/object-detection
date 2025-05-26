# Experience Report

## 1. Goal
The chief aim was to create an object detection model from scratch in a YOLO-like fashion without using any pretrained weights. That is to say, it was really meant to understand and implement core concepts of object detection on images, such as bounding box regression, objectness prediction, and multi-class classification. 

## 2. Description
The object detection architecture consisted of a lightweight CNN backbone for feature extraction; it treated the image as a grid of 7x7 cells, predicting bounding boxes and class probabilities for each cell, just like in YOLO. The Pascal VOC dataset with 20 object classes was utilized for training. A custom loss function, covering losses for localization, confidence, and classification, was used for finding the optimum during training. 

## 3. Training
Training ran for 90 epochs, with an 8-batch size optionally using Adam with cosine annealing steps to reduce the learning rate. Training and validation were split to assess performance. The best weight of the model was saved for the lowest validation loss so it would not overfit and generalize.

## 4. Conclusion
Testing images showed the model to have reasonable detection accuracy; it detected objects like person, dog, and indoor plants. This project allowed for hands-on experience with designing custom detection heads, encoding ground truth labels for object detection, and applying non-max suppression to refine results.
