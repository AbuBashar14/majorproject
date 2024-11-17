import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="yolov4.weights", cfg_path="yolov4.cfg"):
        print("Loading Object Detection")
        print("Running OpenCV DNN with YOLOv4")
        
        # Default parameters
        self.nmsThreshold = 0.4  # Non-Maximum Suppression threshold
        self.confThreshold = 0.5  # Confidence threshold
        self.image_size = 608  # Resize image size for YOLOv4

        # Load YOLO Network
        self.net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:  # Check if CUDA is available
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("CUDA not available, using CPU instead.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.model = cv2.dnn_DetectionModel(self.net)

        # Load class names
        self.classes = []
        self.load_class_names()

        # Define random colors for bounding boxes (one for each class)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Set model input parameters (resize and scale the image)
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="classes.txt"):
        """Load class names from a file"""
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()  # Remove extra whitespace
                self.classes.append(class_name)

        # Ensure there are enough colors for the number of classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        return self.classes

    def detect(self, frame):
        """Detect objects in the input frame"""
        # Perform object detection
        class_ids, confidences, boxes = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

        return class_ids, confidences, boxes

    def draw_boxes(self, frame, class_ids, confidences, boxes):
        """Draw bounding boxes and class labels on the frame"""
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            x, y, w, h = box
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            color = self.colors[class_id]
            
            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
