import cv2
import numpy as np
from od import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0
distance_threshold = 20  # Distance to consider when matching objects

def update_tracking_objects(center_points_cur_frame):
    """Update tracking objects by matching current and previous frame centers."""
    global tracking_objects, track_id

    tracking_objects_copy = tracking_objects.copy()
    center_points_cur_frame_copy = center_points_cur_frame.copy()

    # Iterate over existing tracked objects to see if they still exist in the current frame
    for object_id, pt2 in tracking_objects_copy.items():
        object_exists = False
        for pt in center_points_cur_frame_copy:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < distance_threshold:
                # Object exists, update its position
                tracking_objects[object_id] = pt
                object_exists = True
                if pt in center_points_cur_frame:
                    center_points_cur_frame.remove(pt)
                break

        if not object_exists:
            # Object was lost, remove it from tracking
            tracking_objects.pop(object_id)

    # Add new IDs for unmatched objects
    for pt in center_points_cur_frame:
        tracking_objects[track_id] = pt
        track_id += 1

def detect_objects(frame):
    """Detect objects in the frame."""
    class_ids, scores, boxes = od.detect(frame)
    center_points_cur_frame = []

    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return center_points_cur_frame

def draw_tracking(frame):
    """Draw tracked objects with unique IDs."""
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Detect objects in the current frame
    center_points_cur_frame = detect_objects(frame)

    # Update tracking objects based on the current frame's detections
    if count <= 2:
        # In the first few frames, just initialize the tracking objects
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1
    else:
        # Update tracking based on distance between previous and current frame centers
        update_tracking_objects(center_points_cur_frame)

    # Draw the tracked objects on the frame
    draw_tracking(frame)

    # Show the processed frame
    cv2.imshow("Frame", frame)

    # Update previous frame center points
    center_points_prev_frame = center_points_cur_frame.copy()

    # Wait for the user to press 'Esc' to exit
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
