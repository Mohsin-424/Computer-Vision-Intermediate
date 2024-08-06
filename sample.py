import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker


# Paths for input and output videos
video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'out.mp4')

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame to get the video dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    cap.release()
    exit()

# Initialize video writer
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Tracker()

# Generate random colors for bounding boxes
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# Detection threshold
detection_threshold = 0.5

while ret:
    # Perform object detection
    results = model(frame)

    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

    # Update tracker with detections
    tracker.update(frame, detections)

    # Draw bounding boxes and track IDs
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    # Write the frame with detections to the output video
    cap_out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()
cap_out.release()
cv2.destroyAllWindows()
