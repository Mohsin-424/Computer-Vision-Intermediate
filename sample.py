import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker  # Ensure this is properly implemented or imported

# Path to video file
video_path = os.path.join('.', 'data', 'people.mp4')

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize tracker
tracker = Tracker()

# Generate random colors for bounding boxes
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

if not cap.isOpened():
    print('Error in video playing')
    exit()

print("Video is opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print('No more frames to detect')
        break

    # Perform object detection
    results = model(frame)

    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = map(int, r[:4]) + [float(r[4])]
            detections.append([x1, y1, x2, y2, score])

    # Update tracker with detections
    tracker.update(frame, detections)

    for track in tracker.tracks:
        bbox = track.bbox
        track_id = track.track_id
        color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)

    # Display frame
    cv2.imshow('Frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting")
        break

# Release resources

cap.release()

cv2.destroyAllWindows()

