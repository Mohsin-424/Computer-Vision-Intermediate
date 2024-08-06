import os
import cv2

video_path = os.path.join('.', 'data', 'people.mp4')

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('Error in video playing')
    exit()

print("Video is opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print('No more frames to detect')
        break  # Use break instead of exit to ensure proper cleanup
    
    cv2.imshow('Frame', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting")
        break

cap.release()
cv2.destroyAllWindows()
