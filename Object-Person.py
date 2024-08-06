import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

video_path = os.path.join('.' , 'data' , 'people.mp4')

cap = cv2.VideoCapture(video_path)



model = YOLO('yolov8n.pt')
tracker = Tracker

colors = [(random.randint(0,255) , random.randint(0,255) , random.randint(0,255)) for j in range(10)]
if not cap.isOpened():
    print('Error in video playing ')
    exit()

print("Video is opened Sucessfully")
while True:
    ret, frame = cap.read()
    if not ret:
        print('No more frames to detect')
        exit()
    
    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolists():
            # print(r)
            # to unwrap information
            x1 , y1, x2, y2 ,score , class_id = r
            x1 = int(x1)
            x2= int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            detections.append([x1, y1,  x2 , y2 , score])
            
        tracker.update( frame , detections )
        
        for track in tracker.tracks:
            bbox = track.bbox
            track_id = track.track_id
            # Eveery id will be colored different
            cv2.rectangle(frame , (x1 ,y1) , (x2, y2) , (colors[track_id % len(colors)]) , 3)
               
    
            
            
              
        
    cv2.imshow( 'Frame ' , frame )
    
    
    if cv2.waitKey(1)& 0xFF == ord('q'):
        print("Exiting")
        break
        
cap.release()
cv2.destroyAllWindows()