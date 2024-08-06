import os
import cv2

video_path = os.join('.', 'people.mp4')

cap = cv2.VideoCapture()

ret, frame = cap.read()
while ret:
    cv2.imshow( 'frame ' , frame )
    
    cv2.waitKey()
    
    
cap.release()
cv2.destroyAllWindows()
