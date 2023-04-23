import cv2
import time
import os

cap = cv2.VideoCapture("http://172.20.10.9:4747/video")

timer = time.time() + 10

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

while True:
    ret, frame = cap.read()
    if ret:
        if time.time() >= timer:
                
            # writing the extracted images
            name = './data/frame' + str(timer) + '.jpg'
            print('Creating...' + name)

            # Write the image to file
            cv2.imwrite(name, frame)
            timer = time.time() + 10
            
        cv2.imshow("frame", frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
