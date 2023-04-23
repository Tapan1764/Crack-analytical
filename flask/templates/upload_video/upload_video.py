import sys

# get path of uploaded file
file_path = sys.argv[1]

# do some processing on the uploaded file
print("Processing file:", file_path)

# for example, you can use OpenCV to read and display the image/video
import cv2
import numpy as np

if file_path.lower().endswith('.mp4'):
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
       
