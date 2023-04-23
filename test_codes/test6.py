import cv2
import numpy as np

# Load the image
img = cv2.imread('D:\\ICT\\Sem 6\\HCD\\data\\frame30.03.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to smooth the image and remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply binary thresholding using adaptive thresholding
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw the contours
contour_img = img.copy()

# Find the concrete beam contours
beam_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    area = cv2.contourArea(contour)
    if aspect_ratio > 1.5 and area > 5000:
        beam_contours.append(contour)
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Create a mask of the concrete beams
beam_mask = np.zeros_like(gray)
cv2.drawContours(beam_mask, beam_contours, -1, 255, -1)

# Apply bitwise_and to mask the original image
masked = cv2.bitwise_and(img, img, mask=beam_mask)

# Find the bounding box of the concrete beams
x, y, w, h = cv2.boundingRect(beam_mask)

# Increase the bounding box size by 10% for better cropping
x -= int(w * 0.1)
y -= int(h * 0.1)
w += int(w * 0.2)
h += int(h * 0.2)

# Crop the image
cropped = masked[y:y+h, x:x+w]

# Save the output image
cv2.imshow('output_image', cropped)
cv2.waitKey(0)
