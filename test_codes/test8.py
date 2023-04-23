import cv2
import numpy as np

# Load the image
img = cv2.imread('crack_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a threshold to convert the image to binary
thresh_value = 150
ret, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

# Apply morphological operations to fill in gaps and remove noise
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours in the image
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Get the contour with the maximum area
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Fit a line to the contour using least squares
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)

# Calculate the depth of the crack as the distance between the top and bottom of the crack
crack_depth = abs(righty - lefty)

# Display the result
cv2.imshow('Crack image', img)
cv2.imshow('Thresholded image', thresh)
cv2.imshow('Closed image', closing)
cv2.drawContours(img, [max_contour], 0, (0, 255, 0), 2)
cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 255), 2)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Crack depth:', crack_depth)
