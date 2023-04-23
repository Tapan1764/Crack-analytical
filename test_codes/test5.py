import cv2
import numpy as np

# Load the image
img = cv2.imread('D:\\ICT\\Sem 6\\HCD\\data\\frame120.12.jpg', 0)

# Apply Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Apply Canny edge detection to find edges
edges = cv2.Canny(img_blur, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to only include those with area greater than a certain threshold
min_area = 50
contours_filtered = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        contours_filtered.append(cnt)

# Draw contours on original image
img_contours = cv2.drawContours(img, contours_filtered, -1, (0, 255, 0), 2)

# Find the length, width, and depth of each crack
crack_lengths = []
crack_widths = []
crack_depths = []
for cnt in contours_filtered:
    # Find the minimum area rectangle that fits the contour
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Find the length and width of the rectangle
    length = max(rect[1])
    width = min(rect[1])
    
    # Find the depth of the crack by measuring the difference in intensity between the edge of the crack and the surrounding concrete
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    mean_intensities = cv2.mean(img, mask=mask)
    crack_depth = mean_intensities[0]
    
    crack_lengths.append(length)
    crack_widths.append(width)
    crack_depths.append(crack_depth)

# Print the results
for i in range(len(crack_lengths)):
    print("Crack", i+1, "length:", crack_lengths[i])
    print("Crack", i+1, "width:", crack_widths[i])
    print("Crack", i+1, "depth:", crack_depths[i])

cv2.imshow('im',img_contours)
cv2.waitKey(0)