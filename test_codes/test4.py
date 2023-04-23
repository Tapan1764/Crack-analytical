import cv2

# Load the image
image = cv2.imread('C:/Users/Tapan Khokhariya/Desktop/1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to create a binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours
for contour in contours:
    # Approximate the contour with a polygon
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    
    # Check if the polygon has 4 sides (i.e., a rectangle)
    if len(approx) == 4:
        # Draw a green rectangle around the contour
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        
        # Calculate the length of the rectangle sides
        side1 = cv2.norm(approx[0]-approx[1])
        side2 = cv2.norm(approx[1]-approx[2])
        
        # Print the length of the shorter side
        if side1 < side2:
            print("Crack length:", side1)
        else:
            print("Crack length:", side2)
        
# Display the image
cv2.imshow('Concrete Beam', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
