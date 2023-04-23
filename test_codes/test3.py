import cv2
#import mysql.connector

#mydb = mysql.connector.connect(
#  host="localhost",
#  user="root",
#  password="",
#  database="hcd"
#)

# Create a video capture object for the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to isolate the cracks
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply edge detection algorithm to find the edges of the cracks
    edges = cv2.Canny(thresh, 100, 200)

    # Find the contours of the cracks
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Calculate the length, width, and depth of the cracks
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length1 = w
        width1 = h
        depth1 = cv2.minMaxLoc(gray[y:y+h, x:x+w])[1]

        # Print the crack information
        print("Crack Length:", length1)
        print("Crack Width:", width1)
        print("Crack Depth:", depth1)
	  
        #mycursor = mydb.cursor()
        #sql = "INSERT INTO crack_data (length, width, depth) VALUES (%s, %s, %s)"
        #val = (length1, width1, depth1)
        #mycursor.execute(sql,val)

        #mydb.commit()
	  
        #print(mycursor.rowcount, "record inserted.")

    # Display the resulting image
    cv2.imshow('Crack Detection', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
