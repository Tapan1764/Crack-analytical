import time
import cv2
import mysql.connector
import os

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="hcd"
)

folder_path = "data"
f1 = ""

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".gif"):
        # f1 = os.path.join(folder_path, filename)
        f1 = filename
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thresholding
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Edge detection
        edges = cv2.Canny(thresh, 190, 200)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(edges, kernel)
        erode = cv2.erode(dilate, kernel)

        # Contour detection
        contours, hierarchy = cv2.findContours(
            erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #print(contours)

        img2 = cv2.drawContours(img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite("processed_image/im_"+f1+".jpeg",img2)
        time.sleep(5)
        length_a = []
        width_a = []
        depth_a = []
        # Measurement
        for i in range(len(contours)):
            
            # Length of the crack
            length = cv2.arcLength(contours[i], True)

            # Width of the crack
            x, y, w, h = cv2.boundingRect(contours[i])
            width = w

            # Depth of the crack (assuming uniform depth)
            depth = cv2.minMaxLoc(img[y:y+h, x:x+w])[1]

            if (length > 0):
                # print("Crack {}: Length: {}, Width: {}, Depth: {}".format(
                #     i+1, length, width, depth))
                length_a.append(length)
                width_a.append(width)
                depth_a.append(depth)
                
        folder_path1 = "processed_image"
        f2 = ""
        for filename1 in os.listdir(folder_path1):
            if filename1.endswith(".jpg") or filename1.endswith(".png") or filename1.endswith(".jpeg") or filename1.endswith(".gif"):
                # f2 = os.path.join(folder_path1, filename1)
                f2 = filename1
                
        mycursor = mydb.cursor()
        sql = "INSERT INTO upload_image (original_image, crack_detected, force_applied, length, width, status) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (f1, f2, "-" , round(max(length_a), 2), round(max(width_a), 2), 1)
        mycursor.execute(sql,val)

        mydb.commit()

# print("Max Length: ", length_a)
# print("Max Width: ", width_a)
# print("Max Length: ", max(length_a))
# print("Max Width: ", max(width_a))
# cv2.imshow("im", img2)

cv2.waitKey(0)
