import cv2
import numpy as np
import tensorflow as tf

# Load the trained models for beam detection and crack detection
beam_model = tf.keras.models.load_model('beam_detection_model.h5')
crack_model = tf.keras.models.load_model('crack_detection_model.h5')

# Load the image and preprocess it
image = cv2.imread('D:\\ICT\\Sem 6\\HCD\\data\\frame30.03.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Detect the concrete beam in the image
beam_prediction = beam_model.predict(image)
if beam_prediction[0][0] > 0.5:
    print("Concrete beam detected")
    # Remove the background of the beam using a mask
    beam_mask = np.argmax(beam_prediction, axis=-1)[0]
    beam_mask = np.expand_dims(beam_mask, axis=-1)
    beam_mask = np.repeat(beam_mask, 3, axis=-1)
    beam_image = image[0] * beam_mask
    beam_image = beam_image.astype(np.uint8)
    
    # Detect cracks in the beam
    crack_prediction = crack_model.predict(beam_image)
    if crack_prediction[0][0] > 0.5:
        print("Crack detected")
        # Find the contours of the crack and calculate its length, width, and depth
        gray_image = cv2.cvtColor(beam_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            length = h
            width = w
            depth = np.min(beam_image[y:y+h, x:x+w, :])
            print("Length:", length)
            print("Width:", width)
            print("Depth:", depth)
    else:
        print("No cracks detected")
else:
    print("No concrete beam detected")
