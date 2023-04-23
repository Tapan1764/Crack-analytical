from flask import Flask, render_template, Response
import cv2
import time
import mysql.connector
import os

app = Flask(__name__)

def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    # Set the frame rate to 1 frame per second
    frame_rate = camera.get(cv2.CAP_PROP_FPS)

    # Set the interval to 30 seconds
    interval = 1

    # Initialize the time counter
    time_counter = 0

    # Initialize the frame counter
    frame_counter = 0

    try:

        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Calculate the current time in seconds
            current_time = frame_counter / frame_rate

            # Check if it's time to extract an image
            if current_time >= time_counter + interval:
                # Set the new time counter
                time_counter = current_time
                
                # writing the extracted images
                name = './captured_image/frame' + str(current_time) + '.jpg'
                print('Creating...' + name)

                # Write the image to file
                cv2.imwrite(name, frame)
            
            # increasing counter so that it will
            # show how many frames are created
            frame_counter += 1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
@app.route('/function1', methods=['POST'])
def function1():
    # Execute function 1 code here
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)