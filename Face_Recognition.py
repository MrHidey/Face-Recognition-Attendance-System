from flask import Flask, render_template, Response, redirect, url_for
import face_recognition as fr
import cv2
import numpy as np
from datetime import datetime
import csv
import os

app = Flask(__name__)

# Create the uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

video_capture = cv2.VideoCapture(0)

# Global variables for known faces
known_face_encodings = []
known_face_names = []
students_present = set()  # To track present students
f = None
lnwriter = None

def load_encode():
    global known_face_encodings, known_face_names
    
    # Loading known faces and encoding them
    D_image = fr.load_image_file("uploads/Danush.jpg")
    D_encoding = fr.face_encodings(D_image)[0]
    K_image = fr.load_image_file("uploads/Keerthi.jpg")
    K_encoding = fr.face_encodings(K_image)[0]
    S_image = fr.load_image_file("uploads/Sam.jpg")
    S_encoding = fr.face_encodings(S_image)[0]
    Y_image = fr.load_image_file("uploads/Yash.jpg")
    Y_encoding = fr.face_encodings(Y_image)[0]

    # Known encodings and their names
    known_face_encodings = [D_encoding, K_encoding, S_encoding, Y_encoding]
    known_face_names = ["Danush", "Keerthi", "Sam", "Yash"]

def get_current_date():
    global f, lnwriter
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    try:
        f = open(f"uploads/{current_date}.csv", "w+", newline="")
        lnwriter = csv.writer(f)
        lnwriter.writerow(["Name", "Time"])  # Header for CSV
    except Exception as e:
        print(f"Error opening CSV file: {e}")

def process_frame():
    global students_present
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Couldn't read frame from video capture.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame and get face encodings
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            face_distance = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name not in students_present:
                    students_present.add(name)
                    now = datetime.now()
                    time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, time])
                    print(f"Detected: {name} at {time}")  # Log detected students

                    # Display "Present" + student name on the video feed
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    left_corner = (10, 100)
                    font_scale = 1.5
                    font_color = (255, 0, 0)  # Blue text
                    thickness = 3
                    line_type = 2
                    cv2.putText(frame, f"{name} Present", left_corner, font, font_scale, font_color, thickness, line_type)

        # Encode the frame for HTTP streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    load_encode()
    get_current_date()
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global f
    if f is not None:
        f.close()
        print("Closed the CSV file.")
    video_capture.release()
    cv2.destroyAllWindows()
    return "Video capture stopped and file closed."

if __name__ == "__main__":
    app.run(debug=True)
