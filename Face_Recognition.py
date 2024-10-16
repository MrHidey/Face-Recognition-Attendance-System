from flask import Flask, render_template, request, redirect, url_for, session, Response
import face_recognition as fr
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import threading
import queue

app = Flask(__name__)
app.secret_key = 'your_secret_key'

credentials_file = 'credentials.csv'

# Save username and password to the CSV file
def save_credentials(username, password):
    with open(credentials_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password])

# Check if the provided credentials match those in the CSV file
def check_credentials(username, password):
    if os.path.exists(credentials_file):
        with open(credentials_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username and row[1] == password:
                    return True
    return False

# Variables for face recognition
known_face_encodings = []
known_face_names = []
students = []
present_students = []
video_capture = None
recognizing = False
file_queue = queue.Queue()

# Load known face images and their encodings
def load_encode():
    global known_face_encodings, known_face_names, students

    try:
        images = ["Danush.jpg", "Keerthi.jpg", "Sam.jpg", "Yash.jpg"]
        known_face_names = ["Danush", "Keerthi", "Sam", "Yash"]
        known_face_encodings = []

        for img_name in images:
            img_path = os.path.join("uploads", img_name)
            image = fr.load_image_file(img_path)
            encoding = fr.face_encodings(image)[0]
            known_face_encodings.append(encoding)

        students = known_face_names.copy()
    except Exception as e:
        print(f"Error loading images: {e}")

# Get the current date and return the corresponding file path
def get_current_date():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    file_path = f"{current_date}.csv"
    return file_path

# Write attendance data (name, time) to the CSV file
def write_to_csv(file_path, name, time_str):
    with open(file_path, 'a', newline='') as f:
        lnwriter = csv.writer(f)
        lnwriter.writerow([name, time_str])

# Process each video frame asynchronously for face recognition
def process_frames(frame, file_path):
    global students, present_students

    small_frame = cv2.resize(frame, (0, 0), fx=0.30, fy=0.30)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_small_frame)
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        face_distance = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            if name in students and name not in present_students:
                present_students.append(name)
                now = datetime.now()
                time_str = now.strftime("%H.%M.%S")
                file_queue.put((file_path, name, time_str))

# Stream video frames continuously while recognizing faces asynchronously
def generate_frames():
    global video_capture, recognizing

    while recognizing:
        success, frame = video_capture.read()
        if not success:
            break

        if recognizing:
            file_path = get_current_date()
            threading.Thread(target=process_frames, args=(frame, file_path)).start()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Background thread for safely writing attendance data to the CSV
def file_writer():
    while True:
        file_path, name, time_str = file_queue.get()
        if file_path is None:
            break
        write_to_csv(file_path, name, time_str)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if check_credentials(username, password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            save_credentials(username, password)
            session['user'] = username
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        load_encode()
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/start_recognition')
def start_recognition():
    global recognizing, video_capture, present_students
    recognizing = True
    video_capture = cv2.VideoCapture(0)
    present_students.clear()

    if not video_capture.isOpened():
        recognizing = False
        return "Could not open video device", 500

    threading.Thread(target=file_writer, daemon=True).start()

    return redirect(url_for('dashboard'))

@app.route('/stop_recognition')
def stop_recognition():
    global recognizing, video_capture, present_students
    recognizing = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None

    return render_template('attendance.html', present_students=present_students)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
