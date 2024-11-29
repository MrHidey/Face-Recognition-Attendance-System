from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from deepface import DeepFace
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
uploads_folder = 'uploads'
allowed_extensions = {'png', 'jpg', 'jpeg'}

# Ensure uploads folder exists
os.makedirs(uploads_folder, exist_ok=True)

# Check if the provided credentials match those in the CSV file
def check_credentials(username, password):
    if os.path.exists(credentials_file):
        with open(credentials_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username and row[1] == password:
                    return True
    return False

# Save new credentials to the CSV file
def save_credentials(username, password):
    if not os.path.exists(credentials_file):
        with open(credentials_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Username', 'Password'])  # Write header row only once

    with open(credentials_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password])  # Write the username and password

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

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
        known_face_encodings = []
        known_face_names = []

        for filename in os.listdir(uploads_folder):
            if allowed_file(filename):
                img_path = os.path.join(uploads_folder, filename)
                name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the name
                
                embeddings = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
                if embeddings:
                    for embedding in embeddings:
                        known_face_encodings.append(np.array(embedding['embedding']))
                    known_face_names.append(name)

        students = known_face_names.copy()
    except Exception as e:
        print(f"Error loading images: {e}")

# Route for adding a new student
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['photo']

        if not name or not file:
            flash("Name and photo are required.", "error")
            return redirect(url_for('add_student'))

        if not allowed_file(file.filename):
            flash("Invalid file type. Allowed types: png, jpg, jpeg.", "error")
            return redirect(url_for('add_student'))

        filename = f"{name}.jpg"
        file_path = os.path.join(uploads_folder, filename)
        file.save(file_path)

        # Reload encodings
        load_encode()
        flash("Student added successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template('add_student.html')


# Get the current date and return the corresponding file path
def get_current_date():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    file_path = f"{current_date}.csv"
    return file_path

# Write attendance data (name, time) to the CSV file
def write_to_csv(file_path, name, time_str):
    try:
        with open(file_path, 'a', newline='') as f:
            lnwriter = csv.writer(f)
            lnwriter.writerow([name, time_str])
            f.flush()  # Ensure data is flushed to disk
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Process each video frame asynchronously for face recognition
def process_frame(frame, file_path):
    global students, present_students

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    try:
        embeddings = DeepFace.represent(img_path=rgb_small_frame, model_name="VGG-Face", enforce_detection=False)
        if embeddings:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(rgb_small_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h), embedding in zip(faces, embeddings):
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown

                x_orig = int(x / 0.25)
                y_orig = int(y / 0.25)
                w_orig = int(w / 0.25)
                h_orig = int(h / 0.25)

                cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
                cv2.putText(frame, name, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for i, known_embedding in enumerate(known_face_encodings):
                    distance = np.linalg.norm(np.array(embedding['embedding']) - np.array(known_embedding))

                    if distance < 1:
                        name = known_face_names[i]
                        color = (0, 255, 0)  # Change to green if recognized

                        if name in students and name not in present_students:
                            present_students.append(name)
                            now = datetime.now()
                            time_str = now.strftime("%H.%M.%S")
                            file_queue.put((file_path, name, time_str))

                        cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
                        cv2.putText(frame, name, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except Exception as e:
        print(f"Error in face recognition: {e}")

# Stream video frames continuously while recognizing faces asynchronously
def generate_frames():
    global video_capture, recognizing

    while recognizing:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture frame")
            break

        file_path = get_current_date()
        process_frame(frame, file_path)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Background thread for safely writing attendance data to the CSV
def file_writer():
    while True:
        item = file_queue.get()
        if item is None:
            break  # Stop the loop if None is received

        file_path, name, time_str = item
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
            flash("Invalid username or password.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('register'))

        if check_credentials(username, password):
            flash("Username already exists!", "error")
            return redirect(url_for('register'))

        save_credentials(username, password)
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        load_encode()
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/start_recognition')
def start_recognition():
    global recognizing, video_capture, present_students, students

    recognizing = True
    video_capture = cv2.VideoCapture(0)
    present_students.clear()

    if not video_capture.isOpened():
        recognizing = False
        return "Could not open video device", 500

    threading.Thread(target=file_writer, daemon=True).start()

    # Clear previous absent list and calculate it dynamically when recognition stops
    absent_students = list(set(students) - set(present_students))

    return redirect(url_for('dashboard'))


@app.route('/stop_recognition')
def stop_recognition():
    global recognizing, video_capture, present_students, students

    recognizing = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None

    # Calculate absent students dynamically
    absent_students = list(set(students) - set(present_students))
    file_queue.put(None)

    return render_template('attendance.html', present_students=present_students, absent_students=absent_students)


@app.route('/attendance/<student_name>')
def attendance(student_name):
    file_path = get_current_date()
    attendance_data = []
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == student_name:
                    attendance_data.append({'date': row[1], 'status': 'Present'})
    if attendance_data:
        return jsonify({'attendance': attendance_data})
    else:
        return jsonify({'attendance': [], 'message': 'No attendance found'}), 404


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
