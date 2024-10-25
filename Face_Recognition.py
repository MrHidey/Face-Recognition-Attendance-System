from flask import Flask, render_template, request, redirect, url_for, session, Response
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
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} not found.")
            
            # Generate embeddings for each image using DeepFace with VGG-Face model
            embeddings = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
            if embeddings:
                for embedding in embeddings:
                    known_face_encodings.append(np.array(embedding['embedding']))

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
    try:
        with open(file_path, 'a', newline='') as f:
            lnwriter = csv.writer(f)
            lnwriter.writerow([name, time_str])
            f.flush()  # Ensure data is flushed to disk
            print(f"Written to CSV: {name}, {time_str}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Process each video frame asynchronously for face recognition
def process_frame(frame, file_path):
    global students, present_students

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.30, fy=0.30)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    try:
        # Use DeepFace to generate embeddings for the frame using VGG-Face model
        embeddings = DeepFace.represent(img_path=rgb_small_frame, model_name="VGG-Face", enforce_detection=False)
        if embeddings:
            print("Embeddings generated successfully.")

            # Detect faces in the small frame using OpenCV's face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(rgb_small_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h), embedding in zip(faces, embeddings):
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown

                # Scale back the face coordinates to the original frame size
                x_orig = int(x / 0.30)
                y_orig = int(y / 0.30)
                w_orig = int(w / 0.30)
                h_orig = int(h / 0.30)

                # Draw the red frame around the face
                cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
                cv2.putText(frame, name, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Check if this embedding matches any known faces
                for i, known_embedding in enumerate(known_face_encodings):
                    distance = np.linalg.norm(np.array(embedding['embedding']) - np.array(known_embedding))

                    if distance < 1.1:
                        name = known_face_names[i]
                        color = (0, 255, 0)  # Change to green if recognized
                        print(f"Match found: {name}")
                        print(f"Distance calculated: {distance}")

                        if name in students and name not in present_students:
                            present_students.append(name)
                            now = datetime.now()
                            time_str = now.strftime("%H.%M.%S")
                            print(f"Adding to queue: {name}, {time_str}")
                            file_queue.put((file_path, name, time_str))
                            print(f"Enqueued: {name}, {time_str}")

                        # Update the rectangle color and text to green for recognized faces
                        cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
                        cv2.putText(frame, name, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        else:
            print("No embeddings generated.")
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

        file_path = get_current_date()  # Get file path each time
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

        file_path, name, time_str = item  # Unpack the tuple
        print(f"Writing to CSV from queue: {name}, {time_str}")  # Debugging log
        write_to_csv(file_path, name, time_str)
        print("Write complete.")

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

    # Start the file_writer thread to handle CSV writing asynchronously
    threading.Thread(target=file_writer, daemon=True).start()

    return redirect(url_for('dashboard'))

@app.route('/stop_recognition')
def stop_recognition():
    global recognizing, video_capture, present_students
    recognizing = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    # Ensure we stop the file writer by putting None in the queue
    file_queue.put(None)

    return render_template('attendance.html', present_students=present_students)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
