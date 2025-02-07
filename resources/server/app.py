# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
from gaze_tracking import GazeTracking
import face_recognition
import datetime
import base64
import numpy as np
import threading
import time
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)  # Allow Vite dev server
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

# Global variables
gaze = GazeTracking()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
webcam = None
stop_thread = False
reference_encoding = None
processing_thread = None
global glb_match_text;
glb_match_text = ""

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def gaze_tracking(frame, gaze):
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""
    isCheating = False

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
        isCheating = True
    elif gaze.is_left():
        text = "Looking left"
        isCheating = True
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    return frame, (text != ""), isCheating, text

def compare_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            return "Same person detected!", False
        else:
            return "Different person", True

    return "", False

def process_webcam():
    global stop_thread, webcam, glb_match_text
    last_time = datetime.datetime.now()

    while not stop_thread:
        success, frame = webcam.read()
        if not success:
            continue

        enc_frame = encode_frame(frame)
        socketio.emit("camera_frame", enc_frame)
        result = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        faces_count = len(faces)
        result['face_count'] = faces_count

        annotated_frame, is_gaze_tracked, is_cheating, gaze_result = gaze_tracking(frame, gaze)
        result['isGazeTracked'] = is_gaze_tracked
        result['gazeResult'] = gaze_result
        if is_cheating:
            result['gazeCheating'] = True
        else:
            result['gazeCheating'] = False

        time_diff = datetime.datetime.now() - last_time
        if time_diff.total_seconds() >= 5:
            match_text, different_person = compare_faces(frame)
            glb_match_text = match_text
            result['isSamePerson'] = not different_person
            last_time = datetime.datetime.now()
            is_cheating = different_person or is_cheating

        is_cheating = is_cheating or (faces_count > 1)
        result['isCheating'] = is_cheating

        # Add face count text to frame
        cv2.putText(annotated_frame, f"Faces: {faces_count}", (90, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, glb_match_text, (90, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        # Encode frame to base64 for sending via WebSocket
        encoded_frame = encode_frame(annotated_frame)
        
        # Emit the results and frame
        socketio.emit('monitoring_result', result)

        socketio.emit("monitoring_result_frame", encoded_frame)

        time.sleep(0.033)  # Approximately 30 FPS

@app.route('/')
def index():
    return render_template('index.html')

def encode_reference_face(image_url):
    global reference_encoding
    try:
        # Fetch image from URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Load image into numpy array
        image = Image.open(BytesIO(response.content))
        reference_image = np.array(image)
        
        # Encode face
        encodings = face_recognition.face_encodings(reference_image)
        if not encodings:
            raise ValueError("No face found in the image")

        reference_encoding = encodings[0]
        emit("encode-face-response", {"status": True, "message": "Face Encoded"})
        print("Encoded Face")  # Optional: Return encoding for verification
    
    except requests.exceptions.RequestException as req_err:
        emit("encode-face-response", {"status": False, "message": req_err})
        print(f"Error fetching image: {req_err}")
    except ValueError as val_err:
        emit("encode-face-response", {"status": False, "message": val_err})
        print(f"Face recognition error: {val_err}")
    except Exception as e:
        emit("encode-face-response", {"status": False, "message": e})
        print(f"Unexpected error: {e}")

@socketio.on("connect")
def handle_connection():
    print("Connected to a client")
    emit("connected", {"connected": True})

@socketio.on("encode-face")
def handle_encodeFace(imageId):
    print(f"LInk: {imageId}")
    encode_reference_face(f"http://localhost:3000/drive/image/{imageId}")

@socketio.on('start_monitoring')
def start_monitoring():
    global webcam, stop_thread, processing_thread
    
    if webcam is None:
        webcam = cv2.VideoCapture(0)
        stop_thread = False
        processing_thread = threading.Thread(target=process_webcam)
        processing_thread.start()
        emit('monitoring_status', {'status': 'started'})

@socketio.on('stop_monitoring')
def stop_monitoring():
    global webcam, stop_thread, processing_thread
    
    if webcam is not None:
        stop_thread = True
        if processing_thread:
            processing_thread.join()
        webcam.release()
        webcam = None
        emit('monitoring_status', {'status': 'stopped'})

if __name__ == '__main__':
    # Initialize reference face encoding
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)