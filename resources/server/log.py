import json
import cv2
from gaze_tracking import GazeTracking
import face_recognition
import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

global lastTime;
global prev_result;
lastTime = datetime.datetime.now() 
prev_result = ""

def gaze_tracking(frame, gaze):
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    isCheating = False

    text = ""

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

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    isEyesDetected = False

    if left_pupil and right_pupil:
        isEyesDetected = True

    return isEyesDetected, isCheating, text

def analyzeFrame(frame):
    global lastTime;
    global prev_result;
    result = {}
    isEyeBallDetected, isCheating, eyeAction = gaze_tracking(frame, gaze)
    result['isEyeBallDetected'] = isEyeBallDetected
    result['isCheating'] = isCheating
    result['eyeAction'] = eyeAction

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Display the number of faces detected
    faces_count = len(faces)
    result['face_count'] = faces_count

    timeDiff = datetime.datetime.now() - lastTime
    timeDiffSeconds = timeDiff.total_seconds()

    if timeDiffSeconds >= 5:
        isFaceDetected, isSameFace = compare_faces(frame)
        result['isFaceDetected'] = isFaceDetected
        result['isSameFace'] = isSameFace
        if isFaceDetected and not isSameFace:
            result['isCheating'] = True
        lastTime = datetime.datetime.now()

        # Display the match text and face count text on the frame
    result = json.dumps(result)
    return result


def startTracking(stream):
    
    global frame

    while True:
        # We get a new frame from the webcam
        _, frame = stream.read()
        
        result = analyzeFrame(frame)
        if(result != prev_result):
            print(result)
        if cv2.waitKey(1) == 27:
            break

    stream.release()
    cv2.destroyAllWindows()

def encodeReferenceFace(image_path):
    global reference_image
    reference_image = face_recognition.load_image_file(image_path)
    global reference_encoding
    reference_encoding = face_recognition.face_encodings(reference_image)[0]

def compare_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    isSame = False

    for face_encoding in face_encodings:
        # Compare faces
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            return True, True
        else:
            return True, False

    return False, False 

# Encode the reference face image
encodeReferenceFace(r"C:\Users\jarra\OneDrive\Pictures\Camera Roll 1\WIN_20250123_20_43_40_Pro.jpg")
# Start tracking
startTracking(webcam)