import cv2
from gaze_tracking import GazeTracking
import face_recognition
import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

def gaze_tracking(frame, gaze):
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    return frame

def startTracking(stream):
    
    global frame

    lastTime = datetime.datetime.now()
    match_text = ""  # Variable to hold the text for face comparison result
    faces_count_text = ""  # Variable to hold the number of faces detected

    while True:
        # We get a new frame from the webcam
        _, frame = stream.read()

        # Convert the frame to grayscale for Haar Cascade face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the number of faces detected
        faces_count = len(faces)
        faces_count_text = f"Faces detected: {faces_count}"

        # We send this frame to GazeTracking to analyze it
        annotatedFrame = gaze_tracking(frame, gaze)

        timeDiff = datetime.datetime.now() - lastTime
        timeDiffSeconds = timeDiff.total_seconds()

        if timeDiffSeconds >= 5:
            match_text = compare_faces(frame)  # Get the result text for face comparison
            lastTime = datetime.datetime.now()

        # Display the match text and face count text on the frame
        cv2.putText(annotatedFrame, match_text, (90, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotatedFrame, faces_count_text, (90, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        # Display the frame with gaze tracking info, match result, and face count
        cv2.imshow("Demo", annotatedFrame)

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

    for face_encoding in face_encodings:
        # Compare faces
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            return "Same person detected!"
        else:
            return "Different person"

    return ""  # In case no face is detected

# Encode the reference face image
encodeReferenceFace(r"C:\Users\jarra\OneDrive\Pictures\Camera Roll 1\WIN_20250123_20_43_40_Pro.jpg")
# Start tracking
startTracking(webcam)
