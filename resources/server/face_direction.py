import cv2
import dlib
import numpy as np
from math import atan2, degrees

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat')

# 3D model points for pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

def get_face_angles(shape, size):
    # Get image size
    height, width = size

    # Camera internals
    focal_length = width
    center = (width/2, height/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.zeros((4,1))

    # Get specific facial landmarks
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype=np.float32)

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get Euler angles
    pitch = degrees(atan2(rotation_matrix[2][1], rotation_matrix[2][2]))
    yaw = degrees(atan2(-rotation_matrix[2][0], 
                       np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)))
    
    return yaw, pitch

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            
            # Calculate face angles
            yaw, pitch = get_face_angles(shape, frame.shape[:2])
            
            # Draw face rectangle
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # Display angles
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Face Pose Estimation', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()