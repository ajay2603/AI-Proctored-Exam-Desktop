import os
import cv2
import numpy as np
import dlib
from math import atan2, degrees
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from gaze_tracking import GazeTracking
import face_recognition

# Initialize the necessary components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat')
gaze = GazeTracking()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3D model points for pose estimation (same as in your original code)
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

def analyze_image(image_path, reference_encoding=None):
    """Analyze a single image and determine if cheating is detected"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    result = {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection with opencv for counting faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_count = len(faces)
    result['face_count'] = faces_count
    
    # Gaze tracking
    gaze.refresh(frame)
    is_gaze_tracked = not gaze.is_blinking()
    result['isGazeTracked'] = is_gaze_tracked
    
    # Check gaze direction
    is_cheating = False
    if gaze.is_right() or gaze.is_left():
        result['gazeCheating'] = True
        is_cheating = True
    else:
        result['gazeCheating'] = False
    
    # Check face direction using dlib
    dlib_faces = detector(gray)
    if dlib_faces:
        shape = predictor(gray, dlib_faces[0])
        # Calculate face angles
        yaw, pitch = get_face_angles(shape, frame.shape[:2])
        
        if yaw < -35:
            result['face_direction'] = "Face Left"
            is_cheating = True
        elif yaw > 35:
            result['face_direction'] = "Face Right"
            is_cheating = True
        else:
            result['face_direction'] = "Face Center"
    
    # Compare faces if reference encoding is provided
    if reference_encoding is not None:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        if face_encodings:
            matches = face_recognition.compare_faces([reference_encoding], face_encodings[0])
            if not matches[0]:
                is_cheating = True
    
    # Multiple faces is considered cheating
    is_cheating = is_cheating 
    result['isCheating'] = is_cheating
    
    return result

def evaluate_system(genuine_folder, cheating_folder, reference_image=None):
    """Evaluate system performance on genuine and cheating images"""
    # Load reference face encoding if provided
    reference_encoding = None
    if reference_image:
        ref_img = cv2.imread(reference_image)
        if ref_img is not None:
            encodings = face_recognition.face_encodings(ref_img)
            if encodings:
                reference_encoding = encodings[0]
                print("Reference face encoded successfully")
            else:
                print("WARNING: No face found in reference image")
        else:
            print(f"WARNING: Could not load reference image {reference_image}")
    
    # Process all images
    true_labels = []
    predicted_labels = []
    failed_images = []
    
    # Process genuine images (expected: not cheating)
    for img_name in os.listdir(genuine_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(genuine_folder, img_name)
            result = analyze_image(img_path, reference_encoding)
            
            if result is None:
                failed_images.append(img_path)
                continue
                
            true_labels.append(0)  # 0 = not cheating
            predicted_labels.append(1 if result['isCheating'] else 0)
    
    # Process cheating images (expected: cheating)
    for img_name in os.listdir(cheating_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(cheating_folder, img_name)
            result = analyze_image(img_path, reference_encoding)
            
            if result is None:
                failed_images.append(img_path)
                continue
                
            true_labels.append(1)  # 1 = cheating
            predicted_labels.append(1 if result['isCheating'] else 0)
    
    # Calculate metrics
    if not true_labels or not predicted_labels:
        print("No valid images were processed!")
        return
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=['Genuine', 'Cheating'])
    
    # Print results
    print(f"Evaluation Results:")
    print(f"Total images processed: {len(true_labels)}")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images")
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Cheating'], 
                yticklabels=['Genuine', 'Cheating'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Return metrics for further analysis if needed
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }

if __name__ == "__main__":
    # Update these paths with your actual folder paths
    GENUINE_FOLDER = r"C:\Users\jarra\OneDrive\Pictures\Camera Roll 1\genuine"
    CHEATING_FOLDER = r"C:\Users\jarra\OneDrive\Pictures\Camera Roll 1\cheating"
    REFERENCE_IMAGE = r"C:\Users\jarra\OneDrive\Pictures\Camera Roll 1\WIN_20250224_12_48_17_Pro.jpg" # Optional, set to None if not using face verification
    
    # Run evaluation
    results = evaluate_system(GENUINE_FOLDER, CHEATING_FOLDER, REFERENCE_IMAGE)