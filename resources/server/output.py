import cv2
import dlib
import numpy as np
from math import atan2, degrees
import os
from gaze_tracking import GazeTracking
import face_recognition
from datetime import datetime

class ImageAnnotator:
    def __init__(self, output_folder="annotated_images"):
        # Initialize multiple detectors for better side-face detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat')
        self.gaze = GazeTracking()
        
        # Load both frontal and profile face cascades
        self.face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Create output folder if it doesn't exist
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # 3D model points for pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def get_face_angles(self, shape, size):
        """
        Calculate face angles using facial landmarks and pose estimation
        """
        height, width = size
        focal_length = width
        center = (width/2, height/2)
        
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4,1))
        
        # Get facial landmarks for pose estimation
        try:
            image_points = np.array([
                (shape.part(30).x, shape.part(30).y),     # Nose tip
                (shape.part(8).x, shape.part(8).y),       # Chin
                (shape.part(36).x, shape.part(36).y),     # Left eye left corner
                (shape.part(45).x, shape.part(45).y),     # Right eye right corner
                (shape.part(48).x, shape.part(48).y),     # Left mouth corner
                (shape.part(54).x, shape.part(54).y)      # Right mouth corner
            ], dtype=np.float32)
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs
            )
            
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch = degrees(atan2(rotation_matrix[2][1], rotation_matrix[2][2]))
            yaw = degrees(atan2(-rotation_matrix[2][0], 
                               np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)))
            
            return yaw, pitch
        except Exception as e:
            print(f"Error calculating face angles: {e}")
            return 0, 0

    def detect_faces(self, gray):
        """
        Enhanced face detection using multiple detectors and configurations
        """
        faces = []
        
        # Detect using frontal cascade
        frontal_faces = self.face_cascade_frontal.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,  # Reduced for better detection
            minSize=(30, 30)
        )
        
        # Detect using profile cascade (original image)
        profile_faces_right = self.face_cascade_profile.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # More lenient for profile detection
            minSize=(30, 30)
        )
        
        # Detect using profile cascade (flipped image for left-facing profiles)
        flipped = cv2.flip(gray, 1)
        profile_faces_left = self.face_cascade_profile.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        # Add frontal faces
        for face in frontal_faces:
            faces.append(('frontal', face))
            
        # Add right-profile faces
        for face in profile_faces_right:
            faces.append(('right_profile', face))
            
        # Add left-profile faces (adjust coordinates due to flip)
        img_width = gray.shape[1]
        for face in profile_faces_left:
            x, y, w, h = face
            adjusted_face = (img_width - (x + w), y, w, h)
            faces.append(('left_profile', adjusted_face))
            
        return faces

    def process_image(self, image_path, reference_image_path=None):
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Create a copy for annotations
        annotated_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process gaze
        self.gaze.refresh(frame)
        annotated_frame = self.gaze.annotated_frame()
        
        # Add gaze information
        if self.gaze.is_blinking():
            text = "Blinking"
        elif self.gaze.is_right():
            text = "Looking right"
        elif self.gaze.is_left():
            text = "Looking left"
        elif self.gaze.is_center():
            text = "Looking center"
        else:
            text = "Gaze unknown"
            
        cv2.putText(annotated_frame, "Gaze: " + text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        
        # Add gaze ratios
        h_ratio = self.gaze.horizontal_ratio()
        v_ratio = self.gaze.vertical_ratio()
        if h_ratio is not None:
            cv2.putText(annotated_frame, f"Gaze Horizontal ratio: {h_ratio:.3f}", 
                       (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        if v_ratio is not None:
            cv2.putText(annotated_frame, f"Gaze Vertical ratio: {v_ratio:.3f}", 
                       (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        
        # Detect faces using enhanced detection
        faces = self.detect_faces(gray)
        cv2.putText(annotated_frame, f"Faces: {len(faces)}", 
                   (90, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Draw rectangles around faces with different colors based on detection type
        colors = {
            'frontal': (0, 255, 0),      # Green
            'right_profile': (255, 0, 0), # Blue
            'left_profile': (0, 0, 255)   # Red
        }
        
        for face_type, (x, y, w, h) in faces:
            color = colors[face_type]
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        # Get face angles using dlib for frontal faces
        dlib_faces = self.detector(gray)
        for face in dlib_faces:
            try:
                shape = self.predictor(gray, face)
                yaw, pitch = self.get_face_angles(shape, frame.shape[:2])
                
                # Add face direction with expanded range
                face_direction = "Face Center"
                if yaw < -25:  # More lenient threshold
                    face_direction = "Face Left"
                elif yaw > 25:  # More lenient threshold
                    face_direction = "Face Right"
                    
                cv2.putText(annotated_frame, f"Face direction: {face_direction}", 
                           (90, 280), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Yaw: {yaw:.1f}", 
                           (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Pitch: {pitch:.1f}", 
                           (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face angles: {e}")
        
        # Compare faces if reference image is provided
        if reference_image_path:
            ref_image = cv2.imread(reference_image_path)
            if ref_image is not None:
                ref_encoding = face_recognition.face_encodings(ref_image)
                if len(ref_encoding) > 0:
                    current_encoding = face_recognition.face_encodings(frame)
                    if len(current_encoding) > 0:
                        matches = face_recognition.compare_faces([ref_encoding[0]], current_encoding[0])
                        match_text = "Same person detected!" if matches[0] else "Different person"
                        cv2.putText(annotated_frame, match_text, 
                                  (90, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"annotated_{os.path.basename(image_path).split('.')[0]}_{timestamp}.jpg"
        output_path = os.path.join(self.output_folder, output_filename)
        
        # Save the annotated image
        cv2.imwrite(output_path, annotated_frame)
        return output_path

def main():
    # Example usage
    annotator = ImageAnnotator()
    
    # Process a single image
    try:
        output_path = annotator.process_image(
            r"C:\Users\jarra\OneDrive\Desktop\nar-1.jpg",
            reference_image_path=r"C:\Users\jarra\OneDrive\Pictures\Camera Roll 1\WIN_20250224_12_48_17_Pro.jpg"  # Optional
        )
        print(f"Annotated image saved to: {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()