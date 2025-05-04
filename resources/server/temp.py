import datetime
import os
import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self):
        # Initialize face detectors
        # Regular face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Profile face detector
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        # Dlib's face detector (more robust for different angles)
        self.dlib_detector = dlib.get_frontal_face_detector()

    def draw_rectangle(self, img, x, y, w, h, color=(0, 255, 0), thickness=2):
        """Draw a rectangle around the face"""
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    def detect_faces(self, image_path, output_path=None):
        """
        Detect faces in the image and draw rectangles around them
        Returns the image with rectangles drawn and the number of faces detected
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store all detected faces
        faces = []

        # 1. Detect using OpenCV's frontal face detector
        frontal_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        faces.extend(frontal_faces)

        # 2. Detect using OpenCV's profile face detector
        # Try both original and flipped image for left/right facing profiles
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        faces.extend(profile_faces)

        # Flip image and try again for opposite profile
        flipped = cv2.flip(gray, 1)
        profile_faces_flipped = self.profile_cascade.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        # Adjust coordinates for flipped faces
        for (x, y, w, h) in profile_faces_flipped:
            faces.append((img.shape[1] - x - w, y, w, h))

        # 3. Detect using dlib (more robust for different angles)
        dlib_faces = self.dlib_detector(gray)
        for face in dlib_faces:
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            faces.append((x, y, w, h))

        # Remove overlapping detections
        faces = self.remove_overlapping_boxes(faces)

        # Draw rectangles around all detected faces
        for (x, y, w, h) in faces:
            self.draw_rectangle(img, x, y, w, h)

        # Save the output image if path is provided
        if output_path:
            cv2.imwrite(output_path, img)

        return img, len(faces)

    def remove_overlapping_boxes(self, boxes, overlap_thresh=0.5):
        """Remove overlapping bounding boxes"""
        if not boxes:
            return []

        # Convert to numpy array if not already
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)

        # Get coordinates for comparison
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # Calculate area of each box
        areas = (x2 - x1) * (y2 - y1)

        # Sort by bottom-right y-coordinate
        indices = np.argsort(y2)
        
        keep = []
        
        while len(indices) > 0:
            # Keep the largest box
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)

            # Find overlap with other boxes
            xx1 = np.maximum(x1[i], x1[indices[:last]])
            yy1 = np.maximum(y1[i], y1[indices[:last]])
            xx2 = np.minimum(x2[i], x2[indices[:last]])
            yy2 = np.minimum(y2[i], y2[indices[:last]])

            # Calculate overlap area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[indices[:last]]

            # Remove overlapping boxes
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        return boxes[keep].tolist()

def main():
    # Create detector instance
    detector = FaceDetector()

    try:
        # Define input and output paths
        input_path = r"C:\Users\Public\Documents\AI_Exam\AI-Proctored-Exam-Desktop\resources\server\annotated_images\annotated_WIN_20250224_13_11_08_Pro_20250224_131214.jpg"
        output_folder = r"C:\Users\Public\Documents\AI_Exam\AI-Proctored-Exam-Desktop\resources\server\face_detected_images"
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"face_detected_{os.path.basename(input_path).split('.')[0]}_{timestamp}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        
        # Detect faces and draw rectangles
        result_image, face_count = detector.detect_faces(input_path, output_path)
        
        print(f"Detected {face_count} faces")
        print(f"Processed image saved to: {output_path}")
        
        # Display the image
        cv2.imshow("Detected Faces", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()