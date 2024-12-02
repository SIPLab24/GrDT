import cv2
import dlib

def detect_keypoints(image_path):
    """
    Detect facial landmarks (keypoints) using Dlib's landmark predictor.
    Args:
        image_path (str): Path to the input image.
    Returns:
        keypoints (list): List of (x, y) coordinates for detected keypoints.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    keypoints = []
    for face in faces:
        shape = predictor(gray, face)
        keypoints = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    
    return keypoints
