import cv2
import os
from keypoint_detection import detect_keypoints
from mask_generation import generate_mask, smooth_mask
from texture_extraction import extract_texture_features
from feature_classification import TextureClassifier
from utils.visualization import visualize_keypoints, visualize_mask, plot_texture_features

def main(image_path, classifier_model_path=None):
    """
    Main function for the FRT pipeline: keypoint detection, mask generation, texture feature extraction, and classification.
    Args:
        image_path (str): Path to the input image.
        classifier_model_path (str): Path to a pre-trained classifier model (optional).
    """
    # Step 1: Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 2: Detect keypoints
    print("[INFO] Detecting facial keypoints...")
    keypoints = detect_keypoints(image_path)
    visualize_keypoints(image, keypoints)
    
    # Step 3: Generate and smooth mask
    print("[INFO] Generating mask...")
    mask = generate_mask(image.shape, keypoints)
    smoothed_mask = smooth_mask(mask, sigma=2)
    visualize_mask(image, smoothed_mask)
    
    # Step 4: Extract texture features
    print("[INFO] Extracting texture features...")
    masked_image = cv2.bitwise_and(image, image, mask=(smoothed_mask > 0.5).astype("uint8"))
    features = extract_texture_features(masked_image)
    feature_names = ["Contrast", "Entropy", "Homogeneity", "Energy"] * 4
    plot_texture_features(features, feature_names)
    
    # Step 5: Classify texture features
    print("[INFO] Classifying features...")
    classifier = TextureClassifier()
    if classifier_model_path and os.path.exists(classifier_model_path):
        classifier.load_model(classifier_model_path)  # Assuming a method to load a pre-trained model
    else:
        print("[WARN] No pre-trained model provided. Training a dummy model...")
        # Dummy training data
        X_train = [features]  # Replace with real training data
        y_train = [0]         # Replace with real labels
        classifier.train(X_train, y_train)

    # Prediction
    prediction = classifier.predict([features])
    print(f"[RESULT] Predicted label: {prediction[0]}")

if __name__ == "__main__":
    # Example usage
    image_path = "data/sample_images/sample.jpg"
    classifier_model_path = "data/models/texture_classifier.pkl"
    main(image_path, classifier_model_path)

