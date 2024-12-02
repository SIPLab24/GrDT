from src.texture_extraction import extract_texture_features
import numpy as np

def test_texture_extraction():
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    features = extract_texture_features(image)
    assert len(features) > 0, "Texture feature extraction failed!"
    print("Texture feature extraction test passed!")
