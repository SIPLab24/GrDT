import unittest
import torch
from FacialRepresentationTexture.utils.glcm import compute_glcm
from GNNDeepfakeClassification.lib.GRD import GRDNetwork
from GNNDeepfakeClassification.main import AdaptiveWeightFusion
from FacialRepresentationTexture.texture_extraction import extract_texture_features
from FacialRepresentationTexture.feature_classification import TextureClassifier
from GNNDeepfakeClassification.geometric_classifier import GeometricClassifier


class TestGrDTNetwork(unittest.TestCase):
    def setUp(self):
        """
        Set up dummy data for testing.
        """
        # Dummy image and mask for FRT path
        self.dummy_image = torch.randint(0, 255, (256, 256), dtype=torch.uint8)
        self.dummy_mask = torch.randint(0, 2, (256, 256), dtype=torch.uint8)

        # Dummy feature vectors for FRT and GRD classifiers
        self.dummy_texture_features = torch.rand(1, 256)
        self.dummy_geometric_features = torch.rand(1, 256)

        # Dummy model instances
        self.texture_classifier = TextureClassifier(input_dim=256, output_dim=2)
        self.geometric_classifier = GeometricClassifier(input_dim=256, output_dim=2)
        self.fusion = AdaptiveWeightFusion()

        # Dummy GRD network
        self.grd_network = GRDNetwork(input_dim=256, output_dim=2)

    def test_glcm_computation(self):
        """
        Test GLCM computation for texture analysis.
        """
        glcm = compute_glcm(self.dummy_image.numpy(), distance=1, angle=0)
        self.assertIsNotNone(glcm, "GLCM computation failed")
        self.assertEqual(glcm.shape, (2, 2), "Incorrect GLCM shape for binary image")

    def test_texture_feature_extraction(self):
        """
        Test texture feature extraction.
        """
        features = extract_texture_features(self.dummy_image.numpy())
        self.assertEqual(len(features), 16, "Incorrect number of texture features extracted")

    def test_texture_classifier(self):
        """
        Test the Texture Classifier with dummy features.
        """
        prediction = self.texture_classifier(self.dummy_texture_features)
        self.assertEqual(prediction.shape, (1, 2), "Incorrect prediction shape from Texture Classifier")

    def test_geometric_classifier(self):
        """
        Test the Geometric Classifier with dummy features.
        """
        prediction = self.geometric_classifier(self.dummy_geometric_features)
        self.assertEqual(prediction.shape, (1, 2), "Incorrect prediction shape from Geometric Classifier")

    def test_grd_network(self):
        """
        Test the GRD path network with dummy input.
        """
        output = self.grd_network(self.dummy_geometric_features)
        self.assertEqual(output.shape, (1, 2), "Incorrect output shape from GRD Network")

    def test_adaptive_weight_fusion(self):
        """
        Test the Adaptive Weight Fusion mechanism.
        """
        T_texture = self.texture_classifier(self.dummy_texture_features)
        T_geometric = self.geometric_classifier(self.dummy_geometric_features)
        T_total = self.fusion.forward(T_texture, T_geometric)
        self.assertEqual(T_total.shape, (1, 2), "Incorrect fused output shape")

    def test_pipeline_integration(self):
        """
        Test integration of FRT, GRD, and fusion for a complete pipeline.
        """
        # Texture path prediction
        T_texture = self.texture_classifier(self.dummy_texture_features)

        # Geometric path prediction
        T_geometric = self.geometric_classifier(self.dummy_geometric_features)

        # Fusion
        T_total = self.fusion.forward(T_texture, T_geometric)

        # Check final output
        self.assertEqual(T_total.shape, (1, 2), "Incorrect final output shape from pipeline")

if __name__ == "__main__":
    unittest.main()
