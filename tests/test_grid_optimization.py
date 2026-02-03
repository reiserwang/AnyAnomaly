import sys
import os
from unittest.mock import MagicMock
import unittest

# Mock modules BEFORE importing the module under test
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn.functional'] = MagicMock()

mock_clip = MagicMock()
sys.modules['clip'] = mock_clip

mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2

mock_numpy = MagicMock()
sys.modules['numpy'] = mock_numpy
mock_numpy.zeros.return_value = MagicMock()

mock_utils = MagicMock()
sys.modules['utils'] = mock_utils

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

# Import the modules under test
from core.functions import grid_func
from core.functions import attn_func

class TestGridOptimization(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_clip.reset_mock()
        mock_torch.reset_mock()

        self.device = MagicMock()
        self.model = MagicMock()

        # Setup feature mock
        self.feature_mock = MagicMock()
        self.feature_mock.float.return_value = self.feature_mock
        self.feature_mock.norm.return_value = MagicMock()

        # Setup model behavior
        self.model.encode_text.return_value = self.feature_mock
        self.model.encode_image.return_value = self.feature_mock

        # Mock similarity calculation
        self.similarity_mock = MagicMock()
        self.similarity_mock.cpu.return_value.numpy.return_value = [[0.1, 0.2]] # Dummy similarity
        self.feature_mock.__matmul__.return_value = self.similarity_mock

    def test_patch_selection_optimization(self):
        """
        Test that patch_selection skips tokenization when text_features are provided.
        """
        gpatches = [MagicMock()] # Mock patch list
        # Mock shape[0] for batch size logic
        gpatches[0].shape = [1]

        text = "test_keyword"
        text_features = self.feature_mock

        # CASE 1: With text_features (Optimization enabled)
        # Note: This will fail until we implement the change
        try:
            grid_func.patch_selection(gpatches, text, self.model, self.device, text_features=text_features)

            # If function accepts text_features, check if tokenization was skipped
            print(f"DEBUG: call count for tokenize (optimized) = {mock_clip.tokenize.call_count}")
            if mock_clip.tokenize.call_count > 0:
                print("FAIL: Tokenize called despite text_features being provided")
            else:
                print("PASS: Tokenize skipped")

        except TypeError:
             print("FAIL: patch_selection does not accept text_features argument yet")

        # CASE 2: Without text_features (Standard behavior)
        mock_clip.tokenize.reset_mock()
        grid_func.patch_selection(gpatches, text, self.model, self.device)
        print(f"DEBUG: call count for tokenize (standard) = {mock_clip.tokenize.call_count}")
        if mock_clip.tokenize.call_count == 0:
             print("FAIL: Tokenize NOT called when text_features is missing")


    def test_grid_generation_optimization(self):
        """
        Test that grid_generation accepts text_features and passes them to patch_selection.
        """
        cfg = MagicMock()
        # Enable one scale to trigger patch generation
        cfg.sml_scale = True
        cfg.mid_scale = False
        cfg.lge_scale = False
        cfg.sml_size = (48, 48)
        cfg.sml_size_stride = (48, 48)

        image_inputs = [MagicMock()]
        keyword = "test"
        text_features = self.feature_mock

        # Mock split_images_with_unfold to return dummy patches
        grid_func.split_images_with_unfold = MagicMock(return_value=[MagicMock()])
        # Ensure the mock patch has shape for patch_selection
        grid_func.split_images_with_unfold.return_value[0].shape = [1]

        # Mock grid_image_generation to avoid complexity there
        grid_func.grid_image_generation = MagicMock()
        grid_func.transform2pil = MagicMock()

        # We also need to mock patch_selection to verify it receives text_features
        # But we can't easily mock it if we want to test the integration.
        # Instead, we rely on the fact that if grid_generation passes text_features,
        # and patch_selection uses them, mock_clip.tokenize won't be called.

        mock_clip.tokenize.reset_mock()

        try:
            grid_func.grid_generation(cfg, image_inputs, keyword, self.model, self.device, text_features=text_features)
             # If function accepts text_features, check if tokenization was skipped inside patch_selection
            if mock_clip.tokenize.call_count > 0:
                print("FAIL: grid_generation -> patch_selection did not skip tokenization")
            else:
                print("PASS: grid_generation passed text_features correctly")
        except TypeError:
            print("FAIL: grid_generation does not accept text_features argument yet")


    def test_patch_similarity_inplace_modification(self):
        """
        Test that patch_similarity does not modify text_embedding in-place.
        """
        patches = MagicMock()
        # Create a specific mock for text_embedding to track operations
        text_embedding = MagicMock()
        text_embedding.norm.return_value = MagicMock()

        # We need to simulate the tensor behavior.
        # In-place division calls __itruediv__.
        # Normal division calls __truediv__.

        attn_func.patch_similarity(patches, text_embedding, self.model, self.device)

        # Check calls
        if len(text_embedding.__itruediv__.mock_calls) > 0:
            print("FAIL: text_embedding modified in-place (used /=)")
        else:
             print("PASS: text_embedding NOT modified in-place")

        if len(text_embedding.__truediv__.mock_calls) == 0:
             # Wait, if we fix it, it SHOULD use __truediv__
             # If it uses /=, it uses __itruediv__
             pass

if __name__ == '__main__':
    unittest.main()
