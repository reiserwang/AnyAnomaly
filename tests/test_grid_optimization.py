import sys
import os
from unittest.mock import MagicMock
import unittest

# Mock modules BEFORE importing the module under test
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

mock_clip = MagicMock()
sys.modules['clip'] = mock_clip

mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2

mock_numpy = MagicMock()
sys.modules['numpy'] = mock_numpy
mock_numpy.zeros.return_value = MagicMock() # For grid_image_generation
mock_numpy.argmax.return_value = 0
mock_numpy.max.return_value = 0.5

# Mock backend modules
mock_utils = MagicMock()
sys.modules['utils'] = mock_utils

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/core')))

# Import the module under test
from functions import grid_func

class TestGridOptimization(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_clip.reset_mock()
        mock_torch.reset_mock()

        self.device = MagicMock()
        self.model = MagicMock()

        # Setup model encode returns
        self.feature_mock = MagicMock()
        self.feature_mock.norm.return_value = MagicMock()
        self.feature_mock.float.return_value = self.feature_mock
        self.feature_mock.__truediv__.return_value = self.feature_mock

        self.similarity_mock = MagicMock()
        self.similarity_mock.cpu.return_value.numpy.return_value = [[0.1, 0.2]]
        self.feature_mock.__matmul__.return_value = self.similarity_mock

        self.model.encode_image.return_value = self.feature_mock
        self.model.encode_text.return_value = self.feature_mock

        # Mocking split_images_with_unfold return
        # It returns a list of tensors (grouped patches)
        # patch_selection expects gpatches to be a list of tensors
        self.gpatch_mock = MagicMock()
        self.gpatch_mock.shape = [1, 3, 224, 224] # Batch size 1
        self.gpatches = [self.gpatch_mock]

    def test_patch_selection_optimization(self):
        text = "test"
        text_features = self.feature_mock

        # Test WITH text_features
        mock_clip.tokenize.reset_mock()

        grid_func.patch_selection(self.gpatches, text, self.model, self.device, text_features=text_features)

        print(f"DEBUG: call count for patch_selection = {mock_clip.tokenize.call_count}")
        self.assertEqual(mock_clip.tokenize.call_count, 0, "Tokenize called even when text_features provided!")

        # Test WITHOUT text_features
        mock_clip.tokenize.reset_mock()
        grid_func.patch_selection(self.gpatches, text, self.model, self.device, text_features=None)
        self.assertEqual(mock_clip.tokenize.call_count, 1, "Tokenize should be called once when text_features is None!")

    def test_grid_generation_optimization(self):
        cfg = MagicMock()
        cfg.sml_scale = False
        cfg.mid_scale = False
        cfg.lge_scale = False

        image_inputs = [MagicMock()]
        keyword = "test"
        text_features = self.feature_mock

        # Mock split_images_with_unfold since grid_generation calls it
        # We need to ensure we can split images.
        cfg.sml_scale = True
        cfg.sml_size = (80, 80)
        cfg.sml_size_stride = (80, 80)

        # Mock split_images_with_unfold in grid_func
        original_split = grid_func.split_images_with_unfold
        grid_func.split_images_with_unfold = MagicMock(return_value=[self.gpatch_mock])

        # Mock patch_selection to check arguments
        original_patch_selection = grid_func.patch_selection
        grid_func.patch_selection = MagicMock(return_value=0)

        # Mock grid_image_generation
        original_grid_image_generation = grid_func.grid_image_generation
        grid_func.grid_image_generation = MagicMock(return_value=MagicMock()) # Returns numpy array

        mock_transform2pil = mock_utils.transform2pil

        grid_func.grid_generation(cfg, image_inputs, keyword, self.model, self.device, text_features=text_features)

        # Check if patch_selection was called with text_features
        grid_func.patch_selection.assert_called_with(
            [self.gpatch_mock], keyword, self.model, self.device, text_features=text_features
        )

        # Restore mocks
        grid_func.split_images_with_unfold = original_split
        grid_func.patch_selection = original_patch_selection
        grid_func.grid_image_generation = original_grid_image_generation

if __name__ == '__main__':
    unittest.main()
