import sys
import os
from unittest.mock import MagicMock
import unittest

# Mock modules BEFORE importing the module under test
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

mock_clip = MagicMock()
sys.modules['clip'] = mock_clip

mock_pil = MagicMock()
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil.Image

mock_numpy = MagicMock()
sys.modules['numpy'] = mock_numpy
mock_numpy.argmax.return_value = 0 # Default return for argmax
mock_numpy.argsort.return_value = [0, 1, 2, 3] # Default for argsort

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

# Import the module under test
from core.functions.key_func import KFS, key_frame_selection, key_frame_selection_four_idx

class TestKFSOptimization(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_clip.reset_mock()
        mock_torch.reset_mock()

        # Setup common mocks
        self.device = MagicMock()
        self.model = MagicMock()
        self.preprocess = MagicMock()

        # Setup model encode returns to avoid errors during dot product or norm
        # We need to return a mock that supports arithmetic
        self.feature_mock = MagicMock()
        self.feature_mock.norm.return_value = MagicMock()
        self.feature_mock.float.return_value = self.feature_mock
        # When divided, return self
        self.feature_mock.__truediv__.return_value = self.feature_mock
        # When matmul (@), return a mock that can be converted to cpu().numpy()
        self.similarity_mock = MagicMock()
        self.similarity_mock.cpu.return_value.numpy.return_value = [[0.1, 0.2, 0.3, 0.4]] # Dummy similarity
        self.feature_mock.__matmul__.return_value = self.similarity_mock

        self.model.encode_image.return_value = self.feature_mock
        self.model.encode_text.return_value = self.feature_mock

        # Setup KFS instance
        self.kfs = KFS(select_num=4, clip_length=4, model=self.model, preprocess=self.preprocess, device=self.device)

    def test_key_frame_selection_clip_grouping_optimization(self):
        clip_data = [MagicMock()] * 4
        anomaly_text = "test"
        text_features = self.feature_mock # Provided

        mock_clip.tokenize.reset_mock()
        self.kfs.key_frame_selection_clip_grouping(clip_data, anomaly_text, text_features=text_features)

        # Should be 0 calls if optimized, 1 call if unoptimized
        print(f"DEBUG: call count for clip_grouping = {mock_clip.tokenize.call_count}")
        self.assertEqual(mock_clip.tokenize.call_count, 0, "Tokenize called even when text_features provided!")

        # Test None case
        mock_clip.tokenize.reset_mock()
        self.kfs.key_frame_selection_clip_grouping(clip_data, anomaly_text, text_features=None)
        self.assertEqual(mock_clip.tokenize.call_count, 1, "Tokenize should be called once when text_features is None!")

    def test_key_frame_selection_clip_optimization(self):
        clip_data = [MagicMock()] * 4
        anomaly_text = "test"
        text_features = self.feature_mock

        mock_clip.tokenize.reset_mock()
        self.kfs.key_frame_selection_clip(clip_data, anomaly_text, text_features=text_features)

        print(f"DEBUG: call count for clip = {mock_clip.tokenize.call_count}")
        self.assertEqual(mock_clip.tokenize.call_count, 0, "Tokenize called even when text_features provided!")

        # Test None case
        mock_clip.tokenize.reset_mock()
        self.kfs.key_frame_selection_clip(clip_data, anomaly_text, text_features=None)
        self.assertEqual(mock_clip.tokenize.call_count, 1, "Tokenize should be called once when text_features is None!")

    def test_key_frame_selection_grouping_clip_optimization(self):
        clip_data = [MagicMock()] * 4
        anomaly_text = "test"
        text_features = self.feature_mock

        mock_clip.tokenize.reset_mock()
        self.kfs.key_frame_selection_grouping_clip(clip_data, anomaly_text, text_features=text_features)

        print(f"DEBUG: call count for grouping_clip = {mock_clip.tokenize.call_count}")
        self.assertEqual(mock_clip.tokenize.call_count, 0, "Tokenize called even when text_features provided!")

        # Test None case
        mock_clip.tokenize.reset_mock()
        self.kfs.key_frame_selection_grouping_clip(clip_data, anomaly_text, text_features=None)
        self.assertEqual(mock_clip.tokenize.call_count, 1, "Tokenize should be called once when text_features is None!")

    def test_standalone_key_frame_selection_optimization(self):
        clip_data = [MagicMock()] * 4
        anomaly_text = "test"
        text_features = self.feature_mock

        mock_clip.tokenize.reset_mock()
        key_frame_selection(clip_data, anomaly_text, self.model, self.preprocess, self.device, text_features=text_features)

        print(f"DEBUG: call count for standalone kfs = {mock_clip.tokenize.call_count}")
        self.assertEqual(mock_clip.tokenize.call_count, 0, "Tokenize called even when text_features provided!")

        # Test None case
        mock_clip.tokenize.reset_mock()
        key_frame_selection(clip_data, anomaly_text, self.model, self.preprocess, self.device, text_features=None)
        self.assertEqual(mock_clip.tokenize.call_count, 1, "Tokenize should be called once when text_features is None!")

    def test_standalone_key_frame_selection_four_idx_optimization(self):
        clip_data = [MagicMock()] * 4
        anomaly_text = "test"
        text_features = self.feature_mock

        mock_clip.tokenize.reset_mock()
        key_frame_selection_four_idx(4, clip_data, anomaly_text, self.model, self.preprocess, self.device, text_features=text_features)

        print(f"DEBUG: call count for standalone kfs_four = {mock_clip.tokenize.call_count}")
        self.assertEqual(mock_clip.tokenize.call_count, 0, "Tokenize called even when text_features provided!")

        # Test None case
        mock_clip.tokenize.reset_mock()
        key_frame_selection_four_idx(4, clip_data, anomaly_text, self.model, self.preprocess, self.device, text_features=None)
        self.assertEqual(mock_clip.tokenize.call_count, 1, "Tokenize should be called once when text_features is None!")

if __name__ == '__main__':
    unittest.main()
