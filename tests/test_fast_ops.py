import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Mock modules BEFORE importing fast_ops
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

mock_cv2 = MagicMock()
# Mock cv2 constants
mock_cv2.INTER_LINEAR = 1
sys.modules['cv2'] = mock_cv2

mock_numpy = MagicMock()
class MockNDArray: pass
mock_numpy.ndarray = MockNDArray
sys.modules['numpy'] = mock_numpy

mock_pil = MagicMock()
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil.Image

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

# Import the module under test
from core.functions.fast_ops import FastCLIPPreprocess

class TestFastCLIPPreprocess(unittest.TestCase):
    def setUp(self):
        self.fast_pp = FastCLIPPreprocess(size=224)
        mock_cv2.reset_mock()
        mock_torch.reset_mock()
        mock_numpy.reset_mock()
        mock_pil.reset_mock()

    def test_resize_logic(self):
        # Create a mock image (numpy array)
        # 448x800 image (H=448, W=800)
        img_mock = MagicMock()
        img_mock.shape = (448, 800, 3)
        # Mock numpy.array call
        mock_numpy.array.return_value = img_mock
        # Mock isinstance(img, np.ndarray) to return True if it's our mock
        # But we can't easily mock isinstance globally.
        # In fast_ops: if not isinstance(img, np.ndarray): img = np.array(img)
        # If I pass the mock, isinstance(mock, MagicMock) is True.
        # But isinstance(mock, np.ndarray) is False unless I subclass.
        # So it will hit `np.array(img)`.
        # I'll let `np.array` return the mock itself or a new mock.

        # Let's assume input IS a numpy array mock, but isinstance fails.
        # So np.array(img) is called.

        # To avoid infinite recursion or weirdness, let's make np.array return a NEW mock with same shape
        resized_mock = MagicMock()
        resized_mock.shape = (224, 400, 3) # After resize

        # We need to handle the flow:
        # 1. input mock (448, 800)
        # 2. np.array(input) -> returns input (or copy)
        mock_numpy.array.return_value = img_mock

        # 3. cv2.resize called
        mock_cv2.resize.return_value = resized_mock

        # Call preprocess
        self.fast_pp(img_mock)

        # Verify resize was called
        # Target: short side 224. H=448 -> 224 (0.5x). W=800 -> 400.
        # cv2.resize expects (width, height)
        mock_cv2.resize.assert_called_with(img_mock, (400, 224), interpolation=mock_cv2.INTER_LINEAR)

    def test_resize_logic_vertical(self):
        # 800x448 image
        img_mock = MagicMock()
        img_mock.shape = (800, 448, 3)
        mock_numpy.array.return_value = img_mock

        self.fast_pp(img_mock)

        # Target: short side 224. W=448 -> 224. H=800 -> 400.
        mock_cv2.resize.assert_called_with(img_mock, (224, 400), interpolation=mock_cv2.INTER_LINEAR)

    def test_no_resize_needed(self):
        # 224x300 image
        img_mock = MagicMock()
        img_mock.shape = (224, 300, 3)
        mock_numpy.array.return_value = img_mock

        self.fast_pp(img_mock)

        # Should NOT call resize because min(224, 300) = 224. scale = 1.0.
        mock_cv2.resize.assert_not_called()

    def test_output_structure(self):
        # Test that it calls torch conversion
        img_mock = MagicMock()
        img_mock.shape = (224, 224, 3)
        mock_numpy.array.return_value = img_mock

        # Mock slicing return
        # img[y:..., x:...]
        slice_mock = MagicMock()
        img_mock.__getitem__.return_value = slice_mock

        res = self.fast_pp(img_mock)

        # Verify torch.from_numpy was called with the cropped/sliced image
        mock_torch.from_numpy.assert_called()

    def test_string_input(self):
        # Test string path handling
        path = "test.jpg"
        mock_pil.Image.open.return_value = MagicMock()
        mock_numpy.array.return_value = MagicMock(shape=(224,224,3))

        self.fast_pp(path)

        mock_pil.Image.open.assert_called_with(path)

if __name__ == '__main__':
    unittest.main()
