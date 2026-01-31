
import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Create a dummy class for Image.Image to satisfy isinstance
class MockImageClass:
    pass

class MockArray:
    def __init__(self, shape):
        self.shape = shape
    def astype(self, t): return self
    def transpose(self, *args): return self
    def __truediv__(self, other): return self
    def __sub__(self, other): return self
    def __getitem__(self, item): return self

# Mock dependencies BEFORE importing the module under test
mock_torch = MagicMock()
mock_numpy = MagicMock()
mock_cv2 = MagicMock()
mock_pil = MagicMock()

# Configure specific behaviors
mock_cv2.INTER_LINEAR = 1
mock_cv2.resize.return_value = MockArray((224, 336, 3)) # Dummy return

# Setup numpy array behavior
mock_array_instance = MockArray((400, 600, 3))

mock_numpy.array.return_value = mock_array_instance
mock_numpy.float32 = 'float32'
mock_numpy.from_numpy = MagicMock()
mock_numpy.from_numpy.return_value = MagicMock() # The tensor

# Configure PIL.Image module
mock_pil_image_module = MagicMock()
mock_pil_image_module.Image = MockImageClass

# LINK PIL.Image to PIL.Image module mock
mock_pil.Image = mock_pil_image_module

# Apply mocks to sys.modules
sys.modules["torch"] = mock_torch
sys.modules["numpy"] = mock_numpy
sys.modules["cv2"] = mock_cv2
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil_image_module

# Add path to backend/core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/core')))

# Now import the class
from functions.fast_ops import FastCLIPPreprocess

class TestFastCLIPPreprocess(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.preprocessor = FastCLIPPreprocess(size=224, device=self.device)

    def test_initialization(self):
        self.assertEqual(self.preprocessor.size, 224)
        self.assertEqual(self.preprocessor.device, "cpu")
        # Check if mean/std are created
        self.assertTrue(hasattr(self.preprocessor, 'mean'))
        self.assertTrue(hasattr(self.preprocessor, 'std'))

    def test_call_flow(self):
        # Create a mock image that satisfies isinstance(x, MockImageClass)
        mock_image = MockImageClass()

        # Call the preprocessor
        result = self.preprocessor(mock_image)

        # Verify steps

        # 1. np.array(image) called
        mock_numpy.array.assert_called_with(mock_image)

        # 2. cv2.resize called
        mock_cv2.resize.assert_called()
        args, kwargs = mock_cv2.resize.call_args
        # args[1] should be (new_w, new_h)
        # original (400, 600) -> 400 is smaller.
        # new_h = 224. new_w = int(600 * 224 / 400) = 336.
        # Expected size: (336, 224)
        self.assertEqual(args[1], (336, 224))
        self.assertEqual(kwargs.get('interpolation'), mock_cv2.INTER_LINEAR)

        # 3. torch.from_numpy called
        mock_torch.from_numpy.assert_called()

        # 4. to(device) called
        mock_torch.from_numpy.return_value.to.assert_called_with(self.device)

if __name__ == '__main__':
    unittest.main()
