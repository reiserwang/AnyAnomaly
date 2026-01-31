import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies BEFORE import
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['clip'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()

# Needed for imports to work
import torch
import numpy as np

# Add backend and backend/core to path to resolve imports
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend'))
core_path = os.path.join(backend_path, 'core')
sys.path.append(backend_path)
sys.path.append(core_path)

from core.functions import grid_func

class TestGridOptimization(unittest.TestCase):
    def test_patch_selection_batching(self):
        """
        Verify that patch_selection uses batching (torch.cat) and single encode_image call
        instead of iterating and encoding individually.
        """
        # Setup
        gpatches = [MagicMock(), MagicMock(), MagicMock()] # 3 patches
        text = "test"
        model = MagicMock()
        device = "cpu"
        text_features = MagicMock()

        # Mock torch.cat
        batched = MagicMock()
        batched.to.return_value = batched
        mock_torch.cat.return_value = batched

        # Mock model output
        image_features = MagicMock()
        # Mock chained calls: .float() -> .norm() -> div -> T
        image_features.float.return_value = image_features
        # norm() returns a tensor
        norm_tensor = MagicMock()
        image_features.norm.return_value = norm_tensor
        # div (image_features /= ...) returns tensor
        image_features.__truediv__.return_value = image_features
        image_features.__itruediv__.return_value = image_features

        model.encode_image.return_value = image_features

        # Mock text features ops
        text_features.float.return_value = text_features
        text_features.norm.return_value = MagicMock()
        text_features.__truediv__.return_value = text_features
        text_features.__itruediv__.return_value = text_features

        # Mock similarity calculation
        # similarity = text_features @ image_features.T
        similarity_tensor = MagicMock()
        text_features.__matmul__.return_value = similarity_tensor

        # Mock view
        similarity_tensor.view.return_value = similarity_tensor

        # Mock max (returns values, indices)
        max_vals = MagicMock()
        max_vals.item.return_value = 0 # for argmax call if we used max_vals, but we use argmax on it?
        # Actually code uses torch.argmax(max_vals)
        # But wait: max_vals, _ = similarity.max(dim=1)
        # max_idx = torch.argmax(max_vals).item()
        similarity_tensor.max.side_effect = lambda dim: (max_vals, MagicMock())

        # .cpu().numpy()
        similarity_numpy = MagicMock()
        similarity_tensor.cpu.return_value.numpy.return_value = similarity_numpy

        # Mock numpy operations
        # The code does: max_arr.append(np.max(similarity)) (Old)
        # New code: np.max(similarity, axis=1) -> max_arr
        # We need np.max to work for both logic if possible, or just the new one.
        # Let's mock np.max to return a dummy array or scalar
        np.max.return_value = 0.5
        np.argmax.return_value = 0

        # Mock torch.argmax
        mock_torch.argmax.return_value.item.return_value = 0

        # Execute
        # This will fail with TypeError on the current codebase because text_features arg is missing
        try:
            grid_func.patch_selection(gpatches, text, model, device, text_features=text_features)
        except TypeError:
            self.fail("patch_selection does not accept text_features argument yet.")
        except Exception as e:
            # If it fails for other reasons (e.g. logic error), re-raise
            raise e

        # Verification

        # 1. Verify torch.cat was called to batch the inputs
        mock_torch.cat.assert_called_once()
        # args[0] should be gpatches
        self.assertEqual(mock_torch.cat.call_args[0][0], gpatches)

        # 2. Verify encode_image was called exactly ONCE
        model.encode_image.assert_called_once_with(batched)

        # 3. Verify we didn't use the loop-based encoding
        # (Implicitly verified by call_count == 1, if gpatches has 3 elements)

if __name__ == '__main__':
    unittest.main()
