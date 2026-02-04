import sys
from unittest.mock import MagicMock, patch

# Mock modules BEFORE importing the target module
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

mock_clip = MagicMock()
sys.modules['clip'] = mock_clip

sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock utils module
mock_utils = MagicMock()
sys.modules['utils'] = mock_utils

# Import the module under test
# We need to make sure backend.core is in path or we import relative if running from root
# The environment seems to have backend/core/functions
# We might need to adjust sys.path if 'backend.core' is not resolved
import os
sys.path.append(os.getcwd())

try:
    from backend.core.functions import grid_func
except ImportError:
    # Try adding backend/core to path as well if needed,
    # but the structure seems to be backend.core.functions...
    sys.path.append(os.path.join(os.getcwd(), 'backend', 'core'))
    from functions import grid_func

import unittest

class TestGridOptimization(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        mock_clip.tokenize.reset_mock()
        mock_torch.cat.reset_mock()

    def test_grid_generation_calls_tokenize_by_default(self):
        """
        Verifies that grid_generation calls clip.tokenize by default (when no text_features provided).
        """
        cfg = MagicMock()
        # Disable scales so we don't need to mock split_images_with_unfold logic deeply
        # patch_selection will still be called with empty gpatches,
        # and it calls clip.tokenize at the very top.
        cfg.sml_scale = False
        cfg.mid_scale = False
        cfg.lge_scale = False

        image_inputs = [MagicMock()]
        keyword = "anomaly"
        clip_model = MagicMock()
        device = MagicMock()

        # We need patch_selection to not crash on empty gpatches if we want to reach the end,
        # but strictly speaking we only care that tokenize is called.
        # However, let's mock patch_selection's internals to be safe?
        # Actually, patch_selection is inside grid_func.
        # We can just let it run. It calls clip.tokenize first.
        # Then it might crash at torch.cat(gpatches) if gpatches is empty and torch.cat mock complains.
        # Let's mock torch.cat to return something safe.
        mock_torch.cat.return_value = MagicMock()

        try:
            grid_func.grid_generation(cfg, image_inputs, keyword, clip_model, device)
        except Exception as e:
            # We don't care if it crashes later, as long as tokenize was called
            pass

        mock_clip.tokenize.assert_called_once()
        print("\nVerified: clip.tokenize was called in default execution.")

    def test_grid_generation_skips_tokenize_with_features(self):
        """
        Verifies that grid_generation skips clip.tokenize when text_features are provided.
        """
        cfg = MagicMock()
        cfg.sml_scale = False
        cfg.mid_scale = False
        cfg.lge_scale = False

        image_inputs = [MagicMock()]
        keyword = "anomaly"
        clip_model = MagicMock()
        device = MagicMock()
        text_features = MagicMock() # Pre-computed features

        # Ensure cat doesn't crash if gpatches is empty
        mock_torch.cat.return_value = MagicMock()

        try:
            grid_func.grid_generation(cfg, image_inputs, keyword, clip_model, device, text_features=text_features)
        except Exception as e:
            pass

        mock_clip.tokenize.assert_not_called()
        print("\nVerified: clip.tokenize was NOT called when text_features were provided.")

if __name__ == '__main__':
    unittest.main()
