import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/core')))

# Mock modules
mock_torch = MagicMock()
mock_torch.backends.mps.is_available.return_value = True
mock_device = MagicMock()
mock_device.type = 'mps'
mock_torch.device.return_value = mock_device
mock_torch.float16 = 'float16'
mock_torch.float32 = 'float32'
sys.modules['torch'] = mock_torch

sys.modules['clip'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['cv2'] = MagicMock()
# sys.modules['concurrent.futures'] = MagicMock() # Don't mock standard lib if possible

# Mock functions package
sys.modules['functions'] = MagicMock()
sys.modules['functions.MiniCPM_func'] = MagicMock()
sys.modules['functions.attn_func'] = MagicMock()
sys.modules['functions.grid_func'] = MagicMock()
sys.modules['functions.text_func'] = MagicMock()
sys.modules['functions.key_func'] = MagicMock()
sys.modules['config'] = MagicMock()

# Now import
# We rely on the mocks being in sys.modules so imports in cvad_detector don't fail
try:
    from cvad_detector import CVADDetector
except ImportError:
    # Fallback if simple import fails, though sys.path should handle it
    pass

class TestMPSFallback(unittest.TestCase):

    @patch('cvad_detector.load_lvlm')
    @patch('cvad_detector.update_config')
    @patch('cvad_detector.KFS')
    @patch('cvad_detector.clip.load')
    def test_mps_int4_fallback(self, mock_clip_load, mock_kfs, mock_update_config, mock_load_lvlm):
        mock_clip_load.return_value = (MagicMock(), MagicMock())

        mock_cfg = MagicMock()
        # Initial model path with int4
        mock_cfg.model_path = 'MiniCPM-V-2_6-int4'
        mock_cfg.kfs_num = 4
        mock_cfg.clip_length = 24

        # update_config returns this mock_cfg
        mock_update_config.return_value = mock_cfg

        # side_effect for load_lvlm
        def side_effect(model_path, device):
            if 'int4' in model_path:
                raise Exception("AutoGPTQ failure on MPS")
            return (MagicMock(), MagicMock())

        mock_load_lvlm.side_effect = side_effect

        # Instantiate
        # This should SUCCEED now with fallback
        detector = CVADDetector(model_path='MiniCPM-V-2_6', quantize=True)

        # Verify load_lvlm called twice
        self.assertEqual(mock_load_lvlm.call_count, 2)

        # Args of first call
        self.assertIn('int4', mock_load_lvlm.call_args_list[0][0][0])

        # Args of second call
        self.assertEqual(mock_load_lvlm.call_args_list[1][0][0], 'MiniCPM-V-2_6')

        # Verify that cfg.model_path was updated
        self.assertEqual(detector.cfg.model_path, 'MiniCPM-V-2_6')

if __name__ == '__main__':
    unittest.main()
