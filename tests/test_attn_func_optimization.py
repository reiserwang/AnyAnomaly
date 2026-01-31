import sys
import os
import unittest
import torch
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Add backend paths
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend'))
core_path = os.path.join(backend_path, 'core')
sys.path.append(backend_path)
sys.path.append(core_path)

# Mock matplotlib before importing utils
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

from functions.attn_func import winclip_attention, split_one_image_with_unfold, preprocess_image_for_unfold

class TestAttnFuncOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.sml_scale = True
        self.cfg.mid_scale = True
        self.cfg.lge_scale = True
        self.cfg.sml_size = (48, 48)
        self.cfg.sml_size_stride = (24, 24)
        self.cfg.mid_size = (80, 80)
        self.cfg.mid_size_stride = (40, 40)
        self.cfg.lge_size = (120, 120)
        self.cfg.lge_size_stride = (60, 60)

        # Patch nums need to match logic or be ignored if we mock patch_similarity return shape
        self.cfg.sml_patch_num = (9, 9) # Approximate
        self.cfg.mid_patch_num = (5, 5)
        self.cfg.lge_patch_num = (3, 3)

        self.clip_model = MagicMock()
        self.clip_model.encode_image.return_value = torch.randn(1, 512)

        self.text_embedding = torch.randn(1, 512)

    def test_preprocess_image_for_unfold(self):
        # Test with numpy array
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess_image_for_unfold(img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 240, 240))

        # Test with string (mock cv2.imread)
        with patch('cv2.imread', return_value=img):
            tensor = preprocess_image_for_unfold("dummy.jpg")
            self.assertEqual(tensor.shape, (1, 3, 240, 240))

    def test_split_one_image_with_unfold_accepts_tensor(self):
        # Create a tensor
        tensor_input = torch.randn(1, 3, 240, 240)

        # Call split
        patches = split_one_image_with_unfold(tensor_input, kernel_size=(48, 48), stride_size=(24, 24))

        # Check output
        # Unfold output size depends on logic
        self.assertIsInstance(patches, torch.Tensor)
        # Should be interpolated to 224x224
        self.assertEqual(patches.shape[-2:], (224, 224))

    def test_split_one_image_with_unfold_legacy_support(self):
        # Pass numpy array directly (legacy)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        patches = split_one_image_with_unfold(img, kernel_size=(48, 48), stride_size=(24, 24))
        self.assertIsInstance(patches, torch.Tensor)
        self.assertEqual(patches.shape[-2:], (224, 224))

    def test_winclip_attention_runs(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock patch_similarity to return correct shapes
        # view(1, 1, patch_num[0], patch_num[1])

        def patch_sim_side_effect(patches, text_embedding, model, device, class_adaption, type_id):
            # Infer expected shape from calls?
            # actually winclip_attention calls .view(...) on the result.
            # So patch_similarity should return a flattened tensor of size patch_num[0]*patch_num[1]
            return torch.randn(1, 9*9) # For sml

        # We need to control the return value per call, or make it generic.
        # winclip_attention calls patch_similarity 3 times.
        # 1. sml: needs (1, sml_patch_num[0]*sml_patch_num[1])
        # 2. mid: needs (1, mid_patch_num[0]*mid_patch_num[1])
        # 3. lge: needs (1, lge_patch_num[0]*lge_patch_num[1])

        # Easier to mock calculate_patches in Config to be fixed, and return fixed size
        self.cfg.sml_patch_num = (10, 10)
        self.cfg.mid_patch_num = (10, 10)
        self.cfg.lge_patch_num = (10, 10)

        with patch('functions.attn_func.patch_similarity') as mock_sim:
            mock_sim.return_value = torch.randn(1, 100) # 10*10

            res = winclip_attention(self.cfg, img, self.text_embedding, self.clip_model, self.device)

            self.assertIsNotNone(res)
            # res should be a PIL Image
            from PIL import Image
            self.assertIsInstance(res, Image.Image)

if __name__ == '__main__':
    unittest.main()
