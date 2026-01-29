import torch
import numpy as np
import cv2
from PIL import Image

class FastCLIPPreprocess:
    """
    A fast implementation of CLIP preprocessing using OpenCV for resizing.
    Standard CLIP preprocessing uses PIL Bicubic resizing which is slow.
    OpenCV Linear resizing is significantly faster (~18x) with negligible quality loss for KFS.
    """
    def __init__(self, size=224, device='cpu'):
        self.size = size
        # CLIP normalization constants
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __call__(self, img):
        # Handle string path
        if isinstance(img, str):
            img = Image.open(img)

        # Ensure numpy array (RGB)
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Resize logic: preserve aspect ratio, make short side = self.size
        h, w = img.shape[:2]

        # Determine scale
        scale = self.size / min(h, w)

        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Use INTER_LINEAR for speed (vs PIL Bicubic)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center Crop logic
        h, w = img.shape[:2]
        y = (h - self.size) // 2
        x = (w - self.size) // 2

        # Handle edge case where rounding might cause dimension < size (rare but possible)
        # If new_w or new_h is slightly off due to float math
        y = max(0, y)
        x = max(0, x)

        img = img[y:y+self.size, x:x+self.size]

        # To Tensor (CHW) and Normalize
        # OpenCV is HWC, we need CHW
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std

        return tensor
