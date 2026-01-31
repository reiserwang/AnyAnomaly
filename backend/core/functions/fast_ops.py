import torch
import numpy as np
import cv2
from PIL import Image

class FastCLIPPreprocess:
    def __init__(self, size=224, device="cpu"):
        self.size = size
        self.device = device
        # CLIP mean and std
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __call__(self, image):
        # Convert PIL to Numpy
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Image is assumed to be RGB (from PIL)

        # Resize
        h, w = image.shape[:2]

        # Resize shortest side to self.size
        if h < w:
            new_h = self.size
            new_w = int(w * self.size / h)
        else:
            new_w = self.size
            new_h = int(h * self.size / w)

        # Optimization: INTER_LINEAR is significantly faster than standard bicubic
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center Crop
        h, w = image.shape[:2]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        image = image[top:top+self.size, left:left+self.size]

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # To Tensor (CHW)
        image = image.transpose(2, 0, 1)

        # Create tensor and move to device
        return torch.from_numpy(image).to(self.device)
