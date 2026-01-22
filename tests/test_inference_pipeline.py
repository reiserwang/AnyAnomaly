import pytest
import os
import sys
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Add backend paths
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend'))
core_path = os.path.join(backend_path, 'core')
sys.path.append(backend_path)
sys.path.append(core_path)

# Helper to create dummy video
def create_dummy_video(path, frames=30):
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30, (width, height))
    for _ in range(frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

@pytest.fixture(scope="module")
def setup_detector():
    """
    Setup CVADDetector with mocked models.
    """
    # Patch dependencies globally for the module
    with patch('functions.MiniCPM_func.load_lvlm') as mock_load, \
         patch('clip.load') as mock_clip_load, \
         patch('torch.backends.mps.is_available', return_value=False), \
         patch('torch.cuda.is_available', return_value=False):
        
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        
        mock_clip_model = MagicMock()
        mock_preprocess = MagicMock()
        
        def preprocess_side_effect(img):
            # Return a valid tensor shape for CLIP (C, H, W) e.g., (3, 224, 224)
            # We assume CPU for tests
            import torch
            return torch.rand(3, 224, 224)
            
        mock_preprocess.side_effect = preprocess_side_effect
        mock_clip_load.return_value = (mock_clip_model, mock_preprocess)
        
        # Import after patching
        from cvad_detector import CVADDetector
        
        # Initialize
        detector = CVADDetector(device=None, quantize=False, frame_interval=1)
        
        # Mock self.kfs.call_function to return valid indices
        # We need to verify KFS receives images, so we can wrap it or inspect mocks later
        # But for now let's let KFS run (it uses mocked CLIP model)
        # We just need to make sure clip_model.encode_image works.
        
        def mock_encode_image(image):
            # Return random features (batch, 512)
            import torch
            return torch.randn(image.shape[0], 512)
        
        def mock_encode_text(text):
            import torch
            return torch.randn(1, 512)
            
        mock_clip_model.encode_image.side_effect = mock_encode_image
        mock_clip_model.encode_text.side_effect = mock_encode_text
        
        yield detector

@pytest.fixture
def dummy_video(tmp_path):
    video_path = str(tmp_path / "test.mp4")
    create_dummy_video(video_path, frames=30)
    return video_path

def test_process_video_resolution(setup_detector, dummy_video):
    detector = setup_detector
    # Request specific resize
    frames, fps = detector.process_video(dummy_video, resize_dim=448)
    assert len(frames) == 30
    # Check resolution of a frame (PIL Image)
    # 640x480 -> max dim 448
    # 480 is height. 640 is width. 
    # If resize_dim=448. Logic: if height > width ... else frame = cv2.resize(..., (new_width, new_dim))
    # 480 < 640. So else branch. new_width = (448 / 480) * 640 = 0.9333 * 640 = 597.33 -> 597.
    # So expected size is (597, 448).
    w, h = frames[0].size
    assert h == 448
    assert abs(w - 597) <= 1 # Allow rounding diffs

def test_detect_pipeline_in_memory(setup_detector, dummy_video):
    detector = setup_detector
    
    # Mock lvlm_test to avoid actual inference errors and check inputs
    with patch('cvad_detector.lvlm_test') as mock_lvlm:
        # Return a valid response structure
        # generate_output parses 'Score: 0.8\nReason: ...'
        mock_lvlm.return_value = "Score: 0.8\nReason: Test anomaly"
        
        # Run detect
        result = detector.detect(dummy_video, text_prompt="test", fast_mode=True)
        
        # Verify results
        assert 'scores' in result
        assert 'storyline' in result
        assert len(result['scores']) == 24
        # If fast_mode=True, we process chunks.
        # Clip length is 24. Video is 30.
        # Chunk 1: 0-24. Valid.
        # Chunk 2: 24-30 (6 frames). < 24. Skipped.
        # Result: 24 scores.
        
        # Verify inputs to lvlm_test
        # Should be called with an image, not a path
        assert mock_lvlm.called
        args, _ = mock_lvlm.call_args
        # args[3] is image_path (Should be None)
        # args[4] is image (Should be PIL Image)
        assert args[3] is None
        from PIL import Image
        assert isinstance(args[4], Image.Image)

def test_summarize_pipeline_in_memory(setup_detector, dummy_video):
    detector = setup_detector
    
    with patch('cvad_detector.lvlm_test') as mock_lvlm:
        mock_lvlm.return_value = "A video of standard noise."
        
        # Run summarize
        result = detector.summarize(dummy_video)
        
        assert 'summary' in result
        assert result['summary'] == "A video of standard noise."
        
        # Verify inputs
        assert mock_lvlm.called
        args, _ = mock_lvlm.call_args
        assert args[3] is None # path
        from PIL import Image
        assert isinstance(args[4], Image.Image) # image
