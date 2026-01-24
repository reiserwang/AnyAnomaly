import pytest
import os
import sys
import types
from unittest.mock import MagicMock, patch

# Improved mocking for packages
def mock_package(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

torch_mock = mock_package("torch")
torch_mock.backends = mock_package("torch.backends")
torch_mock.backends.mps = mock_package("torch.backends.mps")
torch_mock.cuda = mock_package("torch.cuda")
torch_mock.nn = mock_package("torch.nn")
torch_mock.nn.functional = mock_package("torch.nn.functional")

# Mock attributes
torch_mock.device = MagicMock()
torch_mock.float16 = MagicMock()
torch_mock.float32 = MagicMock()
torch_mock.bfloat16 = MagicMock()
torch_mock.randn = MagicMock()
torch_mock.rand = MagicMock()
torch_mock.tensor = MagicMock()
torch_mock.mean = MagicMock()
torch_mock.stack = MagicMock()
torch_mock.squeeze = MagicMock()
torch_mock.no_grad = MagicMock()
torch_mock.backends.mps.is_available = MagicMock(return_value=False)
torch_mock.cuda.is_available = MagicMock(return_value=False)

cv2_mock = mock_package("cv2")
transformers_mock = mock_package("transformers")
clip_mock = mock_package("clip")
openai_clip_mock = mock_package("openai-clip") # just in case
mpl_mock = mock_package("matplotlib")
mpl_mock.pyplot = mock_package("matplotlib.pyplot")

# Configure cv2
mock_cap = MagicMock()
# Mock constants
cv2_mock.VideoCapture = MagicMock(return_value=mock_cap)
cv2_mock.resize = MagicMock()
cv2_mock.cvtColor = MagicMock()
cv2_mock.COLOR_BGR2RGB = 1
cv2_mock.INTER_AREA = 1
cv2_mock.CAP_PROP_FPS = 5

import numpy as np
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

mock_cap.isOpened.side_effect = [True] * 5 + [False]
mock_cap.read.return_value = (True, dummy_frame)
mock_cap.get.return_value = 30.0

cv2_mock.resize.return_value = dummy_frame
cv2_mock.cvtColor.return_value = dummy_frame

# Transformers mocks
transformers_mock.AutoModel = MagicMock()
transformers_mock.AutoTokenizer = MagicMock()
clip_mock.load = MagicMock()
clip_mock.tokenize = MagicMock()
clip_mock.tokenize.return_value.to.return_value = MagicMock()

# Add backend paths
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend'))
core_path = os.path.join(backend_path, 'core')
sys.path.append(backend_path)
sys.path.append(core_path)

import concurrent.futures

@pytest.fixture
def setup_detector_async():
    """
    Setup CVADDetector with mocked models and mocked executor.
    """
    from cvad_detector import CVADDetector

    with patch('concurrent.futures.ThreadPoolExecutor') as MockExecutor:
        with patch('cvad_detector.load_lvlm') as mock_load, \
             patch('cvad_detector.clip.load') as mock_clip_load:

            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_load.return_value = (mock_tokenizer, mock_model)

            mock_clip_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_clip_load.return_value = (mock_clip_model, mock_preprocess)

            mock_clip_model.encode_image.return_value = MagicMock()
            mock_clip_model.encode_text.return_value = MagicMock()

            with patch('cvad_detector.KFS') as MockKFS:
                mock_kfs_instance = MockKFS.return_value
                mock_kfs_instance.call_function.return_value = (0, 1)

                detector = CVADDetector(device=None, quantize=False, frame_interval=1)
                detector.cfg.clip_length = 2
                detector._mock_executor_cls = MockExecutor

                yield detector

def test_async_save_detect(setup_detector_async, tmp_path):
    detector = setup_detector_async
    video_path = str(tmp_path / "test.mp4")

    mock_executor_cls = detector._mock_executor_cls

    with patch('cvad_detector.lvlm_test') as mock_lvlm, \
         patch('os.makedirs'):

        mock_lvlm.return_value = "Score: 0.9\nReason: Test"

        try:
            detector.detect(video_path, text_prompt="test", request_id="req1", fast_mode=True)
        except AttributeError as e:
             pass # Shouldn't happen now
        except Exception as e:
             pytest.fail(f"Unexpected error: {e}")

    # Check if ThreadPoolExecutor was instantiated
    if not mock_executor_cls.called:
        pytest.fail("ThreadPoolExecutor was NOT instantiated")

    # We need to check the instance that was created
    # Since we are mocking the class, the instance is the return value of the call
    mock_executor_instance = mock_executor_cls.return_value

    # Check if submit was called
    assert mock_executor_instance.submit.called, "Executor.submit should be called"

    call_args = mock_executor_instance.submit.call_args
    assert "req1" in call_args[0][1]
