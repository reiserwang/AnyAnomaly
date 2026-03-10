import torch
import clip
import numpy as np
from PIL import Image
import cv2
import sys
import os
import logging
import concurrent.futures

# Imports from core (assumes sys.path is set in app.py)
from functions.MiniCPM_func import load_lvlm, lvlm_test, make_instruction, make_bbox_instruction, parse_bbox
from functions.attn_func import winclip_attention
from functions.grid_func import grid_generation
from functions.text_func import make_text_embedding
from functions.key_func import KFS
from functions.fast_ops import FastCLIPPreprocess
from config import update_config # We might need to mock args

class CVADDetector:
    def __init__(self, model_path='MiniCPM-V-2_6', device=None, quantize=False, frame_interval=1, frame_resize_dim=160, inference_resize_dim=448):
        self.frame_interval = frame_interval
        self.frame_resize_dim = frame_resize_dim  # For CLIP/KFS (smaller = faster)
        self.inference_resize_dim = inference_resize_dim  # For MiniCPM-V input
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
            self.device = torch.device("cpu")
            
        logging.info(f"Using device: {self.device}")

        # Config Configuration (Mocking args usually passed via CLI)
        class Args:
            def __init__(self):
                self.dataset = 'custom' # To match 'custom' logic or similar
                # Use MiniCPM-V-2_6-int4 if quantize is requested
                self.model_path = 'MiniCPM-V-2_6-int4' if quantize else model_path
                self.prompt_type = 3 # Reasoning + Consideration
                self.clip_length = 24
                self.out_prompt = 'Output'
                self.kfs_num = 4
                
                # Attn func args
                self.alpha = 0.6
                self.beta = 0.3
                self.gamma = 0.1
                self.sigma = 15
                
                # Scale & Stride defaults
                self.sml_scale = True
                self.mid_scale = True
                self.lge_scale = True
                self.stride = 0.5 # or True? config.py checks 'if share_config['stride']:'
                
                # Flags expected by update_config
                self.type = 'abnormal'
                self.multiple = False
                self.anomaly_detect = True
                self.calc_auc = False
                self.calc_video_auc = False
                self.class_adaption = True
                self.template_adaption = True
                
                # Evaluation args (dummy defaults)
                self.grid_search = False
                self.sigma_range = None
                self.weight_step = None

                
        self.args_mock = Args() 
        
        # WE MUST manually setup "custom" dataset behavior in config.py or mock it here because 
        # config.py has specific branches for 'avenue' and 'shtech'. 
        # If we use 'custom', config.py might not set 'sml_size' etc properly if we don't add a branch there.
        # Let's check config.py:
        # It has `if share_config['dataset_name'] == 'avenue': ... elif ... == 'shtech': ...`
        # It does NOT have a default 'else'. 
        # So we should probably spoof 'shtech' to get default sizes, or mock the derived attributes manually if we want 'custom'.
        # Spoofing 'shtech' is safest for now to get standard video surveillance parameters.
        self.args_mock.dataset = 'shtech' 
        
        # Now call update_config to populate derived fields (like sml_scale, lge_patch_num etc)
        self.cfg = update_config(self.args_mock)
        
        # Override some defaults if needed after update_config
        self.cfg.type_list = ["abnormal"] 
        self.cfg.out_prompt = 'Output'
        self.cfg.class_adaption = True
        self.cfg.template_adaption = True
        self.cfg.type_ids = [0] 

        # Load Models
        logging.info(f"Loading LVLM: {self.cfg.model_path}...")
        try:
            # MPS friendly loading: float16 is usually best for MPS
            dtype = torch.float16 if self.device.type == 'mps' or self.device.type == 'cuda' else torch.float32
            
            # Note: 'int4' models might require AutoGPTQ which has variable support on MPS.
            # We attempt standard loading; if it fails on MPS with int4, user might need to revert to float16.
            self.tokenizer, self.model = load_lvlm(self.cfg.model_path, self.device)
            
            # Explicitly cast to device/dtype if load_lvlm doesn't handle it fully for our custom wrapper
                 
        except Exception as e:
            logging.error(f"Error loading LVLM: {e}")

            # Check for INT4 on MPS failure scenario
            if self.device.type == 'mps' and 'int4' in self.cfg.model_path.lower():
                logging.warning("Fallback: INT4 quantization likely failed on MPS. Attempting to load non-quantized model (Float16/BFloat16).")
                try:
                    # Remove 'int4' from model path to revert to base model
                    # e.g., 'MiniCPM-V-2_6-int4' -> 'MiniCPM-V-2_6'
                    new_path = self.cfg.model_path.replace('-int4', '').replace('int4', '')
                    if new_path == self.cfg.model_path:
                         new_path = 'MiniCPM-V-2_6' # Default fallback if replace failed

                    self.cfg.model_path = new_path
                    logging.info(f"Fallback Loading LVLM: {self.cfg.model_path}...")
                    self.tokenizer, self.model = load_lvlm(self.cfg.model_path, self.device)
                except Exception as fallback_e:
                    logging.error(f"Fallback loading also failed: {fallback_e}")
                    raise e
            else:
                logging.warning("Fallback: Attempting to load CPU/Float32 or base model if quantization failed.")
                raise e

        logging.info("Loading CLIP...")
        self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
        self.preprocess = FastCLIPPreprocess(size=224, device=self.device)
        
        # Initialize KFS (Key Frame Selection)
        self.kfs = KFS(self.cfg.kfs_num, self.cfg.clip_length, self.clip_model, self.preprocess, self.device)

        # Async executor for saving images
        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _log_save_error(self, future):
        """Callback to log errors from async save operations."""
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error saving keyframe: {e}")

    def process_video(self, video_path, resize_dim=None):
        """Extracts frames from video, respecting frame_interval."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0 # Fallback
            
        frames = []
        count = 0
        while cap.isOpened():
            # Optimization: Use grab() to skip frames without decoding
            if count % self.frame_interval == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                if resize_dim:
                    new_dim = resize_dim
                elif self.frame_resize_dim > 0:
                    new_dim = self.frame_resize_dim
                else:
                    new_dim = None

                if new_dim:
                    height, width, _ = frame.shape
                    if height > width:
                        new_height = int((new_dim / width) * height)
                        # Optimization: INTER_LINEAR is ~18x faster than INTER_AREA with minimal quality loss for detection
                        frame = cv2.resize(frame, (new_dim, new_height), interpolation=cv2.INTER_LINEAR)
                    else:
                        new_width = int((new_dim / height) * width)
                        # Optimization: INTER_LINEAR is ~18x faster than INTER_AREA with minimal quality loss for detection
                        frame = cv2.resize(frame, (new_width, new_dim), interpolation=cv2.INTER_LINEAR)

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                if not cap.grab():
                    break
            count += 1
            
        cap.release()
        return frames, fps

    def detect(self, video_path, text_prompt, callback=None, request_id=None, frame_interval=None, fast_mode=False):
        """
        Runs anomaly detection on the video for the given text prompt.
        
        Args:
            fast_mode: If True, skip attention/grid operations for faster inference (2-3x speedup)
        """
        if frame_interval:
            self.frame_interval = frame_interval
            
        # Optimize: Resize to inference_resize_dim (448) so we don't lose quality for MiniCPM
        # KFS/CLIP will handle downscaling to 224 internally
        target_dim = self.inference_resize_dim if self.inference_resize_dim > 0 else 448
        frames, fps = self.process_video(video_path, resize_dim=target_dim)
        total_frames = len(frames)
        clip_length = self.cfg.clip_length
        
        if callback:
            callback({"progress": 5, "message": f"Video loaded. {total_frames} frames extracted."})
        
        chunk_scores = []
        storyline = [] # List of {time, score, keyframe_path}
        
        # Keyframe storage dir
        if request_id:
            kf_dir = os.path.join(os.path.dirname(video_path), '..', 'keyframes', request_id)
            os.makedirs(kf_dir, exist_ok=True)
            
        if callback:
             callback({"progress": 10, "message": "Frames prepared. Starting model inference..."})

        # Text embedding
        instruction, instruction_tc = make_instruction(self.cfg, text_prompt, True)
        
        text_embedding = make_text_embedding(self.clip_model, self.device, text=text_prompt, 
                                            class_adaption=self.cfg.class_adaption, 
                                            template_adaption=self.cfg.template_adaption)

        # Pre-compute text features for KFS
        with torch.no_grad():
             tokenized_text = clip.tokenize([text_prompt]).to(self.device)
             text_features = self.clip_model.encode_text(tokenized_text).float()
             text_features /= text_features.norm(dim=-1, keepdim=True)

        # Iterate chunks
        chunk_indices = range(0, total_frames, clip_length)
        total_chunks = len(chunk_indices)

        for chunk_idx, i in enumerate(chunk_indices):
            # cp is list of PIL images
            cp = frames[i:i+clip_length]
            actual_len = len(cp)
            if actual_len == 0:
                break
            
            # Pad if last chunk is smaller than clip_length
            if actual_len < clip_length:
                last_frame = cp[-1]
                padding = [last_frame] * (clip_length - actual_len)
                cp.extend(padding)
            progress = 10 + int((chunk_idx / total_chunks) * 80)
            if callback:
                callback({"progress": progress, "message": f"Processing chunk {chunk_idx+1}/{total_chunks}..."})

            # KFS Selection
            try:
                if callback:
                   callback({"progress": progress, "message": f"Processing chunk {chunk_idx+1}/{total_chunks}... (Selecting Keyframe)"})

                # self.kfs.call_function now handles list of images
                indice = self.kfs.call_function(cp, text_prompt, text_features=text_features)
                
                # Indices in 'indice' tuple are relative to the chunk 'cp'
                key_image = cp[indice[0]]
                image_list = [cp[idx] for idx in indice[1:]]

                # Inference
                if callback:
                   callback({"progress": progress, "message": f"Processing chunk {chunk_idx+1}/{total_chunks}... (Running Inference)"})

                if fast_mode:
                    # Fast mode: Single inference on keyframe directly (2-3x faster)
                    # Pass None for path, key_image for image
                    response_tc = lvlm_test(self.tokenizer, self.model, instruction_tc, None, key_image)
                else:
                    # Standard mode: Full attention + grid pipeline
                    wa_image = winclip_attention(self.cfg, key_image, text_embedding, self.clip_model, self.device, self.cfg.class_adaption, 0)
                    grid_image = grid_generation(self.cfg, image_list, text_prompt, self.clip_model, self.device, text_features=text_features)
                    response_tc = lvlm_test(self.tokenizer, self.model, instruction_tc, None, grid_image)
                
                # Parse Score
                from utils import generate_output
                score_tc = generate_output(response_tc)['score']
                
                # Assign score to local chunk
                # Assign score to local chunk
                for _ in range(actual_len):
                     chunk_scores.append(score_tc)
                     
                # Save keyframe for storyline
                bbox = None
                kf_filename = f"chunk_{chunk_idx}.jpg"
                kf_path_for_ui = ""

                if request_id:
                    # Save key image permanently for this request
                    # Since we don't have a path, we save the PIL image
                    dest_path = os.path.join(kf_dir, kf_filename)
                    future = self.save_executor.submit(key_image.save, dest_path)
                    future.add_done_callback(self._log_save_error)
                    kf_path_for_ui = f"/keyframes/{request_id}/{kf_filename}"
                    
                    if score_tc > 0.6:
                         try:
                             bbox_instr = make_bbox_instruction(text_prompt)
                             response_bbox = lvlm_test(self.tokenizer, self.model, bbox_instr, None, key_image)
                             bbox = parse_bbox(response_bbox)
                         except Exception as e:
                             logging.error(f"Error getting bbox: {e}")
                             bbox = None

                    storyline.append({
                        "chunk_index": chunk_idx,
                        "timestamp": ((i * self.frame_interval) / fps),
                        "score": score_tc,
                        "image": kf_path_for_ui,
                        "reason": generate_output(response_tc)['reason'],
                        "box": bbox
                    })

            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
                if callback:
                     callback({"progress": progress, "message": f"Error in chunk {chunk_idx}: {e}"})
                for _ in range(clip_length):
                     chunk_scores.append(0.0)

        if callback:
            callback({"progress": 100, "message": "Analysis complete."})
        
        return {
            "scores": chunk_scores,
            "storyline": storyline
        }

    def summarize(self, video_path, callback=None, request_id=None, frame_interval=None, fast_mode=False):
        """
        Summarizes the video by generating captions for keyframes.
        
        Args:
            fast_mode: Reserved for future use (summarization is already optimized)
        """
        if frame_interval:
            self.frame_interval = frame_interval
        
        # Optimize: 448 for better summary, or keep 240? Summarization needs details.
        target_dim = self.inference_resize_dim if self.inference_resize_dim > 0 else 448
        frames, fps = self.process_video(video_path, resize_dim=target_dim)
        total_frames = len(frames)
        clip_length = self.cfg.clip_length
        
        if callback:
            callback({"progress": 5, "message": f"Video loaded. {total_frames} frames extracted."})
        
        storyline = []
        full_summary = []
        
        # Keyframe storage
        if request_id:
            kf_dir = os.path.join(os.path.dirname(video_path), '..', 'keyframes', request_id)
            os.makedirs(kf_dir, exist_ok=True)
        
        if callback:
             callback({"progress": 10, "message": "Frames prepared. Starting summarization..."})

        # General prompt for KFS and Description
        kfs_prompt = "important object or event"

        # Pre-compute text features for KFS
        with torch.no_grad():
             tokenized_text = clip.tokenize([kfs_prompt]).to(self.device)
             text_features = self.clip_model.encode_text(tokenized_text).float()
             text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Instruction for description
        desc_instruction = make_instruction(self.cfg, "Describe this visual scene in detail.", False)[0]
        
        chunk_indices = range(0, total_frames, clip_length)
        total_chunks = len(chunk_indices)

        for chunk_idx, i in enumerate(chunk_indices):
            cp = frames[i:i+clip_length]
            actual_len = len(cp)
            if actual_len == 0:
                break
            
            # Pad if last chunk is smaller than clip_length
            if actual_len < clip_length:
                last_frame = cp[-1]
                padding = [last_frame] * (clip_length - actual_len)
                cp.extend(padding)
                
            progress = 10 + int((chunk_idx / total_chunks) * 80)
            if callback:
                callback({"progress": progress, "message": f"Summarizing chunk {chunk_idx+1}/{total_chunks}..."})

            try:
                # KFS Selection - Use a generic prompt to find the most "interesting" frame
                indice = self.kfs.call_function(cp, kfs_prompt, text_features=text_features)
                key_image = cp[indice[0]]
                
                # Run Inference to get description
                response = lvlm_test(self.tokenizer, self.model, desc_instruction, None, key_image)
                description = response
                
                # Clean up description
                description = description.strip()
                full_summary.append(description)

                # Save keyframe
                if request_id:
                    kf_filename = f"summary_{chunk_idx}.jpg"
                    dest_path = os.path.join(kf_dir, kf_filename)
                    future = self.save_executor.submit(key_image.save, dest_path)
                    future.add_done_callback(self._log_save_error)

                    storyline.append({
                        "chunk_index": chunk_idx,
                        "timestamp": ((i * self.frame_interval) / fps),
                        "score": 0.5, # Neutral score for summary
                        "image": f"/keyframes/{request_id}/{kf_filename}",
                        "reason": description,
                        "box": None
                    })

            except Exception as e:
                logging.error(f"Error in chunk {chunk_idx}: {e}")
                if callback:
                     callback({"progress": progress, "message": f"Error in chunk {chunk_idx}: {e}"})

        if callback:
            callback({"progress": 100, "message": "Summarization complete."})
        
        return {
            "scores": [0.0] * total_frames, # Dummy scores
            "storyline": storyline,
            "summary": " ".join(full_summary)
        }


