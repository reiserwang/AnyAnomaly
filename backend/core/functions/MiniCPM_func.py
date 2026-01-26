import torch
import os
import sys
import re
import logging
from PIL import Image
from transformers import AutoModel, AutoTokenizer
        

def make_instruction(cfg, keyword, temporal_context=False):
    # simple
    if cfg.prompt_type == 0:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
        )

    # complex (+ consideration)
    elif cfg.prompt_type == 1:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
        )

    # complex (+ reasoning)
    elif cfg.prompt_type == 2:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )

    # complex (+ reasoning, consideration)
    elif cfg.prompt_type == 3:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )
    
    if temporal_context == False:
        return instruction
    
    # insturction for temporal context
    else:
        # simple
        if cfg.prompt_type == 0:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
            )

        # complex (+ consideration)
        elif cfg.prompt_type == 1:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
            )

        # complex (+ reasoning)
        elif cfg.prompt_type == 2:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
            )

        # complex (+ reasoning, consideration)
        elif cfg.prompt_type == 3:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"    
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
            )

        return instruction, tc_instruction
    
def make_bbox_instruction(keyword):
    """
    Generates a prompt to ask the model for bounding boxes.
    """
    return f"Please detect **{keyword}** in this image. Provide the bounding box coordinates as [ymin, xmin, ymax, xmax] normalized to 0-1000. If there are multiple instances, provide the most prominent one."

def parse_bbox(response_text):
    """
    Parses bounding box from model response. 
    Expected format: [ymin, xmin, ymax, xmax] or similar text.
    Returns: [ymin, xmin, ymax, xmax] normalized to 0-1 (float) or None.
    """
    try:
        # Look for pattern like [123, 456, 789, 101]
        # MiniCPM often outputs <box>ymin, xmin, ymax, xmax</box> or just [y, x, y, x]
        # Let's try to find a list of 4 integers.
        matches = re.findall(r'(\d{1,4})', response_text)
        if len(matches) >= 4:
            # Take the last 4 numbers found, assuming they are the box coords if mixed with other text
            # Or better, look for the first sequence of 4
            coords = [int(x) for x in matches[-4:]]
            
            # Normalize 0-1000 to 0-1
            return [x / 1000.0 for x in coords]
    except Exception:
        pass
    return None

def load_lvlm(model_path, device):
    # Determine precision and repo name
    # Prioritize passed model_path, then environment variable
    env_precision = os.environ.get('MODEL_PRECISION', '').lower()
    
    if 'int4' in model_path.lower() or env_precision == 'int4':
        repo_name = 'openbmb/MiniCPM-V-2_6-int4'
        dtype = torch.float16 # Compute dtype for INT4
        logging.info("Configuration: Using INT4 quantized model.")
    elif 'mini' in model_path.lower(): # Check if standard model path checking
         repo_name = 'openbmb/MiniCPM-V-2_6'
         dtype = torch.bfloat16 if device.type == 'cuda' or device.type == 'mps' else torch.float32
         logging.info(f"Configuration: Using BF16 model on {device.type}.")
    else:
        # Fallback based on env only if model_path is not specific
        if env_precision == 'int4':
             repo_name = 'openbmb/MiniCPM-V-2_6-int4'
             dtype = torch.float16
        else:
             repo_name = 'openbmb/MiniCPM-V-2_6'
             dtype = torch.bfloat16 if device.type == 'cuda' or device.type == 'mps' else torch.float32
             logging.info(f"Configuration: Using BF16 (Env/Default) on {device.type}.")

    logging.info(f"Loading model from {repo_name} with trust_remote_code=True on {device}")
    
    # Load model
    logging.info("DEBUG: Calling AutoModel.from_pretrained with low_cpu_mem_usage=True...")
    
    # Determine device_map to use accelerate's optimization
    # For MPS, 'auto' or explicit device can help with low_cpu_mem_usage
    if device.type == 'cuda':
        d_map = device.type
    elif device.type == 'mps':
        d_map = 'auto' # Accelerate supports MPS
    else:
        d_map = None

    try:
        model = AutoModel.from_pretrained(
            repo_name, 
            trust_remote_code=True, 
            attn_implementation='sdpa', 
            torch_dtype=dtype,
            device_map=d_map,
            low_cpu_mem_usage=True 
        )
    except Exception as e:
        logging.warning(f"Failed to load with device_map={d_map}: {e}. Falling back to default.")
        model = AutoModel.from_pretrained(
            repo_name, 
            trust_remote_code=True, 
            attn_implementation='sdpa', 
            torch_dtype=dtype,
            low_cpu_mem_usage=True # Still try this
        )
        
    logging.info("DEBUG: AutoModel.from_pretrained returned.")
    
    logging.info("DEBUG: Calling AutoTokenizer.from_pretrained...")
    tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
    logging.info("DEBUG: AutoTokenizer.from_pretrained returned.")
    
    # Explicit .to(device) if needed (if device_map didn't put it there)
    if not hasattr(model, 'hf_device_map') and model.device.type != device.type:
        logging.info(f"DEBUG: Moving model to device {device}...")
        model = model.to(device=device)
        logging.info(f"DEBUG: Model moved to {device}.")
        
    logging.info("DEBUG: Setting model to eval mode...")
    model = model.eval()
    logging.info("DEBUG: Model set to eval.")
    return tokenizer, model


def lvlm_test(tokenizer, model, qs, image_path, image=None):
    if image is None:
        image = Image.open(image_path)
    
    image = image.convert('RGB')
    
    msgs = [{'role': 'user', 'content': qs}]
    
    answer = model.chat(
        image=image,
        msgs=msgs,
        tokenizer = tokenizer,
        sampling=False,
        temperature=0.7,
    )
    
    return answer