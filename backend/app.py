import os
import sys
import json
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from flask import send_from_directory

# Ensure backend/core is in path to import modules from Paper-AnyAnomaly
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from cvad_detector import CVADDetector

import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Initialize Detector (Global Load)
try:
    logging.info("Initializing C-VAD Detector (this may take a while)...")
    
    # Read optimization flags from environment
    # Read optimization flags from environment
    quantize_model = os.environ.get('QUANTIZE_MODEL', 'false').lower() == 'true'
    frame_interval = int(os.environ.get('FRAME_INTERVAL', 1))
    frame_resize_dim = int(os.environ.get('FRAME_RESIZE_DIM', 240))
    
    # Read device from environment
    inference_device_str = os.environ.get('INFERENCE_DEVICE', 'auto').lower()
    device = None
    if inference_device_str != 'auto':
        try:
            device = torch.device(inference_device_str)
        except Exception as e:
            logging.warning(f"Invalid device '{inference_device_str}', falling back to auto-detection. Error: {e}")
            device = None
            
    logging.info(f"Configuration: DEVICE={inference_device_str}, QUANTIZE={quantize_model}, FRAME_INTERVAL={frame_interval}")
    
    detector = CVADDetector(device=device, quantize=quantize_model, frame_interval=frame_interval, frame_resize_dim=frame_resize_dim)
    logging.info("C-VAD Detector Initialized.")
except Exception as e:
    logging.error(f"Failed to initialize detector: {e}")
    detector = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if not detector:
        return jsonify({'error': 'Detector not initialized'}), 500

    file = request.files.get('video')
    text_prompt = request.form.get('prompt', '')
    youtube_url = request.form.get('youtube_url', '')
    mode = request.form.get('mode', 'detection')

    if not text_prompt and mode != 'summarize':
        return jsonify({'error': 'No text prompt provided'}), 400
        
    filepath = None
    
    # Generate unique request ID for this session
    import uuid
    request_id = str(uuid.uuid4())
    
    if youtube_url:
        # Will be downloaded in worker thread
        pass

    elif file and file.filename != '':
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
             return jsonify({'error': 'Invalid file type'}), 400
    else:
        return jsonify({'error': 'No video file or YouTube URL provided'}), 400

    from flask import Response, stream_with_context
    import queue
    import threading
    
    # Queue for communicating between thread and generator
    msg_queue = queue.Queue()
    
    # Extract parameters from request before starting thread
    frame_interval_arg = int(request.form.get('frame_interval', 0))
    mode_arg = request.form.get('mode', 'detection')
    fast_mode_arg = request.form.get('fast_mode', 'false').lower() == 'true'
    turbo_arg = request.form.get('turbo', 'false').lower() == 'true'

    def worker():
        try:
            target_filepath = filepath

            def callback(data):
                # Synchronize server console
                # logging.info(f"[PROGRESS] {data.get('progress', 0)}%: {data.get('message', '')}")
                msg_queue.put({"type": "progress", "data": data})

            if youtube_url:
                try:
                    import yt_dlp
                    callback({"progress": 0, "message": "Downloading video from YouTube..."})
                    # Create a temporary file path
                    temp_name = f"yt_{os.urandom(8).hex()}"
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',
                        'outtmpl': os.path.join(UPLOAD_FOLDER, f"{temp_name}.%(ext)s"),
                        'noplaylist': True,
                        'quiet': True,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(youtube_url, download=True)
                        filename = ydl.prepare_filename(info)
                        target_filepath = filename
                    callback({"progress": 5, "message": "Download complete. Starting analysis..."})
                except Exception as e:
                    msg_queue.put({"type": "error", "message": f"Failed to download YouTube video: {str(e)}"})
                    return

            if not target_filepath:
                msg_queue.put({"type": "error", "message": "No valid video file to process"})
                return

            # Use captured variables
            final_frame_interval = frame_interval_arg
            
            # Turbo mode: Use higher frame_interval for speed
            if turbo_arg and final_frame_interval < 60:
                final_frame_interval = 60
                logging.info("Turbo mode enabled: frame_interval set to 60")
            
            if mode_arg == 'summarize':
                 if final_frame_interval > 0:
                     results = detector.summarize(target_filepath, callback=callback, request_id=request_id, frame_interval=final_frame_interval, fast_mode=fast_mode_arg)
                 else:
                     results = detector.summarize(target_filepath, callback=callback, request_id=request_id, fast_mode=fast_mode_arg)
            else:
                 if final_frame_interval > 0:
                      results = detector.detect(target_filepath, text_prompt, callback=callback, request_id=request_id, frame_interval=final_frame_interval, fast_mode=fast_mode_arg)
                 else:
                      results = detector.detect(target_filepath, text_prompt, callback=callback, request_id=request_id, fast_mode=fast_mode_arg)
            
            # Send final results
            msg_queue.put({
                "type": "result", 
                "data": {
                    'status': 'success',
                    'prompt': text_prompt if mode_arg == 'detection' else 'Video Summary',
                    'results': results['scores'],
                    'storyline': results['storyline'],
                    'summary': results.get('summary', ''),
                    'filename': os.path.basename(target_filepath)
                }
            })
        except Exception as e:
            msg_queue.put({"type": "error", "message": str(e)})
        finally:
            msg_queue.put(None) # Sentinel to stop

    thread = threading.Thread(target=worker)
    thread.start()

    def generate():
        while True:
            msg = msg_queue.get()
            if msg is None:
                break
            # NDJSON format
            yield json.dumps(msg) + "\n"
            
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'detector_loaded': detector is not None})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/keyframes/<path:filename>')
def keyframe_file(filename):
    # Security: Ensure we are only serving from the keyframes directory
    # Simplified approach for demo:
    KEYFRAMES_DIR = os.path.join(os.path.dirname(__file__), 'keyframes')
    return send_from_directory(KEYFRAMES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
