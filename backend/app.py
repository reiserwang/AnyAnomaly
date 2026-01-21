from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import random

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Mock response
    data = request.json
    prompt = data.get('prompt', '')
    
    # Simulate processing time
    time.sleep(1)
    
    # Mock data structure matching the user's request
    # Temporal: blue
    # Spatial: red
    
    response_data = {
        "status": "success",
        "prompt": prompt,
        "video_duration": 20.0,
        "storyline": [
            {
                "id": 1,
                "timestamp": 3.5,
                "frame_url": "https://images.unsplash.com/photo-1560252829-804f1aedf1be?q=80&w=600&auto=format&fit=crop", # Running
                "description": "Person running detected",
                "spatial_matches": [
                    {"x": 150, "y": 80, "w": 200, "h": 300} # Red box
                ],
                "temporal_segment": {"start": 2.0, "end": 5.0} # Blue segment
            },
            {
                "id": 2,
                "timestamp": 11.0,
                "frame_url": "https://images.unsplash.com/photo-1543351611-58f69d7c1781?q=80&w=600&auto=format&fit=crop", # Action
                "description": "Suspicious activity",
                "spatial_matches": [
                    {"x": 100, "y": 150, "w": 120, "h": 120}, # Red box 1
                    {"x": 350, "y": 200, "w": 100, "h": 150}  # Red box 2
                ],
                "temporal_segment": {"start": 9.0, "end": 13.0} # Blue segment
            },
            {
                "id": 3,
                "timestamp": 16.5,
                "frame_url": "https://images.unsplash.com/photo-1596727147705-54a9d6c3927d?q=80&w=600&auto=format&fit=crop", # Another
                "description": "Event conclusion",
                "spatial_matches": [], # No spatial match in this frame, maybe just temporal context
                "temporal_segment": {"start": 15.0, "end": 18.0} # Blue segment
            }
        ]
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
