import pytest
import os
import sys
import json
import logging

# Add backend to path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app import app

TEST_VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/uploads/fighting.mp4'))

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data['status'] == 'running'

def test_analyze_detection(client):
    """Test standard anomaly detection mode."""
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}")

    data = {
        'prompt': 'fighting',
        'mode': 'detection',
        'frame_interval': '30',  # Speed up test by processing fewer frames
        'video': (open(TEST_VIDEO_PATH, 'rb'), 'fighting.mp4')
    }
    
    # We need to simulate a long running request and capture the stream
    # However, for integration test, we might just want to see if it starts and eventually finishes.
    # verify=False for stream? 
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    
    # Parse NDJSON stream
    messages = []
    stream_content = response.data.decode('utf-8')
    for line in stream_content.split('\n'):
        if line.strip():
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
                
    # Check for success message
    success_msg = next((m for m in messages if m.get('type') == 'result'), None)
    if success_msg is None:
        print(f"DEBUG: Messages received: {messages}")
    assert success_msg is not None, "Did not receive result message"
    assert success_msg['data']['status'] == 'success'
    assert 'results' in success_msg['data']
    assert len(success_msg['data']['results']) > 0

def test_analyze_summarization(client):
    """Test video summarization mode."""
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}")

    data = {
        'mode': 'summarize',
        'frame_interval': '30',  # Speed up test by processing fewer frames
        'video': (open(TEST_VIDEO_PATH, 'rb'), 'fighting.mp4')
    }
    
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    
    messages = []
    stream_content = response.data.decode('utf-8')
    for line in stream_content.split('\n'):
        if line.strip():
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    success_msg = next((m for m in messages if m.get('type') == 'result'), None)
    assert success_msg is not None, "Did not receive result message"
    assert success_msg['data']['status'] == 'success'
    assert 'summary' in success_msg['data']
    assert len(success_msg['data']['summary']) > 0
    print(f"Summary generated: {success_msg['data']['summary']}")
