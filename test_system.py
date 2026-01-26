import requests
import time
import sys
import os
import json

BASE_URL = "http://localhost:5001"
VIDEO_PATH = "input.mp4"

def wait_for_detector(timeout=600):
    print("Waiting for detector to initialize...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('detector_loaded'):
                    print("\nDetector is ready!")
                    return True
                elif 'error' in data:
                    print(f"\nDetector failed to initialize: {data['error']}")
                    return False
                else:
                    sys.stdout.write(".")
                    sys.stdout.flush()
            else:
                print(f"Health check failed with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("Server not accessible yet...")
        
        time.sleep(5)
    
    print("\nTimeout waiting for detector.")
    return False

def run_test():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return

    print(f"\nrunning analysis on {VIDEO_PATH}...")
    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {'video': f}
            data = {'prompt': 'person falling', 'mode': 'detection'}
            
            # Use stream=True to handle NDJSON stream
            response = requests.post(f"{BASE_URL}/analyze", files=files, data=data, stream=True)
            
            if response.status_code != 200:
                print(f"Analysis failed: {response.status_code}")
                # print(response.text)
                # Try to print JSON error if possible
                try: 
                    print(response.json()) 
                except: 
                    print(response.text)
                return

            print("Analysis started. Streaming results...\n")
            
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"result_{int(time.time())}.json")
            
            full_response = []
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(decoded_line)
                    try:
                        full_response.append(json.loads(decoded_line))
                    except json.JSONDecodeError:
                        pass
            
            with open(output_file, 'w') as f:
                json.dump(full_response, f, indent=2)
            print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"Test execution failed: {e}")

if __name__ == "__main__":
    if wait_for_detector():
        run_test()
    else:
        sys.exit(1)
