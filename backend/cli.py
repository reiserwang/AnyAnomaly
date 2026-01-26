import argparse
import os
import sys
import json
import torch
import logging

# Ensure backend/core is in path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from cvad_detector import CVADDetector

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="AnyAnomaly Console Mode (CLI)")
    
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--mode", choices=["detect", "summarize"], default="detect", help="Operation mode (default: detect)")
    parser.add_argument("--keywords", "-k", help="Comma-separated list of anomaly keywords (Required for 'detect' mode)")
    parser.add_argument("--quantize", "-q", action="store_true", help="Enable Int4 quantization for memory efficiency")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text", help="Output format (default: text)")
    parser.add_argument("--save-results", help="Path to save the output file (optional)")
    parser.add_argument("--frame-interval", type=int, default=1, help="Process every Nth frame (default: 1)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'detect' and not args.keywords:
        parser.error("--keywords are required for detect mode.")

    # validate input file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Initialize Detector
    logging.info(f"Initializing C-VAD Detector for {args.mode}ion...")
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    try:
        detector = CVADDetector(
            device=device, 
            quantize=args.quantize,
            frame_interval=args.frame_interval
        )
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        sys.exit(1)

    logging.info("Detector initialized. Starting analysis...")
    
    # Run Analysis
    try:
        # Define a simplified callback
        def callback(data):
            if args.output == 'text':
                pass # distinct progress bar implementation could go here, keeping it clean for now
                # logging.info(f"Progress: {data.get('progress')}% - {data.get('message')}")

        if args.mode == 'summarize':
            results = detector.summarize(
                args.video_path,
                callback=callback,
                frame_interval=args.frame_interval
            )
        else:
            results = detector.detect(
                args.video_path, 
                text_prompt=args.keywords, 
                callback=callback,
                frame_interval=args.frame_interval
            )
        
        # Format Output
        output_data = {
            'status': 'success',
            'video_path': args.video_path,
            'mode': args.mode,
            'storyline': results['storyline'],
        }
        
        if args.mode == 'detect':
            output_data['keywords'] = args.keywords
            output_data['scores'] = results['scores']
        else:
            output_data['summary'] = results.get('summary', '')

        if args.output == 'json':
            json_output = json.dumps(output_data, indent=2)
            print(json_output)
            if args.save_results:
                with open(args.save_results, 'w') as f:
                    f.write(json_output)
        else:
            # Text Mode
            print("\n" + "="*40)
            print(f"       {args.mode.upper()} RESULTS       ")
            print("="*40)
            print(f"Video: {args.video_path}")
            if args.mode == 'detect':
                print(f"Keywords: {args.keywords}")
            print("-" * 40)

            if args.mode == 'detect':
                print(f"{'Time':<10} | {'Score':<6} | {'Description'}")
                print("-" * 40)
                
                # Sort storyline by timestamp
                sorted_events = sorted(results['storyline'], key=lambda x: x['timestamp'])
                
                for event in sorted_events:
                    timestamp = f"{event['timestamp']:.2f}s"
                    score = f"{event['score']:.2f}"
                    desc = event.get('reason', 'N/A')
                    print(f"{timestamp:<10} | {score:<6} | {desc}")

            elif args.mode == 'summarize':
                print("Summary:")
                print(results.get('summary', 'No summary generated.'))
                print("-" * 40)
                print("Keyframe Descriptions:")
                
                sorted_events = sorted(results['storyline'], key=lambda x: x['timestamp'])
                for event in sorted_events:
                     timestamp = f"{event['timestamp']:.2f}s"
                     desc = event.get('reason', 'N/A')
                     print(f"[{timestamp}] {desc}")
                
            print("="*40)
            
            if args.save_results:
                with open(args.save_results, 'w') as f:
                    f.write(f"Video: {args.video_path}\nMode: {args.mode}\n")
                    if args.mode == 'detect':
                        f.write(f"Keywords: {args.keywords}\n")
                    elif args.mode == 'summarize':
                        f.write(f"Summary: {results.get('summary', '')}\n")
                        
                    for event in sorted_events:
                        if args.mode == 'detect':
                             f.write(f"{event['timestamp']:.2f}s | Score: {event['score']:.2f} | {event.get('reason', 'N/A')}\n")
                        else:
                             f.write(f"{event['timestamp']:.2f}s | {event.get('reason', 'N/A')}\n")
                             
                print(f"\nResults saved to {args.save_results}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
