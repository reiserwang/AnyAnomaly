#!/bin/bash
# Remove temp frames if any
rm -rf *_frames
rm -rf backend/core/results

# Remove logs
rm -f *.log

# Remove json results
rm -f fast_mode_result.json summarization_result.json

# Remove pycache
find . -name "__pycache__" -type d -exec rm -rf {} +

# Remove system files
find . -name ".DS_Store" -type f -delete

echo "Cleanup complete."
