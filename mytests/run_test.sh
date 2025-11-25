#!/bin/bash
# Visualize Tom Tracking processed data
DATA_DIR="${1:-tom_tracking_output}"
NUM_SAMPLES="${2:-3}"
python mytests/test_tom_tracking_data.py --data_dir "$DATA_DIR" --num_samples "$NUM_SAMPLES"

