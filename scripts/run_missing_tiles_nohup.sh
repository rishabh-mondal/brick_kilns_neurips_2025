#!/bin/bash

# Set timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
state_name="jharkhand"
# Log file path
log_file="missing_tiles_download_$state_name$timestamp.log"

# Run the Python script with nohup and redirect output to the log file
nohup python /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/scripts/run_missing_tiles_download.py > "$log_file" 2>&1 &
