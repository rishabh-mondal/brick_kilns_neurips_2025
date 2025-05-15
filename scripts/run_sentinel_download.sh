#!/bin/bash

# Set region name
region_name="afghanistan"

# Navigate to the script directory
cd /home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/scripts || exit

# Create log directory if it doesn't exist
log_dir="data_downloading_log"

# Run the script using nohup and save logs under the log directory
nohup python sentinel_tile_bulk_download.py > "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/${log_dir}/sentinel_tile_download_${region_name}.log" 2>&1 &
                                                               