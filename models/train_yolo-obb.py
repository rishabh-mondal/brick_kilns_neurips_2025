# import os
# import glob
# import subprocess
# from multiprocessing import Pool
# from pathlib import Path

# # Define directories
# yaml_dir = "/home/suruchi.hardaha/cosmos/ijcai_2025_data/yaml_files"
# output_dir = "/home/suruchi.hardaha/cosmos/ijcai_2025_data/runs"

# # Number of GPUs to use (2 or 3)
# num_gpus = 2 # Adjust to 2 if preferred

# # Function to generate and run nohup command for training and optional testing
# def train_yolo(args):
#     yaml_file, gpu_id = args
#     try:
#         # Get run name from YAML filename
#         run_name = os.path.basename(yaml_file).replace(".yaml", "")
#         run_dir = os.path.join(output_dir, run_name)
        
#         # Create run directory
#         os.makedirs(run_dir, exist_ok=True)
        
#         # Define nohup command for training
#         train_command = [
#             "nohup", "yolo", "train",
#             "model=yolo11l-obb.pt",  # Adjust model size (n, s, m, l, x)
#             f"data={yaml_file}",
#             "epochs=50",              # Adjust as needed
#             "batch=64",               # Increased batch size for 80 GB GPUs
#             "imgsz=128",              # Adjust if needed
#             f"device={gpu_id}",       # Assign specific GPU
#             f"project={output_dir}",
#             f"name={run_name}",
#             "exist_ok=True",
#             "task=obb",
#             # "patience=10",            # Early stopping
#             "save=True",
#             "verbose=True"
#         ]
        
#         # Define log file for training
#         train_log_file = os.path.join(run_dir, "nohup_train.out")
        
#         # Run training command
#         print(f"Starting training for {yaml_file} on GPU {gpu_id}")
#         with open(train_log_file, "w") as f:
#             process = subprocess.Popen(train_command, stdout=f, stderr=f)
#             train_pid = process.pid
#             print(f"Training PID: {train_pid}. Logs saved to {train_log_file}")
        
#         # Wait for training to complete (for testing)
#         process.wait()
        
#         # Run evaluation on test set
#         # test_log_file = os.path.join(run_dir, "nohup_test.out")
#         # test_command = [
#         # #     "nohup", "yolo", "val",
#         # #     f"model={os.path.join(run_dir, 'weights', 'best.pt')}",
#         # #     f"data={yaml_file}",
#         # #     "task=obb",
#         # #     "split=test",
#         # #     f"device={gpu_id}",
#         # #     f"project={output_dir}",
#         # #     f"name={run_name}_test",
#         # #     "exist_ok=True",
#         # #     "verbose=True"
#         # ]
#         print(f"Starting testing for {yaml_file} on GPU {gpu_id}")
#         with open(test_log_file, "w") as f:
#             process = subprocess.Popen(test_command, stdout=f, stderr=f)
#             test_pid = process.pid
#             print(f"Testing PID: {test_pid}. Logs saved to {test_log_file}")
        
#         return True, (train_pid, test_pid)
    
#     except Exception as e:
#         print(f"Error processing {yaml_file} on GPU {gpu_id}: {e}")
#         return False, (None, None)

# # Main function to process all YAML files
# def main():
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get all YAML files
#     yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
#     if not yaml_files:
#         print("No YAML files found in", yaml_dir)
#         return
    
#     # Assign GPUs to tasks (cycle through 0, 1, 2)
#     tasks = [(yaml_file, i % num_gpus) for i, yaml_file in enumerate(yaml_files)]
    
#     # Track results
#     successful_runs = 0
#     failed_runs = 0
#     pids = []
    
#     # Run training jobs in parallel
#     with Pool(processes=num_gpus) as pool:
#         results = pool.map(train_yolo, tasks)
    
#     # Collect results
#     for (success, (train_pid, test_pid)), (yaml_file, _) in zip(results, tasks):
#         if success:
#             successful_runs += 1
#             pids.append((yaml_file, train_pid, test_pid))
#         else:
#             failed_runs += 1
    
#     # Print summary
#     print(f"\nSummary: Started {successful_runs} training runs, {failed_runs} failed")
#     print("Running processes:")
#     for yaml_file, train_pid, test_pid in pids:
#         print(f"YAML: {yaml_file}, Train PID: {train_pid}, Test PID: {test_pid if test_pid else 'Not run'}")

# if __name__ == "__main__":
#     main()

import os
import glob
import subprocess
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import yaml

# Define directories
yaml_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/yaml_files"
output_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/cross_seasons_obb"
os.makedirs(output_dir,exist_ok=True)

# Number of GPUs to use
num_gpus = 2

# Pretrained model
model_path = "yolo11l-obb.pt"  # Update to full path if needed

# Function to check environment and model
def check_environment():
    try:
        print(f"NumPy version: {np.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        from ultralytics import YOLO
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Download from Ultralytics or specify correct path.")
        model = YOLO(model_path)
        print(f"Ultralytics environment check passed. Model {model_path} loaded successfully")
        return True
    except Exception as e:
        print(f"Environment check failed: {e}")
        return False

# Function to generate and run nohup command for training
def train_yolo(args):
    yaml_file, gpu_id = args
    try:
        # Get run name from YAML filename
        run_name = os.path.basename(yaml_file).replace(".yaml", "")
        run_dir = os.path.join(output_dir, run_name)
        
        # Create run directory
        os.makedirs(run_dir, exist_ok=True)
        
        # Define nohup command for training
        train_command = [
            "nohup", "yolo", "train",
            f"model={model_path}",
            f"data={yaml_file}",
            "epochs=50",
            "batch=256",
            "imgsz=128",  # Changed to 640 for better accuracy
            f"device={gpu_id}",
            f"project={output_dir}",
            f"name={run_name}",
            "exist_ok=True",
            "task=obb",
            # "patience=10",
            "save=True",
            "verbose=True"
        ]
        
        # Define log file for training
        train_log_file = os.path.join(run_dir, "nohup_train.out")
        
        # Run training command
        print(f"Starting training for {yaml_file} on GPU {gpu_id}")
        with open(train_log_file, "w") as f:
            process = subprocess.Popen(train_command, stdout=f, stderr=f)
            train_pid = process.pid
            print(f"Training PID: {train_pid}. Logs saved to {train_log_file}")
        
        return True, (train_pid, None)  # No testing performed
    
    except Exception as e:
        print(f"Error processing {yaml_file} on GPU {gpu_id}: {e}")
        return False, (None, None)

# Main function to process all YAML files
def main():
    # Check environment
    if not check_environment():
        print("Exiting due to environment issues. Please fix and retry.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all YAML files
    yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
    if not yaml_files:
        print("No YAML files found in", yaml_dir)
        return
    
    # Assign GPUs to tasks
    tasks = [(yaml_file, i % num_gpus) for i, yaml_file in enumerate(yaml_files)]
    
    # Track results
    successful_runs = 0
    failed_runs = 0
    pids = []
    
    # Run training jobs in parallel
    with Pool(processes=num_gpus) as pool:
        results = pool.map(train_yolo, tasks)
    
    # Collect results
    for (success, (train_pid, test_pid)), (yaml_file, _) in zip(results, tasks):
        if success:
            successful_runs += 1
            pids.append((yaml_file, train_pid, None))  # No test PID
        else:
            failed_runs += 1
    
    # Print summary
    print(f"\nSummary: Started {successful_runs} training runs, {failed_runs} failed")
    print("Running processes:")
    for yaml_file, train_pid, test_pid in pids:
        print(f"YAML: {yaml_file}, Train PID: {train_pid}, Test PID: {test_pid if test_pid else 'Not run'}")

if __name__ == "__main__":
    main()