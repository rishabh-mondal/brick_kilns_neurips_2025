

import os
import glob
import subprocess
import torch
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import yaml

# Define directories
yaml_dir = "/home/suruchi.hardaha/cosmos/ijcai_2025_data/symlinked_aa_yaml"
output_dir = "/home/suruchi.hardaha/cosmos/ijcai_2025_data/runs_rtdetr"

# Number of GPUs to use
num_gpus = 2

# Pretrained model
model_path = "rtdetr-l.pt"  # Update to full path if needed

# Function to check environment and model
def check_environment():
    try:
        print(f"NumPy version: {np.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU support required.")
        num_gpus_available = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus_available}")
        for i in range(num_gpus_available):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        if num_gpus_available < 4:
            raise RuntimeError("At least 4 GPUs required, but fewer detected. GPUs 0, 1, 2, and 3 are needed, with 0 and 3 in use.")
        # Check if GPUs 0 and 3 are in use
        if torch.cuda.memory_allocated(0) == 0:
            print("Warning: GPU 0 appears to be free, but will not be used as requested.")
        if torch.cuda.memory_allocated(3) == 0:
            print("Warning: GPU 3 appears to be free, but will not be used as requested.")
        from ultralytics import RTDETR
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Download from Ultralytics or specify correct path.")
        # Explicitly load model on a dummy device to avoid GPU 0 initialization
        device = torch.device("cpu")  # Load initially on CPU
        model = RTDETR(model_path).to(device)
        print(f"Ultralytics environment check passed. Model {model_path} loaded successfully on CPU")
        return True
    except Exception as e:
        print(f"Environment check failed: {e}")
        return False

# Function to generate and run nohup command for training
def train_rtdetr(args):
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
            "batch=32",
            "imgsz=128",  # Set to 128x128 as requested
            f"device={gpu_id}",  # Set to GPU 1 or 2
            f"project={output_dir}",
            f"name={run_name}",
            "exist_ok=True",
            "task=detect",  # RT-DETR supports detect task for AA labels
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
    
    # Assign tasks to GPUs 1 and 2 only
    tasks = [(yaml_file, 1 + (i % 2)) for i, yaml_file in enumerate(yaml_files)]  # Alternates between GPU 1 and 2
    
    # Track results
    successful_runs = 0
    failed_runs = 0
    pids = []
    
    # Run training jobs in parallel (using 2 GPUs)
    with Pool(processes=num_gpus) as pool:
        results = pool.map(train_rtdetr, tasks)
    
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
