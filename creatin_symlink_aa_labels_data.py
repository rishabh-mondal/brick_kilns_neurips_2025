

import os
import glob
from itertools import combinations
from multiprocessing import Pool, cpu_count

# Define base directories
base_source_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/sentinel"
regions = {
    "seasons1": os.path.join(base_source_dir, "seasons1_lucknow_airshed"),
    "seasons2": os.path.join(base_source_dir, "seasons2_lucknow_airshed"),
    "seasons3": os.path.join(base_source_dir, "seasons3_lucknow_airshed"),
    "seasons4": os.path.join(base_source_dir, "seasons4_lucknow_airshed")
}

# Target directory for symlinked data
target_base_dir = "./symlinked_AA_DATA"
output_dir_format = os.path.join(target_base_dir, "train_{train_regions}_test_{test_region}")

# Function to create symlink for a single file
def create_symlink(args):
    source_path, target_dir, prefix = args
    try:
        file_name = os.path.basename(source_path)
        symlink_name = f"{prefix}{file_name}" if prefix else file_name
        symlink_path = os.path.join(target_dir, symlink_name)
        
        if os.path.exists(symlink_path):
            print(f"Skipping {source_path}: Symlink {symlink_path} already exists")
            return False
        
        os.symlink(source_path, symlink_path)
        print(f"Created symlink: {symlink_path} -> {source_path}")
        return True
    except Exception as e:
        print(f"Error creating symlink for {source_path}: {e}")
        return False

# Function to process a region and return symlink tasks for matched pairs
def process_region(region_path, target_images_dir, target_labels_dir, state_name=""):
    tasks = []
    # Handle state subfolders for India, Afghanistan, Pakistan
    if region_path in [regions["seasons1"], regions["seasons2"], regions["seasons3"]]:
        state_dirs = [d for d in glob.glob(os.path.join(region_path, "*")) if os.path.isdir(d)]
        for state_dir in state_dirs:
            state = os.path.basename(state_dir)
            images_dir = os.path.join(state_dir, "images")
            labels_dir = os.path.join(state_dir, "aa_labels")
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                # Get all image and label basenames
                img_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".png")}
                lbl_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")}
                # Find matching pairs
                matched_files = img_files & lbl_files
                for base_name in matched_files:
                    img_path = os.path.join(images_dir, f"{base_name}.png")
                    lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
                    tasks.append((img_path, target_images_dir, f"{state}_"))
                    tasks.append((lbl_path, target_labels_dir, f"{state}_"))
    # Handle Bangladesh without state subfolders
    elif region_path == regions["bangladesh"]:
        images_dir = os.path.join(region_path, "images")
        labels_dir = os.path.join(region_path, "aa_labels")
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            # Get all image and label basenames
            img_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".png")}
            lbl_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")}
            # Find matching pairs
            matched_files = img_files & lbl_files
            for base_name in matched_files:
                img_path = os.path.join(images_dir, f"{base_name}.png")
                lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
                tasks.append((img_path, target_images_dir, ""))
                tasks.append((lbl_path, target_labels_dir, ""))
    return tasks

# Main function to create symlinks for all combinations with parallel processing
def main():
    # Number of jobs set to 2 (assuming 2 free cores out of 4)
    njobs = 2
    print(f"Using {njobs} parallel jobs")

    # Generate all combinations of 3 regions for train and 1 for test
    region_names = list(regions.keys())
    for train_regions in combinations(region_names, 3):
        test_region = next(iter(set(region_names) - set(train_regions)))
        train_region_paths = [regions[r] for r in train_regions]
        test_region_path = regions[test_region]

        # Define output directories
        output_dir = output_dir_format.format(train_regions="_".join(sorted(train_regions)), test_region=test_region)
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        train_images_dir = os.path.join(train_dir, "images")
        train_labels_dir = os.path.join(train_dir, "labels")
        test_images_dir = os.path.join(test_dir, "images")
        test_labels_dir = os.path.join(test_dir, "labels")

        # Create directories if they don't exist
        for dir_path in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Process train regions
        print(f"Creating symlinks for train regions: {', '.join(train_regions)}")
        train_tasks = []
        for region_path in train_region_paths:
            train_tasks.extend(process_region(region_path, train_images_dir, train_labels_dir))
        with Pool(processes=njobs) as pool:
            pool.map(create_symlink, train_tasks)

        # Process test region
        print(f"Creating symlinks for test region: {test_region}")
        test_tasks = process_region(test_region_path, test_images_dir, test_labels_dir)
        with Pool(processes=njobs) as pool:
            pool.map(create_symlink, test_tasks)

        print(f"Completed symlink creation for {output_dir}")

if __name__ == "__main__":
    main()
