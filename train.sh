#!/bin/bash

# Set base variables
name="delhi_airshed"
model="rtdetr-l"  # Options: yolov8l-obb, yolov8x-worldv2, etc.
task="aa"          # Options: obb or aa

# Adjust task if using RT-DETR
if [[ $model == rtdetr-l ]]; then
    task="aa"
fi

# Determine yolo task
if [[ $task == "obb" ]]; then
    yolo_task="obb"
else
    yolo_task="detect"
fi

# Set other configs
log_file_path="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/log_files"
satellite_type="sentinel"
imgsz=128
batch_size=16
epochs=100

# Folder and experiment naming
data_folder="${name}_${task}_labels_${satellite_type}"
data_path="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/crossval/${data_folder}"
experimentName="${data_folder}_model_${model}_epochs_${epochs}"

# Logging the config
echo "Name: $name"
echo "Model: $model"
echo "Task: $task"
echo "YOLO Task: $yolo_task"
echo "Epochs: $epochs"
echo "Data Folder: $data_folder"
echo "Data Path: $data_path"
echo "Experiment Name: $experimentName"
echo "Image Size: $imgsz"

# Training loop for 4-fold cross-validation
for fold in {0..3}; do
    device=$fold
    echo "Starting training for Fold $fold on GPU $device..."

    nohup yolo "$yolo_task" train \
        model="$model.pt" \
        data="$data_path/$fold/data.yml" \
        device="$device" \
        imgsz="$imgsz" \
        batch="$batch_size"\
        epochs="$epochs" \
        val=True \
        cache=True \
        workers=8\
        name="${experimentName}_${fold}_${imgsz}" \
        save=True \
        save_conf=True \
        save_txt=True \
        exist_ok=True\
        > "$log_file_path/${experimentName}_${fold}_${imgsz}.log" 2>&1 &

    echo "Fold $fold fired on GPU $device!"
done
