#!/bin/bash

python src/dataset_generation/generate_boxes_data.py \
    --num_samples 2200 \
    --output_dir data/boxes_nso_zero_shot_exp2_max3 \
    --num_operations 12 \
    --expected_num_items_per_box 2 \
    --max_items_per_box 3 \
    --zero_shot

