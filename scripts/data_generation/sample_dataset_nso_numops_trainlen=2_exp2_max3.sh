#!/bin/bash

python src/dataset_generation/generate_boxes_data.py \
    --num_samples 2200 \
    --disjoint_numops \
    --output_dir data/boxes_nso_numops_trainlen=2_exp2_max3 \
    --num_operations 2 \
    --expected_num_items_per_box 2 \
    --max_items_per_box 3
