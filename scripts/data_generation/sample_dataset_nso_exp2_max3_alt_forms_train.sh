#!/bin/bash

python src/dataset_generation/generate_boxes_data.py \
    --num_samples 2200 \
    --output_dir data/boxes_nso_exp2_max3_alt_forms_train \
    --expected_num_items_per_box 2 \
    --max_items_per_box 3\
    --alternative_forms train \
    --num_operations 12 \
    --disjoint_object_vocabulary_file data/objects_not_in_bnc.csv \
    --disjoint_object_splits train 
