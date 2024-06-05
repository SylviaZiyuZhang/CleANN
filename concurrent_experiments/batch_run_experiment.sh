#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "~/data/new_filtered_ann_datasets/" \
    --gt_data_prefix "~/data/" \
    --datasets redcaps-512-angular \
    --metrics l2 \
    --sizes 5000 50000 \
    --experiments small_batch_gradual_update_experiment static_recall_experiment \
    --setting_name test \