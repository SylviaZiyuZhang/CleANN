#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics cosine \
    --sizes 500000 \
    --experiments small_batch_gradual_update_adjust_query_load_experiment \
    --setting_name mem_overhead_3 > mem_overhead_adversarial.log
