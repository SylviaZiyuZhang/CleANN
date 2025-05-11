#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --query_k 10 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name eager_mem > adversarial_eager_mem.log
python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 500000 \
    --query_k 10 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name eager_mem > spacev_eager_mem.log
python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 500000 \
    --query_k 10 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name eager_mem > glove_eager_mem.log
