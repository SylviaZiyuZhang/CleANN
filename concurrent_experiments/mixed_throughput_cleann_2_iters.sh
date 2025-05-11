#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > sift_rolling_update_mixed_cleann_c_sensitivity_t56_k50_long.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 50000 \
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > glove_rolling_update_mixed_cleann_c_sensitivity_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > adversarial_rolling_update_mixed_cleann_c_sensitivity_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 50000 \
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > huffpost_rolling_update_mixed_cleann_c_sensitivity_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 500000\
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > redcaps_rolling_update_mixed_cleann_c_sensitivity_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 500000\
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > spacev_rolling_update_mixed_cleann_c_sensitivity_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 500000\
    --query_k 50\
    --experiments mixed_throughput_experiment \
    --setting_name cleann_rolling_update_2_iters > yandex_rolling_update_mixed_cleann_c_sensitivity_t56_k50.log

# Make sure that mixed_throughput_experiment takes n_iters=2