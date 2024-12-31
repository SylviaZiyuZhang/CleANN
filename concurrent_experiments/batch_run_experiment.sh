#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --query_k 50\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > sift_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 500000 \
    --query_k 50\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > glove_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --query_k 50\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > adversarial_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 50000 \
    --query_k 50\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > huffpost_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets redcaps-512-angular \
#     --metrics cosine \
#     --sizes 500000\
#     --query_k 50\
#     --experiments small_batch_gradual_update_train_test_split_experiment \
#     --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > redcaps_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 500000\
    --query_k 50\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > spacev_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 500000\
    --query_k 50\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch > yandex_rolling_update_reverse_edge_gather_prob10_omp_nested_block_1024_consolidate_every_batch.log

# ---------------------------------------------------------------------------
