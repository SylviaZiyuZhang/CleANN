#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > sift_rolling_update_static_recompute.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 500000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > glove_rolling_update_static_recompute.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > adversarial_rolling_update_static_recompute.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 50000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > huffpost_rolling_update_static_recompute.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 500000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > redcaps_rolling_update_static_recompute.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 500000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > spacev_rolling_update_static_recompute.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 500000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > yandex_rolling_update_static_recompute.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/$whoami/" \
#     --datasets redcaps-512-angular \
#     --metrics cosine \
#     --sizes 5000000\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name rolling_update_10m > redcaps_rolling_update_10m.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/$whoami/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 5000000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name rolling_update > spacev_rolling_update_10m.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/$whoami/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/$whoami/" \
#     --datasets yandextti-200-mips-10m-subset \
#     --metrics mips \
#     --sizes 5000000\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name rolling_update > yandex_rolling_update_10m.log
