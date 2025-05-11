#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > sift_freshvamana_every_batch_default_sync_rolling_update.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 500000 \
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > glove_freshvamana_every_batch_default_sync_rolling_update.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 500000 \
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > adversarial_freshvamana_every_batch_default_sync_rolling_update.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 50000 \
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > huffpost_freshvamana_every_batch_default_sync_rolling_update.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 500000\
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > redcaps_freshvamana_every_batch_default_sync_rolling_update.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 500000\
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > spacev_freshvamana_every_batch_default_sync_rolling_update.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 500000\
    --experiments mixed_throughput_consolidate_experiment \
    --setting_name freshvamana_every_batch_default_sync_rolling_update > yandex_freshvamana_every_batch_default_sync_rolling_update.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets redcaps-512-angular \
#     --metrics cosine \
#     --sizes 5000000\
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update_10m > redcaps_freshvamana_every_batch_default_sync_rolling_update_10m.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets spacev-100-euclidean-30m-samples \
#     --metrics l2 \
#     --sizes 5000000\
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > spacev_freshvamana_every_batch_default_sync_rolling_update_10m.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets yandextti-200-mips-10m-subset \
#     --metrics mips \
#     --sizes 5000000\
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > yandex_freshvamana_every_batch_default_sync_rolling_update_10m.log
