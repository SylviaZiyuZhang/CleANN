#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > sift_rolling_update_reverse_edge_naivevamana.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 50000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > glove_rolling_update_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > adversarial_rolling_update_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 5000 \
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > huffpost_rolling_update_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 50000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > redcaps_rolling_update_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 50000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > spacev_rolling_update_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 50000\
    --experiments small_batch_gradual_update_experiment \
    --setting_name reverse_edge_naivevamana > yandex_rolling_update_reverse_edge_naivevamana.log

# ---------------------------------------------------------------------------

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > sift_mixed_throughput_reverse_edge_naivevamana.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 50000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > glove_mixed_throughput_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > adversarial_mixed_throughput_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 5000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > huffpost_mixed_throughput_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 50000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > redcaps_mixed_throughput_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 50000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > spacev_mixed_throughput_reverse_edge_naivevamana.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 50000 \
    --experiments mixed_throughput_experiment \
    --setting_name reverse_edge_naivevamana > yandex_mixed_throughput_reverse_edge_naivevamana.log


# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets redcaps-512-angular \
#     --metrics cosine \
#     --sizes 5000000\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name rolling_update_10m > redcaps_rolling_update_10m.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets spacev-100-euclidean-30m-samples \
#     --metrics l2 \
#     --sizes 5000000\
#     --experiments mixed_throughput_experiment \
#     --setting_name reverse_edge_naivevamana > spacev_mixed_throughput_reverse_edge_naivavamana_10m.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets spacev-100-euclidean-30m-samples \
#     --metrics l2 \
#     --sizes 5000000\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name reverse_edge_naivevamana > spacev_rolling_update_reverse_edge_naivavamana_10m.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets yandextti-200-mips-10m-subset \
#     --metrics mips \
#     --sizes 5000000\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name rolling_update > yandex_rolling_update_10m.log
