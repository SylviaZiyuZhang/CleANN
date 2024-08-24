#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 5000 50000 500000 \
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > sift_A.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 5000 50000 500000 \
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > glove_A.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 5000 50000 500000 \
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > adversarial_A.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 5000 50000 \
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > huffpost_A.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 5000 50000 500000\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > redcaps_A.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 5000 50000 500000\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > spacev_A.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 5000 50000 500000\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > yandex_A.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 5000000\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > redcaps_A_10m.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 5000000\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > spacev_A_10m.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 5000000\
    --experiments small_batch_gradual_update_train_test_split_experiment \
    --setting_name A > yandex_A_10m.log