#!/bin/bash

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets sift-128-euclidean \
#     --metrics l2 \
#     --sizes 500000 \
#     --query_k 50\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name static_recompute_k50 > sift_rolling_update_staticvamana_t56_k50.log \

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets glove-100-angular \
#     --metrics cosine \
#     --sizes 500000 \
#     --query_k 50\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name static_recompute_k50 > glove_rolling_update_staticvamana_t56_k50.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets adversarial-128-euclidean \
#     --metrics l2 \
#     --sizes 500000 \
#     --query_k 50\
#     --experiments small_batch_gradual_update_experiment \
#     --setting_name static_recompute_k50 > adversarial_rolling_update_staticvamana_t56_k50.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets huffpost-256-angular \
#     --metrics cosine \
#     --sizes 50000 \
#     --query_k 50\
#     --experiments small_batch_gradual_update_train_test_split_experiment \
#     --setting_name rolling_update_freshvamana > huffpost_rolling_update_freshvamana_t56.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets redcaps-512-angular \
#     --metrics cosine \
#     --sizes 500000\
#     --query_k 50\
#     --experiments small_batch_gradual_update_train_test_split_experiment \
#     --setting_name rolling_update_freshvamana_param_search > redcaps_rolling_update_freshvamana_t56_param_search.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets spacev-100-euclidean-30m-samples \
#     --metrics l2 \
#     --sizes 500000\
#     --query_k 50\
#     --experiments small_batch_gradual_update_train_test_split_experiment \
#     --setting_name rolling_update_freshvamana > spacev_rolling_update_freshvamana_t56.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets yandextti-200-mips-10m-subset \
#     --metrics mips \
#     --sizes 500000\
#     --query_k 50\
#     --experiments small_batch_gradual_update_train_test_split_experiment \
#     --setting_name rolling_update_freshvamana > yandex_rolling_update_freshvamana_t56.log

# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets redcaps-512-angular \
# #     --metrics cosine \
# #     --sizes 5000000\
# #     --experiments small_batch_gradual_update_train_test_split_experiment \
# #     --setting_name rolling_update_10m > redcaps_rolling_update_10m.log

# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets spacev-100-euclidean-30m-samples \
# #     --metrics l2 \
# #     --sizes 5000000\
# #     --experiments small_batch_gradual_update_train_test_split_experiment \
# #     --setting_name rolling_update > spacev_rolling_update_10m.log

# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets yandextti-200-mips-10m-subset \
# #     --metrics mips \
# #     --sizes 5000000\
# #     --experiments small_batch_gradual_update_train_test_split_experiment \
# #     --setting_name rolling_update > yandex_rolling_update_10m.log



# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets sift-128-euclidean \
# #     --metrics l2 \
# #     --sizes 500000 \
# #     --experiments mixed_throughput_consolidate_experiment \
# #     --setting_name freshvamana_every_batch_default_sync_rolling_update > sift_freshvamana_every_batch_default_sync_rolling_update.log \

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets glove-100-angular \
#     --metrics cosine \
#     --sizes 500000 \
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > glove_freshvamana_every_batch_default_sync_rolling_update.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets adversarial-128-euclidean \
#     --metrics l2 \
#     --sizes 500000 \
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > adversarial_freshvamana_every_batch_default_sync_rolling_update.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets huffpost-256-angular \
#     --metrics cosine \
#     --sizes 50000 \
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > huffpost_freshvamana_every_batch_default_sync_rolling_update.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets redcaps-512-angular \
#     --metrics cosine \
#     --sizes 500000\
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > redcaps_freshvamana_every_batch_default_sync_rolling_update.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets spacev-100-euclidean-30m-samples \
#     --metrics l2 \
#     --sizes 500000\
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > spacev_freshvamana_every_batch_default_sync_rolling_update.log

# python3 experiment_batch_runner.py \
#     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
#     --gt_data_prefix "/storage/sylziyuz/" \
#     --datasets yandextti-200-mips-10m-subset \
#     --metrics mips \
#     --sizes 500000\
#     --experiments mixed_throughput_consolidate_experiment \
#     --setting_name freshvamana_every_batch_default_sync_rolling_update > yandex_freshvamana_every_batch_default_sync_rolling_update.log

# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets redcaps-512-angular \
# #     --metrics cosine \
# #     --sizes 5000000\
# #     --experiments mixed_throughput_consolidate_experiment \
# #     --setting_name freshvamana_every_batch_default_sync_rolling_update_10m > redcaps_freshvamana_every_batch_default_sync_rolling_update_10m.log

# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets spacev-100-euclidean-30m-samples \
# #     --metrics l2 \
# #     --sizes 5000000\
# #     --experiments mixed_throughput_consolidate_experiment \
# #     --setting_name freshvamana_every_batch_default_sync_rolling_update > spacev_freshvamana_every_batch_default_sync_rolling_update_10m.log

# # python3 experiment_batch_runner.py \
# #     --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
# #     --gt_data_prefix "/storage/sylziyuz/" \
# #     --datasets yandextti-200-mips-10m-subset \
# #     --metrics mips \
# #     --sizes 5000000\
# #     --experiments mixed_throughput_consolidate_experiment \
# #     --setting_name freshvamana_every_batch_default_sync_rolling_update > yandex_freshvamana_every_batch_default_sync_rolling_update_10m.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets sift-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > sift_rolling_update_cleann_no_improvement_no_skip_t56_k50_long.log \

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets glove-100-angular \
    --metrics cosine \
    --sizes 50000 \
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > glove_rolling_update_cleann_no_improvement_no_skip_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets adversarial-128-euclidean \
    --metrics l2 \
    --sizes 50000 \
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > adversarial_rolling_update_cleann_no_improvement_no_skip_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets huffpost-256-angular \
    --metrics cosine \
    --sizes 50000 \
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > huffpost_rolling_update_cleann_no_improvement_no_skip_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 500000\
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > redcaps_rolling_update_cleann_no_improvement_no_skip_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets spacev-100-euclidean-30m-samples \
    --metrics l2 \
    --sizes 500000\
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > spacev_rolling_update_cleann_no_improvement_no_skip_t56_k50_long.log

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets yandextti-200-mips-10m-subset \
    --metrics mips \
    --sizes 500000\
    --query_k 50\
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name cleann_no_improvement_no_skip > yandex_rolling_update_cleann_no_improvement_no_skip_t56_k50.log
