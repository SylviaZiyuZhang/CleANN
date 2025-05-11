#!/bin/bash

python3 experiment_batch_runner.py \
    --data_prefix "/storage/sylziyuz/new_filtered_ann_datasets/" \
    --gt_data_prefix "/storage/sylziyuz/" \
    --datasets redcaps-512-angular \
    --metrics cosine \
    --sizes 500000 \
    --experiments small_batch_gradual_update_split_long_running_experiment \
    --setting_name corrected_dynamic_consolidate_no_fix_restriction_not_search_no_skip > corrected_dynamic_consolidate_no_fix_restriction_not_search_no_skip.log
