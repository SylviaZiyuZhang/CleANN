"""
Usage: 
python script.py --data_prefix "~/new_path/" --dataset_names dataset1 dataset2 --metrics metric1
"""
import argparse
import numpy as np
from pathlib import Path

import test_redcaps_rolling_update

from test_redcaps_rolling_update import load_or_create_test_data

DATASET_NAME_DICT = {
    "redcaps-512-angular": "redcaps",
    "sift-128-euclidean": "sift",
}

def main():
    # Define default values
    default_gt_data_prefix = Path('~/data/').expanduser()
    default_data_prefix = Path('~/data/new_filtered_ann_datasets/').expanduser()
    default_datasets = ["redcaps-512-angular", "sift-128-euclidean"]
    default_metrics = ["cosine", "l2"]
    default_sizes = [5000]
    default_experiments = ["small_batch_gradual_update_experiment"]
    default_setting_name = "setting_name"

    # Create the parser
    parser = argparse.ArgumentParser(description="Settings for concurrent ANN experiments with CLI.")

    # Add optional arguments with default values
    parser.add_argument('--gt_data_prefix', type=str, default=default_gt_data_prefix, help='Ground truth data path prefix. E.g. /storage/usr')
    parser.add_argument('--data_prefix', type=str, default=default_data_prefix, help='Data path prefix. E.g. /storage/usr')
    parser.add_argument('--datasets', type=str, nargs='+', default=default_datasets, help='List of dataset full names. E.g. redcaps-512-angular')
    parser.add_argument('--metrics', type=str, nargs='+', default=default_metrics, help='List of metrics. E.g. l2, cosine')
    parser.add_argument('--experiments', type=str, nargs='+', default=default_experiments, help='List of experiments to run. E.g. small_batch_gradual_update_experiment')
    parser.add_argument('--sizes', type=int, nargs='+', default=default_sizes, help='List of starting index sizes in int.')
    parser.add_argument('--setting_name', type=str, default=default_setting_name, help='Name of the setting.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle the dataset.')
    parser.add_argument('--random_queries', action='store_true', default=False, help='Whether to use randomized in-distribution queries.')

    # Parse the arguments
    args = parser.parse_args()

    # Process the arguments
    gt_data_prefix = args.gt_data_prefix
    data_prefix_path = Path(args.data_prefix).expanduser()
    datasets = args.datasets
    dataset_names = [DATASET_NAME_DICT[nm] for nm in datasets]
    metrics = args.metrics
    experiments = args.experiments
    sizes = args.sizes
    setting_name = args.setting_name
    shuffle = args.shuffle
    random_queries = args.random_queries

    # Print the parsed arguments
    print("==============================================")
    print(f"Ground Truth Data Prefix: {gt_data_prefix}")
    print(f"Data Prefix Path: {data_prefix_path}")
    print(f"Datasets: {datasets}")
    print(f"Dataset Names: {dataset_names}")
    print(f"Metrics: {metrics}")
    print(f"Sizes: {sizes}")
    print(f"Setting Name: {setting_name}")
    print(f"Shuffle: {shuffle}")
    print("==============================================")
    print("----------- Starting Experiments -------------")

    for i, dataset in enumerate(datasets):
        dataset_name = dataset_names[i]
        data_suffix = "{}/{}.npy".format(dataset_name, dataset) if not shuffle else "{}/{}_shuffled.npy".format(dataset_name, dataset) 
        query_suffix = "{}/{}_queries.npy".format(dataset_name, dataset) if not random_queries else "{}/{}_random_queries.npy".format(dataset_name, dataset)
        try:
            data = np.load(data_prefix_path/data_suffix)
            queries = np.load(data_prefix_path/query_suffix)
        except Exception as ex:
            print(ex)
            data, queries = load_or_create_test_data(data_prefix_path/"{}/{}.hdf5".format(dataset_name, dataset))

        print("Loaded dataset and queries for {}".format(dataset_name))
        for size in sizes:
            for metric in metrics:
                for experiment in experiments:
                    print("Running experiment {} for size {} metric {}".format(experiment, size, metric))
                    experiment_func = getattr(test_redcaps_rolling_update, experiment)
                    experiment_func(
                        data=data,
                        queries=queries,
                        dataset_name=dataset_name,
                        gt_data_prefix=gt_data_prefix,
                        setting_name=setting_name,
                        size=size,
                        metric=metric,
                    )

if __name__ == "__main__":
    main()