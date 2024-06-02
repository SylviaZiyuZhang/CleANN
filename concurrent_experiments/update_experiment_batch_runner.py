"""
Usage: 
python script.py --data_prefix "~/new_path/" --dataset_names dataset1 dataset2 --metrics metric1
"""
import argparse
from pathlib import Path

from test_redcaps_rolling_update import (
    small_batch_gradual_update_experiment,
    small_batch_gradual_update_insert_only_experiment,
    static_recall_experiment,
)

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
    default_setting_name = "setting_name"

    # Create the parser
    parser = argparse.ArgumentParser(description="Settings for concurrent ANN experiments with CLI.")

    # Add optional arguments with default values
    parser.add_argument('--gt_data_prefix', type=str, default=default_gt_data_prefix, help='Ground truth data prefix path.')
    parser.add_argument('--data_prefix', type=str, default=default_data_prefix, help='Data prefix path.')
    parser.add_argument('--datasets', type=str, nargs='+', default=default_datasets, help='List of dataset full names. E.g. redcaps-512-angular')
    parser.add_argument('--metrics', type=str, nargs='+', default=default_metrics, help='List of metrics.')
    parser.add_argument('--sizes', type=int, nargs='+', default=default_sizes, help='List of sizes.')
    parser.add_argument('--setting_name', type=str, default=default_setting_name, help='Name of the setting.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle the dataset.')
    parser.add_argument('--random_queries', action='store_true', default=False, help='Whether to use randomized in-distribution queries.')

    # Parse the arguments
    args = parser.parse_args()

    # Process the arguments
    gt_data_prefix = Path(args.gt_data_prefix).expanduser()
    data_prefix = Path(args.data_prefix).expanduser()
    datasets = args.datasets
    dataset_names = [DATASET_NAME_DICT[nm] for nm in datasets]
    metrics = args.metrics
    sizes = args.sizes
    setting_name = args.setting_name
    shuffle = args.shuffle
    random_queries = args.random_queries

    # Print the parsed arguments
    print("==============================================")
    print(f"Ground Truth Data Prefix: {gt_data_prefix}")
    print(f"Data Prefix: {data_prefix}")
    print(f"Datasets:" {datasets})
    print(f"Dataset Names: {dataset_names}")
    print(f"Metrics: {metrics}")
    print(f"Sizes: {sizes}")
    print(f"Setting Name: {setting_name}")
    print(f"Shuffle: {shuffle}")
    print("==============================================")
    print("----------- Starting Experiment --------------")

    for i, dataset in enumerate(datasets):
        dataset_name = dataset_names[i]
        data_suffix = "{}/{}.npy".format(dataset_name, dataset) if not shuffle else "{}/{}_shuffled.npy".format(dataset_name, dataset) 
        query_suffix = "{}/{}_queries.npy".format(dataset_name, dataset) if not randomize_queries else "{}/{}_random_queries.npy".format(dataset_name, dataset)
        data = np.load(data_prefix/data_suffix)
        queries = np.load(data_prefix/data_suffix)
        print("Loaded dataset and queries for {}".format(dataset_name))
        for size in sizes:
            for metric in metrics:
                small_batch_gradual_update_experiment(
                    data=data,
                    queries=queries,
                    dataset_name=dataset_name,
                    gt_data_prefix=gt_data_prefix,
                    setting_name=setting_name,
                    size=size,
                    metric=metric,
                    randomize_queries=False
                )

if __name__ == "__main__":
    main()
