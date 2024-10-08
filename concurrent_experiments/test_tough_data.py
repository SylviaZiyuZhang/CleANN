import numpy as np
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]

data_path = f"data/{dataset_name}"
data, queries, gt_neighbors = parse_ann_benchmarks_hdf5(data_path)

chunk_size = 8
for i in range(0, len(data), chunk_size):
    data[i : i + chunk_size] = data[i] + 0.01 * np.random.normal(
        size=(chunk_size, len(data[0]))
    )

data = data[:100000]

indexing_plan = [(0, i) for i in range(len(data))]
plans = [
    ("Indexing", data, queries, indexing_plan, None),
]

run_dynamic_test(plans, max_vectors=len(data))
