import numpy as np
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test, test_static_index

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]

data_path = f"data/{dataset_name}"
data, queries, gt_neighbors = parse_ann_benchmarks_hdf5(data_path)

indexing_plan = [(0, i) for i in range(len(data))]
querying_plan = [(1, i) for i in range(len(queries))]

plans = [
    ("Indexing", data, queries, indexing_plan, None),
    ("Querying", data, queries, querying_plan, None),
]

run_dynamic_test(plans, max_vectors=len(data))
