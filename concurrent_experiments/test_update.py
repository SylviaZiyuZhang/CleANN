import numpy as np
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]

data_path = f"data/{dataset_name}"
data, queries, gt_neighbors, gt_dists = parse_ann_benchmarks_hdf5(data_path)

def augment_data_with_random_clusters(data, dist="normal", alpha=0.01, chunk_size=8):
    # TODO: data should be passed with reference
    if dist == "normal":
        for i in range(0, len(data), chunk_size):
            # TODO: fix the variance of these normal distributions
            data[i : i + chunk_size] = data[i] + alpha * np.random.normal(
                size=(chunk_size, len(data[0]))
            )

def half_dataset_update_experiment(data, queries, gt_neighbors, gt_dists):
    #data_1 = data[:500000]
    #data_2 = data[500000:1000000]

    data_1 = data[:50000]
    data_2 = data[50000:100000]
    data = data[:100000]

    #data_1 = data[:500000]
    #data_2 = data[500000:1000000]
    #data = data[:1000000]

    # indexing_plan = [(0, i) for i in range(len(data_1))]
    indexing_plan = [(0, i) for i in range(len(data))]
    # initial_lookup = [(1, i) for i in range(len(queries))]
    initial_lookup = [(1, i) for i in range(1000)]
    set_size = 100

    update_plan = []
    for i in range(0, len(data_2), set_size):
        for j in range(set_size):
            if i+j < len(data_2):
                update_plan.append((2, len(data_1) + i+j))
        for j in range(set_size):
            if i+j < len(data_2):
                update_plan.append((0, len(data_1) + i+j))
                if len(data_1)+i+j < len(queries):
                    update_plan.append((1, len(data_1)+i+j))
        update_plan += initial_lookup
        
    # indexing_plan = [(0, i) for i in range(len(data))]

    plans = [
        ("Indexing", data, queries, indexing_plan, None),
        ("Initial Search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
        ("Update", data, queries, update_plan, None),
        ("Re-search", data, queries, initial_lookup, None),
    ]
    run_dynamic_test(plans, gt_neighbors, gt_dists, max_vectors=len(data))
