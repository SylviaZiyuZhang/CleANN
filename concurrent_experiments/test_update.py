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

# data_1 = data[:500000]
# data_2 = data[500000:1000000]

data_1 = data[:50000]
data_2 = data[50000:100000]
data = data[:100000]

indexing_plan = [(0, i) for i in range(len(data_1))]
initial_lookup = [(1, i) for i in range(len(data_1))]
set_size = 100

update_plan = []
for i in range(0, len(data_2), set_size):
    for j in range(set_size):
        if i+j < len(data_2):
            update_plan.append((0, len(data_2) + i+j))
            update_plan.append((1, len(data_2)+i+j))
    
    for j in range(set_size):
        if i+j < len(data_2):
            update_plan.append((2, len(data_2) + i+j))
    

print(len(update_plan))
print(update_plan[:30])
# indexing_plan = [(0, i) for i in range(len(data))]

plans = [
    ("Indexing", data_1, queries, indexing_plan, None),
    ("Initial Search", data_1, queries, initial_lookup, None),
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality
    ("Update", data, queries, update_plan, None),
    ("Re-search", data_1, queries, initial_lookup, None), # TODO: compare search quality

]

run_dynamic_test(plans, max_vectors=len(data))
