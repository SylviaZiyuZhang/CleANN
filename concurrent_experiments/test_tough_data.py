import h5py
import numpy as np
import yaml
import diskannpy
import time
from utils import parse_ann_benchmarks_hdf5

dynamic_test = diskannpy._diskannpy.run_dynamic_test

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]
dimension = int(dataset_name.split("-")[1])
query_k = 10

alpha = 1.2
build_complexity = 64
graph_degree = 64
query_complexity = 64

data_path = f"data/{dataset_name}"
data, queries, gt_neighbors = parse_ann_benchmarks_hdf5(data_path)

chunk_size = 8
for i in range(0, len(data), chunk_size):
    data[i: i + chunk_size] = data[i] + 0.01 * np.random.normal(size=(chunk_size, len(data[0])))
print(data.shape, queries.shape)

times = []
for num_threads in [1, 2, 4, 8]:
    start = time.time()

    dynamic_index = diskannpy.DynamicMemoryIndex(
        distance_metric="l2",
        vector_dtype=np.float32,
        dimensions=dimension,
        max_vectors=len(data),
        complexity=build_complexity,
        graph_degree=graph_degree,
    )

    results = dynamic_test(
        dynamic_index._index,
        data,
        queries,
        [(0, i) for i in range(len(data))],
        query_k=query_k,
        query_complexity=query_complexity,
        num_threads=num_threads,
    )

    times.append(time.time() - start)

    print(num_threads, times[-1], times[0] / times[-1])
