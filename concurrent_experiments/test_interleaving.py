import h5py
import numpy as np
import yaml
import diskannpy
import time
import os
from utils import test_static_index, num_recalled

dynamic_test = diskannpy._diskannpy.run_dynamic_test

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]
dimension = int(dataset_name.split("-")[1])
k = 10

alpha = 1.2
build_complexity = 64
graph_degree = 64
query_complexity = 64


def parse_ann_benchmarks_hdf5(data_path):
    with h5py.File(data_path, "r") as file:
        gt_neighbors = np.array(file["neighbors"])
        queries = np.array(file["test"])
        data = np.array(file["train"])

        return data, queries, gt_neighbors


data_path = f"data/{dataset_name}"
data, queries, gt_neighbors = parse_ann_benchmarks_hdf5(data_path)

# test_static_index(
#     data=data,
#     queries=queries,
#     gt_neighbors=gt_neighbors,
#     build_complexity=build_complexity,
#     graph_degree=graph_degree,
#     alpha=alpha,
#     query_complexity=query_complexity,
#     query_k=k,
#     dataset_short_name=dataset_short_name,
# )

data = data[:10000]
num_adversarial_per_data = 16
adversarial_queries = []
for i in range(8):
    for d in data:
        adversarial_queries.append(d + 1 * np.random.normal(size=len(d)))
adversarial_queries = np.array(adversarial_queries)

for strategy in ["Easy", "Hard"]:

    times = []
    for num_threads in [1, 2, 4, 8]:
        start = time.time()

        dynamic_index = diskannpy.DynamicMemoryIndex(
            distance_metric="l2",
            vector_dtype=np.float32,
            dimensions=dimension,
            max_vectors=len(data),
            complexity=64,
            graph_degree=64,
        )

        print(data.shape, adversarial_queries.shape)

        if strategy == "Easy":
            updates = [(0, i) for i in range(len(data))] + [(1, i) for i in range(len(adversarial_queries))]
        else:
            updates = [(i % (num_adversarial_per_data + 1) > 0, i // (num_adversarial_per_data + 1)) for i in range((num_adversarial_per_data + 1) * len(data))]

        results = dynamic_test(
            dynamic_index._index,
            data,
            adversarial_queries,
            updates,
            query_k=10,
            query_complexity=64,
            num_threads=num_threads,
        )

        end = time.time()
        times.append(end - start)


        print(strategy, num_threads, times[-1], times[0] / times[-1])
    # total_recalled = 0
    # for result, query_gt in zip(results[0], gt_neighbors):
    #     total_recalled += num_recalled(result, query_gt, k)

    # recall = total_recalled / len(queries) / k
    # print(num_threads, recall, end - start)

