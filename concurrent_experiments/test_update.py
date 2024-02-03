import heapq
import numpy as np
import h5py # FileNotFoundError
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]

data_path = f"data/{dataset_name}"
data, queries, gt_neighbors, gt_dists = parse_ann_benchmarks_hdf5(data_path)

print(data[0])
print(queries[0])
print(gt_neighbors[0])
print(gt_dists[0])

def brute_force_knn(data, start, end, query, k=10):
    """
    data: dataset
    start: the starting index to compute ground truth neighbors to
    end: the ending index (non-inclusive) to compute ground truth neighbors to
    query: the query vector
    k: the number of ground truth neighbors to compute
    REQUIRES:
        - data.shape[1] == len(query) (matching dimensions)
        - 0 <= start <= end <= len(data)
    """
    neighbors = []
    cur_k = 0
    for i in range(start, end):
        heapq.heappush(neighbors, (-np.sqrt(np.dot(query, data[i])), i))
        if cur_k >= k:
            heapq.heappop(neighbors)
    neighbor_ids = [i for (_, i) in neighbors]
    dists = [-d for (d, _) in neighbors]
    return neighbor_ids, dists

def load_or_create_test_data(path, size, dimension, n_queries, gt_k=100):
    """
    Requires:
        path: string
        size: number of datapoints in the dataset, positive integer
        dimension: number of dimensions of each data point in the dataset,
            positive integer >= 32, power of 2
        n_queries: number of queries in the dataset, positive integer
        gt_kk: number of ground truth neighbors to compute up to, integer
    Ensures:
        data: size x dimension real number np array
        queries: n_queries x dimension real number np array
        gt_neighbors: n_queries x gt_k np array of dataset ID (pos ints)
        gt_dists: n_queries x gt_k np array of positive real numbers
    Side effect:
        if the data is not loaded from the path supplied successfully,
        attempt to create the requested dataset and store it at path
    """
    try:
        return parse_ann_benchmarks_hdf5(path)
    except h5py.FileNotFoundError:
        pass
    
    data = np.random.normal(size=(size, dimension), scale=1000.0)
    queries = np.random.normal(size=(n_queries, dimension), scale=1000.0)
    gt_neighbors = []
    gt_dists = []
    for i in range(n_queries):
        nn, dd = brute_force_knn(data, 0, len(data), queries[i], gt_k)
        gt_neighbors.append(nn)
        gt_dists.append(dd)
    
    h5f = h5py.file('manual_data.h5', 'w')
    h5f.create_dataset("train", data=data)
    h5f.create_dataset("test", data=queries)
    h5f.create_dataset("neighbors", data=gt_neighbors)
    h5f.create_dataset("distances", data=gt_dists)
    h5f.close()
    return data, queries, gt_neighbors, gt_dists

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
