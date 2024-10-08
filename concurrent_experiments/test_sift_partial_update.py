import heapq
import numpy as np
import h5py # FileNotFoundError
import struct
import numpy as np
import os
import math
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test
import concurrent.futures
from pathlib import Path

#path = Path('~/data/tmp/').expanduser()
#path.mkdir(parents=True, exist_ok=True)

def brute_force_knn(data, start, end, query, k=10, return_set=False):
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
        heapq.heappush(neighbors, (-np.linalg.norm(query-data[i]), i))
        cur_k += 1
        if cur_k > k:
            heapq.heappop(neighbors)

    if return_set:
        return set(neighbors)
    neighbor_ids = [i for (_, i) in neighbors]
    dists = [-d for (d, _) in neighbors]
    return neighbor_ids, dists

def get_ground_truth_batch(data, start, end, queries, k=10, dataset_name="sift", size=10000):
    all_neighbors = [[] for _ in queries]
    all_neighbor_ids = [[] for _ in queries]
    all_dists = [[] for _ in queries]
    for j, q in enumerate(queries):
        #neighbors = []
        cur_k = 0
        for i in range(start, end):
            heapq.heappush(all_neighbors[j], (-np.linalg.norm(q-data[i]), i))
            cur_k += 1
            if cur_k > k:
                heapq.heappop(all_neighbors[j])
    
    for j, neighbors in enumerate(all_neighbors):
        all_neighbor_ids[j] = [i for (_, i) in neighbors]
        all_dists[j] = [-d for (d, _) in neighbors]
    return all_neighbor_ids, all_dists

def get_or_create_rolling_update_ground_truth(path, data, data_to_update, queries, save=False, batch_num=100, dataset_name="sift", k=10):
    if path is not None:
        return np.load(path/'ids.npy'), np.load(path/'dists.npy')
    path = Path('~/data/ann_rolling_update_gt/'+dataset_name+"_"+str(len(data))+"_"+str(batch_num)).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    batch_size = len(data) // batch_num
    bigger_k = 5 * k
    #np.save(path/'x', x)
    #np.save(path/'y', y)
    assert len(data) == len(data_to_update)
    all_results_ids = [] # first dimension is batch, first item is initial ground truth (without any updates. Second dimension is queries
    all_results_dists = []
    all_neighbors = [set() for _ in queries]
    # keep the same section of data in cache for locality for when no update has happened
    # first compute when no update has happened
    # Maybe implement KD heap here lmao
    for i, v in enumerate(data):
        for j, q in enumerate(queries):
            all_neighbors[j].add((-np.linalg.norm(q-data[i]), i))
            if len(all_neighbors[j]) > bigger_k:
                min_elem = (math.inf, -1)
                for kk in all_neighbors[j]:
                    if kk[0] < min_elem[0]:
                        min_elem = kk
                all_neighbors[j].remove(min_elem)
    
    all_neighbor_ids = [[] for _ in queries]
    all_dists = [[] for _ in queries]
    for j, neighbors in enumerate(all_neighbors):
        knn = heapq.nlargest(k, list(neighbors))
        all_neighbor_ids[j] = [i for (_, i) in knn]
        all_dists[j] = [-d for (d, _) in knn]
    all_results_ids.append(all_neighbor_ids)
    all_results_dists.append(all_dists)

    # Then, iteratively filter out ids that are too small to mimic the deletion
    for b in range(0, len(data), batch_size):
        for j, q in enumerate(queries):
            all_neighbors[j] = set(filter(lambda kk: kk[1] >= b+batch_size, all_neighbors[j]))
            if len(all_neighbors[j]) < k:
                all_neighbors[j] = brute_force_knn(np.concatenate((data, data_to_update)), b, b+len(data), q, k=bigger_k, return_set=True)
                continue
            for i in range(b, b + batch_size):
                # insert the new things
                neg_new_dist = -np.linalg.norm(q-data_to_update[i])
                min_elem = (math.inf, -1)
                for kk in all_neighbors[j]:
                    if kk[0] < min_elem[0]:
                        min_elem = kk
                if min_elem[0] < neg_new_dist:
                    all_neighbors[j].add((neg_new_dist, i+len(data)))
                    if len(all_neighbors[j]) >= bigger_k:
                        all_neighbors[j].remove(min_elem)

        all_neighbor_ids = [[] for _ in queries]
        all_dists = [[] for _ in queries]
        for j, neighbors in enumerate(all_neighbors):
            knn = heapq.nlargest(k, list(neighbors))
            all_neighbor_ids[j] = [i for (_, i) in knn]
            all_dists[j] = [-d for (d, _) in knn]
        all_results_ids.append(all_neighbor_ids)
        all_results_dists.append(all_dists)
    
    if save:
        raise NotImplementedError
    return all_results_ids, all_results_dists
            
    

def get_ground_truth_batch_parallel(data, start, end, queries, k=10, dataset_name="sift", size=10000):
    all_neighbors = [[] for _ in queries]
    all_neighbor_ids = [[] for _ in queries]
    all_dists = [[] for _ in queries]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(brute_force_knn, data, start, end, q, k) for q in queries]
        for j, future in enumerate(concurrent.futures.as_completed(futures)):
            all_neighbor_ids[j], all_dists[j] = future.result()

    return all_neighbor_ids, all_dists

def load_or_create_test_data(path, size=100, dimension=10, n_queries=10, gt_k=100, creat=False):
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
    except:
        if not create:
            return [], [], [], []
    
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
    if dist == "normal":
        for i in range(0, len(data), chunk_size):
            # TODO: fix the variance of these normal distributions
            data[i : i + chunk_size] = data[i] + alpha * np.random.normal(
                size=(chunk_size, len(data[0]))
            )

def close_insertion_experiment(data, queries):
    data = data[:1000]
    queries = queries[:100]
    chunk_size = 8
    l = len(data)
    alpha = 1
    batches_to_insert = []
    for q in queries:
        b = q + alpha * np.random.normal(size=(chunk_size, len(q)))
        batches_to_insert.append(b)

    indexing_plan = [(0, i) for i in range(len(data))]
    query_plan = [(1, i) for i in range(len(queries))]
    for b in batches_to_insert:
        data = np.concatenate((data, b))
    print(data.shape)
    gt_before = []
    dists_before = []
    gt_after = []
    dists_after = []
    # data, start, end, query
    for i, q in enumerate(queries):
        gt, dists = brute_force_knn(data, 0, l + i * chunk_size, q)
        gt_before.append(gt)
        dists_before.append(dists)
        gt, dists = brute_force_knn(data, 0, l + (i+1)*chunk_size, q)
        gt_after.append(gt)
        dists_after.append(dists)
    
    update_plan = []
    update_gt_before = []
    update_gt_after = []
    for i, q in enumerate(queries):
        for j in range(chunk_size//2):
            update_plan.append((0, l + i*chunk_size + j))
            update_gt_before.append([])
            update_gt_after.append([])
        update_plan.append((1, i))
        update_gt_before.append(gt_before[i])
        update_gt_after.append(gt_after[i])
        for j in range(chunk_size//2, chunk_size):
            update_plan.append((0, l + i*chunk_size + j))
            update_gt_before.append([])
            update_gt_after.append([])
    
    plans = [
        ("Indexing", data, queries, indexing_plan, None),
        ("Just Queries Before", data, queries, query_plan, gt_before),
        ("Concur Before", data, queries, update_plan, update_gt_before),
        ("Just Queries After", data, queries, query_plan, gt_after),
    ]
    print("Calling run_dynamic_test")
    # print(update_plan[10:])
    run_dynamic_test(plans, update_gt_before, dists_before, max_vectors=len(data))

    plans = [
        ("Indexing", data, queries, indexing_plan, None),
        ("Just Queries Before", data, queries, query_plan, gt_before),
        ("Concur After", data, queries, update_plan, update_gt_after),
        ("Just Queries After", data, queries, query_plan, gt_before),
    ]
    run_dynamic_test(plans, update_gt_after, dists_after, max_vectors=len(data))

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
    indexing_plan = [(0, i) for i in range(len(data_1))]
    # initial_lookup = [(1, i) for i in range(len(queries))]
    initial_lookup = [(1, i) for i in range(100)]
    initial_lookup_gt = gt_neighbors[:100]
    set_size = 10

    update_plan = []
    update_plan_gt = []
    for i in range(0, len(data_2), set_size):
        
        for j in range(set_size):
            if i+j < len(data_2):
                update_plan.append((2, len(data_1) + i+j))
                update_plan_gt.append([])
        
        for j in range(set_size):
            if i+j < len(data_2):
                update_plan.append((0, len(data_1) + i+j))
                update_plan_gt.append([])
                if len(data_1)+i+j < len(queries):
                    update_plan.append((1, len(data_1)+i+j))
                    update_plan_gt.append(gt_neighbors[len(data_1)+i+j])
        for e in initial_lookup:
            update_plan.append(e)
        for e in initial_lookup_gt:
            update_plan_gt.append(e)
    

    # indexing_plan = [(0, i) for i in range(len(data))]

    plans = [
        ("Indexing", data, queries, indexing_plan, None),
        ("Initial Search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
        ("Update", data, queries, update_plan, update_plan_gt),
        ("Re-search", data, queries, initial_lookup, initial_lookup_gt),
    ]
    run_dynamic_test(plans, gt_neighbors, gt_dists, max_vectors=len(data))

def small_batch_gradual_update_experiment(data, queries):
    size = 500000
    data = data[:2 * size]
    # data_to_update = data[size:2 * size] 
    update_batch_size = 5000
    n_update_batch = 100

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]
    
    
    plans = [("Indexing", data, queries, indexing_plan, None)]
    all_gt_neighbors, all_gt_dists = get_or_create_rolling_update_ground_truth(
        path=None,
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=False
    )
    initial_lookup_gt_neighbors = all_gt_neighbors[0]
    initial_lookup_gt_dists = all_gt_dists[0]
    for i in range(0, size, update_batch_size):
        update_plan = []
        for j in range(update_batch_size):
            delete_id = i + j
            insert_id = delete_id + size
            update_plan.append((0, insert_id))
            update_plan.append((2, delete_id))
        plans.append(("Update", data, queries, update_plan, None))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors))

    run_dynamic_test(plans, gt_neighbors, gt_dists, max_vectors=len(data))

data, queries, _, _ = load_or_create_test_data(path="../data/sift-128-euclidean.hdf5")
small_batch_gradual_update_experiment(data, queries[:1000])
