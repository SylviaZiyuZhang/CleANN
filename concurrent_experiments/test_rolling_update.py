import heapq
import numpy as np
import h5py # FileNotFoundError
import struct
import numpy as np
import os
import math
import random
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test
import concurrent.futures
from pathlib import Path
from multiprocessing import cpu_count, Pool, RawArray

ROLLING_UPDATE_GET_STATIC_BASELINE = True
BATCH_INSERT_GET_STATIC_BASELINE = True

def get_cosine_dist(A, B):
    return - np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))

def get_l2_dist(A, B):
    return np.linalg.norm(A-B)

def get_mips_dist(A, B):
    # Maximum inner product search
    return - np.dot(A, B)

def calculate_medoid(data, metric="l2"):
    # This selects the vector from the dataset that is the medoid
    # (instead of calculating the medoid in the spatial sense)
    dist_func = get_l2_dist
    if metric == "cosine":
        dist_func = get_cosine_dist
    if metric == "mips":
        dist_func = get_mips_dist
    distance_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i, len(data)):
            dist = dist_func(data[j], data[i])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    medoid_id = np.argmin(distance_matrix.sum(axis=0))
    return data[medoid_id]

def brute_force_knn(data, start, end, query, k=10, return_set=False, metric="l2"):
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
    dist_func = get_l2_dist
    if metric == "cosine":
        dist_func = get_cosine_dist
    if metric == "mips":
        dist_func = get_mips_dist

    for i in range(start, end):
        heapq.heappush(neighbors, (-dist_func(query, data[i]), i))
        cur_k += 1
        if cur_k > k:
            heapq.heappop(neighbors)

    if return_set:
        return set(neighbors)
    neighbor_ids = [i for (_, i) in neighbors]
    dists = [-d for (d, _) in neighbors]
    return neighbor_ids, dists

def brute_force_knn_two_parts(data1, data2, start, end, query, k=10, return_set=False, metric="l2"):
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
    dist_func = get_l2_dist
    if metric == "cosine":
        dist_func = get_cosine_dist
    if metric == "mips":
        dist_func = get_mips_dist

    end1 = len(data1) if end > len(data1) else end
    for i in range(start, end1):
        heapq.heappush(neighbors, (-dist_func(query, data1[i]), i))
        cur_k += 1
        if cur_k > k:
            heapq.heappop(neighbors)
    
    start2 = 0 if start <= len(data1) else start - len(data1)
    for i in range(start2, end-len(data1)):
        heapq.heappush(neighbors, (-dist_func(query, data2[i]), i+len(data1)))
        cur_k += 1
        if cur_k > k:
            heapq.heappop(neighbors)

    if return_set:
        return set(neighbors)
    neighbor_ids = [i for (_, i) in neighbors]
    dists = [-d for (d, _) in neighbors]
    return neighbor_ids, dists

query_worker_var_dict = {}

def init_sliding_window_ground_truth_single_query_worker(data_raw, data_to_update_raw, data_shape, data_to_update_shape):
    query_worker_var_dict['data_raw'] = data_raw
    query_worker_var_dict['data_to_update_raw'] = data_to_update_raw
    query_worker_var_dict['data_shape'] = data_shape
    query_worker_var_dict['data_to_update_shape'] = data_to_update_shape

def get_sliding_window_ground_truth_single_query(args):
    query_index, q, batch_size, k, bigger_k, dist_func, rolling = args
    data = np.frombuffer(query_worker_var_dict['data_raw']).reshape(query_worker_var_dict['data_shape'])
    data_to_update = np.frombuffer(query_worker_var_dict['data_to_update_raw']).reshape(query_worker_var_dict['data_to_update_shape'])
    neighbors = set()
    result_ids = []
    result_dists = []

    for i, v in enumerate(data):
        dist = -dist_func(q, data[i])
        neighbors.add((dist, i))
        if len(neighbors) > bigger_k:
            min_elem = (math.inf, -1)
            for kk in neighbors:
                if kk[0] < min_elem[0]:
                    min_elem = kk
            assert(min_elem[1] != -1)
            neighbors.remove(min_elem)
    knn = heapq.nlargest(k, list(neighbors))
    result_ids.append([i for (_, i) in knn])
    result_dists.append([-d for (d, _) in knn])

    # Then, iteratively filter out ids that are too small to mimic the deletion if rolling=True
    # Groundtruth for insert-only workload if rolling=False
    for b in range(0, len(data_to_update), batch_size):
        if rolling:
            neighbors = set(filter(lambda kk: kk[1] >= b + batch_size, neighbors))
        if len(neighbors) < k:
            neighbors = brute_force_knn_two_parts(data, data_to_update, b, b+len(data), q, k=bigger_k, return_set=True)
            knn = heapq.nlargest(k, list(neighbors))
            result_ids.append([i for (_, i) in knn])
            result_dists.append([-d for (d, _) in knn])
            continue
        for i in range(b, b + batch_size):
            # insert the new things
            neg_new_dist = -dist_func(q, data_to_update[i])
            min_elem = (math.inf, -1)
            for kk in neighbors:
                if kk[0] < min_elem[0]:
                    min_elem = kk
            if min_elem[0] < neg_new_dist:
                neighbors.add((neg_new_dist, i+len(data)))
                if len(neighbors) >= bigger_k:
                    neighbors.remove(min_elem)
        knn = heapq.nlargest(k, list(neighbors))
        result_ids.append([i for (_, i) in knn])
        result_dists.append([-d for (d, _) in knn])

    return query_index, result_ids, result_dists

def get_or_create_ground_truth_batch(path, data, start, end, queries, save=False, k=10, dataset_name="sift", size=10000, metric="l2", shuffled_data=False, random_queries=False):
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    if path is None:
        path = Path('/storage/sylziyuz/ann_static_gt/'+dataset_name+"_"+metric+"_"+str(len(data))+suffix).expanduser()
    try:
        return np.load(path/'ids.npy'), np.load(path/'dists.npy')
    except FileNotFoundError:
        if save:
            path.mkdir(parents=True, exist_ok=True)
        dist_func = get_l2_dist
        if metric == "cosine":
            dist_func = get_cosine_dist
        if metric == "mips":
            dist_func = get_mips_dist
        all_neighbors = [[] for _ in queries]
        all_neighbor_ids = [[] for _ in queries]
        all_dists = [[] for _ in queries]
        for j, q in enumerate(queries):
            cur_k = 0
            for i in range(start, end):
                heapq.heappush(all_neighbors[j], (-dist_func(q, data[i]), i))
                cur_k += 1
                if cur_k > k:
                    heapq.heappop(all_neighbors[j])
        
        for j, neighbors in enumerate(all_neighbors):
            all_neighbor_ids[j] = [i for (_, i) in neighbors]
            all_dists[j] = [-d for (d, _) in neighbors]
        
        if save:
            np.save(path/'ids.npy', all_neighbor_ids)
            np.save(path/'dists.npy', all_dists)

        return all_neighbor_ids, all_dists



def get_or_create_rolling_update_ground_truth(path, data, data_to_update, queries, save=False, batch_num=100, dataset_name="sift", k=10, metric="l2", shuffled_data=False, random_queries=False):
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    if path is None:
        path = Path('/storage/sylziyuz/ann_rolling_update_gt_k50/'+dataset_name+"_"+metric+"_"+str(len(data))+"_"+str(batch_num)+suffix).expanduser()
    file_suffix = dataset_name+"_"+str(len(data))+"_rolling_update_gt.hdf5"
    try:
        with h5py.File(path/file_suffix, "r") as file:
            assert(file.attrs['batch_num'] == batch_num)
            assert(file.attrs['metric'] == metric)
            ids = np.array(file["neighbors"])
            dists = np.array(file["distances"])
        return ids, dists
    except FileNotFoundError:
        if save:
            path.mkdir(parents=True, exist_ok=False)
        dist_func = get_l2_dist
        if metric == "cosine":
            dist_func = get_cosine_dist
        if metric == "mips":
            dist_func = get_mips_dist
        batch_size = len(data) // batch_num
        bigger_k = 5 * k

        all_results_ids_by_query = [[] for _ in queries] # first dimension is batch, first item is initial ground truth (without any updates. Second dimension is queries
        all_results_dists_by_query = [[] for _ in queries]
        data_shape = data.shape
        data_raw = RawArray('d', data.shape[0] * data.shape[1])
        data_np = np.frombuffer(data_raw, dtype=float).reshape(data_shape)
        np.copyto(data_np, data)
        data_to_update_shape = data_to_update.shape
        data_to_update_raw = RawArray('d', data_to_update.shape[0] * data_to_update.shape[1])
        data_to_update_np = np.frombuffer(data_to_update_raw).reshape(data_to_update_shape)
        np.copyto(data_to_update_np, data_to_update)
        proc_count = max(1, cpu_count() - 4)

        with Pool(processes=proc_count, initializer=init_sliding_window_ground_truth_single_query_worker, initargs=(data_raw, data_to_update_raw, data_shape, data_to_update_shape)) as pool:
            args = [
                (j, q, batch_size, k, bigger_k, dist_func, True)
                for j, q in enumerate(queries)
            ]
            results = pool.map(get_sliding_window_ground_truth_single_query, args)
            for j, result_ids, result_dists in results:
                all_results_ids_by_query[j] = result_ids
                all_results_dists_by_query[j] = result_dists 
        
        all_results_ids_np = np.transpose(np.array(all_results_ids_by_query), (1, 0, 2))
        all_results_dists_np = np.transpose(np.array(all_results_dists_by_query), (1, 0, 2))
        assert(all_results_dists_np.shape[0] == batch_num + 1)
        assert(all_results_dists_np.shape[1] == len(queries))
        assert(all_results_dists_np.shape[2] == k)
        assert(all_results_ids_np.shape[0] == batch_num + 1)
        assert(all_results_ids_np.shape[1] == len(queries))
        assert(all_results_ids_np.shape[2] == k)
        if save:
            with h5py.File(path/file_suffix, "w") as file:
                file.attrs['batch_num'] = batch_num
                file.attrs['metric'] = metric
                file.create_dataset('neighbors', data=all_results_ids_np)
                file.create_dataset('distances', data=all_results_dists_np)
            #np.save(path/'ids.npy', all_results_ids_np)
            #np.save(path/'dists.npy', all_results_dists_np)
        return all_results_ids_np, all_results_dists_np


def get_or_create_insert_only_ground_truth(path, data, data_to_update, queries, save=False, batch_num=100, dataset_name="sift", k=10, metric="l2", shuffled_data=False, random_queries=False):
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    if path is None:
        path = Path('/storage/sylziyuz/ann_batch_insert_gt/'+dataset_name+"_"+metric+"_"+str(len(data))+"_"+str(batch_num)+suffix).expanduser()
    file_suffix = dataset_name+"_"+str(len(data))+"_batch_insert_gt.hdf5"
    try:
        with h5py.File(path/file_suffix, "r") as file:
            assert(file.attrs['batch_num'] == batch_num)
            assert(file.attrs['metric'] == metric)
            ids = np.array(file["neighbors"])
            dists = np.array(file["distances"])
        return ids, dists
    except FileNotFoundError:
        if save:
            path.mkdir(parents=True, exist_ok=False)
        dist_func = get_l2_dist
        if metric == "cosine":
            dist_func = get_cosine_dist
        if metric == "mips":
            dist_func = get_mips_dist
        batch_size = len(data) // batch_num
        bigger_k = 5 * k

        all_results_ids_by_query = [[] for _ in queries] # first dimension is batch, first item is initial ground truth (without any updates. Second dimension is queries
        all_results_dists_by_query = [[] for _ in queries]
        data_shape = data.shape
        print(data.shape[0], data.shape[1])
        data_raw = RawArray('d', data.shape[0] * data.shape[1])
        data_np = np.frombuffer(data_raw, dtype=float).reshape(data_shape)
        np.copyto(data_np, data)
        data_to_update_shape = data_to_update.shape
        data_to_update_raw = RawArray('d', data_to_update.shape[0] * data_to_update.shape[1])
        data_to_update_np = np.frombuffer(data_to_update_raw).reshape(data_to_update_shape)
        np.copyto(data_to_update_np, data_to_update)
        proc_count = max(1, cpu_count() - 4)

        with Pool(processes=proc_count, initializer=init_sliding_window_ground_truth_single_query_worker, initargs=(data_raw, data_to_update_raw, data_shape, data_to_update_shape)) as pool:
            args = [
                (j, q, batch_size, k, bigger_k, dist_func, False) # rolling=False, not filtering out old neighbors.
                for j, q in enumerate(queries)
            ]
            results = pool.map(get_sliding_window_ground_truth_single_query, args)
            for j, result_ids, result_dists in results:
                all_results_ids_by_query[j] = result_ids
                all_results_dists_by_query[j] = result_dists 
        
        all_results_ids_np = np.transpose(np.array(all_results_ids_by_query), (1, 0, 2))
        all_results_dists_np = np.transpose(np.array(all_results_dists_by_query), (1, 0, 2))
        print(all_results_dists_np.shape, all_results_ids_np.shape)
        assert(all_results_dists_np.shape[0] == batch_num + 1)
        assert(all_results_dists_np.shape[1] == len(queries))
        assert(all_results_dists_np.shape[2] == k)
        assert(all_results_ids_np.shape[0] == batch_num + 1)
        assert(all_results_ids_np.shape[1] == len(queries))
        assert(all_results_ids_np.shape[2] == k)
        if save:
            with h5py.File(path/file_suffix, "w") as file:
                file.attrs['batch_num'] = batch_num
                file.attrs['metric'] = metric
                file.create_dataset('neighbors', data=all_results_ids_np)
                file.create_dataset('distances', data=all_results_dists_np)
        return all_results_ids_np, all_results_dists_np

    

def get_ground_truth_batch_parallel(data, start, end, queries, k=10, dataset_name="sift", size=10000):
    all_neighbors = [[] for _ in queries]
    all_neighbor_ids = [[] for _ in queries]
    all_dists = [[] for _ in queries]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(brute_force_knn, data, start, end, q, k) for q in queries]
        for j, future in enumerate(concurrent.futures.as_completed(futures)):
            all_neighbor_ids[j], all_dists[j] = future.result()

    return np.array(all_neighbor_ids), np.array(all_dists)

def load_or_create_test_data(path, size=100, dimension=10, n_queries=10, gt_k=100, create=False):
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
        ("Indexing", data, queries, indexing_plan, None, False),
        ("Just Queries Before", data, queries, query_plan, gt_before, False),
        ("Concur Before", data, queries, update_plan, update_gt_before, False),
        ("Just Queries After", data, queries, query_plan, gt_after, False),
    ]
    run_dynamic_test(plans, update_gt_before, dists_before, max_vectors=len(data))

    plans = [
        ("Indexing", data, queries, indexing_plan, None, False),
        ("Just Queries Before", data, queries, query_plan, gt_before, False),
        ("Concur After", data, queries, update_plan, update_gt_after, False),
        ("Just Queries After", data, queries, query_plan, gt_before, False),
    ]
    run_dynamic_test(plans, update_gt_after, dists_after, max_vectors=len(data))


def get_static_recall(data, queries, start, end, gt_neighbors, gt_dists):

    indexing_plan = [(0, i) for i in range(start, end)]
    lookup = [(1, i) for i in range(len(queries))]
    
    plans = [("Indexing", data, queries, indexing_plan, None, False)]
    plans.append(("Search", data, queries, lookup, gt_neighbors, False))

    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name="redcaps_no_shuffle_10000_fresh_update_static_recall_"+str(start)
    )

def static_recall_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, build_complexity=64, query_complexity=64, graph_degree=64,
):
    data = data[:2 * size]
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans = []
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    lookup_gt_neighbors, lookup_gt_dists = get_or_create_ground_truth_batch(
        path=Path(gt_data_prefix+'/ann_static_gt/'+dataset_name+"_"+metric+"_"+str(len(data))+suffix).expanduser(),
        data=data,
        start=0,
        end=2*size,
        queries=queries,
        save=True,
        k=10,
        dataset_name=dataset_name,
        size=2*size,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )

    plans.append(("Search", data, queries, initial_lookup, lookup_gt_neighbors, False))

    run_dynamic_test(
        plans,
        lookup_gt_neighbors,
        lookup_gt_dists,
        max_vectors=len(data),
        experiment_name="{}_{}_{}_{}_static".format(dataset_name, size, setting_name, metric),
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data,
        batch_build_tags=[i for i in range(1, len(data)+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )

def small_batch_gradual_update_immediate_delete_experiment(data, queries, dataset_name, gt_data_prefix, setting_name="setting_name", size=5000, metric="l2"):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * 2)
    data = data[:2 * size]
    n_update_batch = 100
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans=[]
    all_gt_neighbors, all_gt_dists = get_or_create_rolling_update_ground_truth(
        path=Path(gt_data_prefix +'/ann_rolling_update_gt/'+dataset_name+"_"+metric+"_"+str(size)+"_100").expanduser(),
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=False,
        dataset_name=dataset_name,
        metric=metric,
    )
    initial_lookup_gt_neighbors = all_gt_neighbors[0]
    initial_lookup_gt_dists = all_gt_dists[0]
    for i in range(0, size, update_batch_size):
        for j in range(update_batch_size):
            update_plan = []
            delete_id = i + j
            insert_id = delete_id + size
            update_plan.append((0, insert_id))
            update_plan.append((2, delete_id))
            consolidate = True if j > 0 else False
            plans.append(("Update", data, queries, update_plan, None, consolidate))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors, False))
    
    experiment_name = "{}_{}_{}_{}_rolling_update_immediate_delete".format(dataset_name, size, setting_name, metric)
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name=experiment_name,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)]
        )
    
    

    # ============================== get static recall ==============================

    if ROLLING_UPDATE_GET_STATIC_BASELINE:
        for i in range(0, size, update_batch_size):
            lookup = [(1, i) for i in range(len(queries))]
            experiment_name = "{}_{}_{}_{}_rolling_update_static_baseline_{}_{}".format(dataset_name, size, setting_name, metric, i, i+size)
            run_dynamic_test(
                [("Search", data, queries, lookup, gt_neighbors, False)],
                all_gt_neighbors[1 + i // update_batch_size],
                all_gt_dists[1 + i // update_batch_size],
                max_vectors=len(data),
                experiment_name=experiment_name,
                batch_build=True,
                batch_build_data=data[i:i+size],
                batch_build_tags=[i for i in range(i+1, i+size+1)]
            )


def small_batch_gradual_update_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, query_complexity=64, build_complexity=64, graph_degree=64
):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * 2)
    data = data[:2 * size]
    n_update_batch = 100
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans=[]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    all_gt_neighbors, all_gt_dists = get_or_create_rolling_update_ground_truth(
        path=Path(gt_data_prefix +'/ann_rolling_update_gt_k50/'+dataset_name+"_"+metric+"_"+str(size)+"_100"+suffix).expanduser(),
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=True,
        k=query_k,
        dataset_name=dataset_name,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
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
        plans.append(("Update", data, queries, update_plan, None, False))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]
        consolidate = False
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors, consolidate))
    
    experiment_name = "{}_{}_{}_{}_rolling_update".format(dataset_name, size, setting_name, metric)
    
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name=experiment_name,
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )
    
    

    # ============================== get static recall ==============================

    if ROLLING_UPDATE_GET_STATIC_BASELINE:
        for i in range(0, size + update_batch_size, update_batch_size):
            lookup = [(1, j) for j in range(len(queries))]
            experiment_name = "{}_{}_{}_{}_rolling_update_static_baseline_{}_{}".format(dataset_name, size, setting_name, metric, i, i+size)
            run_dynamic_test(
                [("Search", data, queries, lookup, all_gt_neighbors[i // update_batch_size], False)],
                all_gt_neighbors[i // update_batch_size],
                all_gt_dists[i // update_batch_size],
                max_vectors=len(data),
                experiment_name=experiment_name,
                distance_metric=metric,
                batch_build=True,
                batch_build_data=data[i:i+size],
                batch_build_tags=[j for j in range(i+1, i+size+1)],
                query_k=query_k,
                build_complexity=build_complexity,
                query_complexity=query_complexity,
                graph_degree=graph_degree,
            )

def small_batch_gradual_update_train_test_split_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, query_complexity=64, build_complexity=64, graph_degree=64
):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * 2)
    data = data[:2 * size]
    n_update_batch = 100
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans=[]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    all_gt_neighbors, all_gt_dists = get_or_create_rolling_update_ground_truth(
        path=Path(gt_data_prefix +'/ann_rolling_update_gt/'+dataset_name+"_"+metric+"_"+str(size)+"_100"+suffix).expanduser(),
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=True,
        k=query_k,
        dataset_name=dataset_name,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )
    initial_lookup_gt_neighbors = all_gt_neighbors[0]
    initial_lookup_gt_dists = all_gt_dists[0]
    apx_total_initial_gt = 0.0
    for ans in initial_lookup_gt_dists:
        apx_total_initial_gt += abs(ans[0])
    # apx_average_initial_gt helps perturb the query workload to generating training set
    apx_average_initial_gt = apx_total_initial_gt / len(initial_lookup_gt_dists)

    train_prop = 0.02
    plans = []

    for i in range(0, size, update_batch_size):
        update_plan = []
        for j in range(update_batch_size):
            delete_id = i + j
            insert_id = delete_id + size
            update_plan.append((0, insert_id))
            update_plan.append((2, delete_id))
        plans.append(("Update", data, queries, update_plan, None, False))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]

        n_train = max(int(queries.shape[0] * train_prop), 100)
        
        train_queries = queries[np.random.choice(queries.shape[0], int(n_train), replace=False)]
        if train_queries.dtype == np.dtype('int8'):
            for j in range(len(train_queries)):
                train_queries[j] += np.random.randint(-math.ceil(apx_average_initial_gt), math.ceil(apx_average_initial_gt), size=train_queries.shape[1], dtype=np.int8)
        else:
            for j in range(len(train_queries)):
                train_queries[j] += np.random.normal(train_queries[j], apx_average_initial_gt, size=train_queries.shape[1])
        
        plans.append(("Train"+str(i), data, train_queries, [(3, i) for i in range(len(train_queries))], gt_neighbors, False))
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors, False))
    
    experiment_name = "{}_{}_{}_{}_rolling_update_train_test_split".format(dataset_name, size, setting_name, metric)
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name=experiment_name,
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )

def small_batch_gradual_update_split_long_running_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, query_complexity=64, build_complexity=64, graph_degree=64, n_iter=3,
):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * n_iter)
    data = data[:n_iter * size]
    n_update_batch = 100 * n_iter
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans=[]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    first_iter_gt_neighbors, first_iter_gt_dists = get_or_create_rolling_update_ground_truth(
        path=Path(gt_data_prefix +'/ann_rolling_update_gt_k50/'+dataset_name+"_"+metric+"_"+str(size)+"_100"+suffix).expanduser(),
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=True,
        k=query_k,
        dataset_name=dataset_name,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )
    initial_lookup_gt_neighbors = first_iter_gt_neighbors[0]
    initial_lookup_gt_dists = first_iter_gt_dists[0]
    apx_total_initial_gt = 0.0
    for ans in initial_lookup_gt_dists:
        apx_total_initial_gt += abs(ans[0])
    # apx_average_initial_gt helps perturb the query workload to generating training set
    apx_average_initial_gt = apx_total_initial_gt / len(initial_lookup_gt_dists)

    later_iters_gt_neighbors, later_iters_gt_dists = get_or_create_rolling_update_ground_truth(
        path=Path(gt_data_prefix +'/ann_rolling_update_gt_k50_long_running/'+dataset_name+"_"+metric+"_"+str(size)+"_"+str(n_update_batch-100)+suffix).expanduser(),
        data=data[size:2 * size],
        data_to_update=data[2 * size:n_iter * size],
        queries=queries,
        save=True,
        k=query_k,
        dataset_name=dataset_name,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )
    later_iters_gt_neighbors += size
    all_gt_neighbors = np.concatenate((first_iter_gt_neighbors, later_iters_gt_neighbors[1:]))
    all_gt_dists = np.concatenate((first_iter_gt_dists, later_iters_gt_dists[1:]))
    assert(len(all_gt_neighbors) == (n_iter - 1) * 100 + 1)
    assert(len(all_gt_dists) == (n_iter - 1) * 100 + 1)

    for i in range(0, (n_iter - 1) * size, update_batch_size):
        update_plan = []
        for j in range(update_batch_size):
            delete_id = i + j
            insert_id = delete_id + size
            update_plan.append((0, insert_id))
            update_plan.append((2, delete_id))
        plans.append(("Update", data, queries, update_plan, None, False))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]
        # train_queries = queries[np.random.choice(queries.shape[0], min(queries.shape[0] // 10, 200), replace=False)]
        train_queries = queries[np.random.choice(queries.shape[0], queries.shape[0] // 2, replace=False)]
        if train_queries.dtype == np.dtype('int8'):
            for j in range(len(train_queries)):
                train_queries[j] += np.random.randint(-math.ceil(apx_average_initial_gt), math.ceil(apx_average_initial_gt), size=train_queries.shape[1], dtype=np.int8)
        else:
            for j in range(len(train_queries)):
                train_queries[j] += np.random.normal(train_queries[j], apx_average_initial_gt, size=train_queries.shape[1])
        plans.append(("Train"+str(i), data, train_queries, [(3, i) for i in range(len(train_queries))], gt_neighbors, False))
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors, True))
    
    experiment_name = "{}_{}_{}_{}_rolling_update_train_test_split_long".format(dataset_name, size, setting_name, metric)
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name=experiment_name,
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )

    if ROLLING_UPDATE_GET_STATIC_BASELINE:
        for i in range(size, (n_iter-1) * size + update_batch_size, update_batch_size):
            lookup = [(1, j) for j in range(len(queries))]
            experiment_name = "{}_{}_{}_{}_rolling_update_static_baseline_{}_{}".format(dataset_name, size, setting_name, metric, i, i+size)
            run_dynamic_test(
                [("Search", data, queries, lookup, all_gt_neighbors[i // update_batch_size], False)],
                all_gt_neighbors[i // update_batch_size],
                all_gt_dists[i // update_batch_size],
                max_vectors=len(data),
                experiment_name=experiment_name,
                distance_metric=metric,
                batch_build=True,
                batch_build_data=data[i:i+size],
                batch_build_tags=[j for j in range(i+1, i+size+1)],
                query_k=query_k,
                build_complexity=build_complexity,
                query_complexity=query_complexity,
                graph_degree=graph_degree,
            )

def small_batch_gradual_update_insert_only_experiment(data, queries, dataset_name, gt_data_prefix, setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, query_complexity=64, build_complexity=64, graph_degree=64):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * 2)
    data = data[:2 * size]
    n_update_batch = 100
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]
    
    print(len(data), size)


    plans = [("Indexing", data, queries, indexing_plan, None, False)]
    # plans=[]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    all_gt_neighbors, all_gt_dists = get_or_create_insert_only_ground_truth(
        path=Path(gt_data_prefix +'/ann_batch_insert_gt_k50/'+dataset_name+"_"+metric+"_"+str(size)+"_100"+suffix).expanduser(),
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=True,
        k=query_k,
        dataset_name=dataset_name,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )
    initial_lookup_gt_neighbors = all_gt_neighbors[0]
    initial_lookup_gt_dists = all_gt_dists[0]
    for i in range(0, size, update_batch_size):
        update_plan = []
        for j in range(update_batch_size):
            delete_id = i + j
            insert_id = delete_id + size
            update_plan.append((0, insert_id))
        plans.append(("Insert", data, queries, update_plan, None, False))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors, False))
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name = "{}_{}_{}_{}_batch_insert".format(dataset_name, size, setting_name, metric),
        distance_metric=metric,
        batch_build=False,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )


    # ============================== get static recall ==============================
    if BATCH_INSERT_GET_STATIC_BASELINE:
        for i in range(0, size + update_batch_size, update_batch_size):
            lookup = [(1, j) for j in range(len(queries))]
            experiment_name = "{}_{}_{}_{}_batch_insert_static_baseline_{}_{}".format(dataset_name, size, setting_name, metric, 0, i+size)
            run_dynamic_test(
                [("Search", data, queries, lookup, all_gt_neighbors[i // update_batch_size], False)],
                all_gt_neighbors[i // update_batch_size],
                all_gt_dists[i // update_batch_size],
                max_vectors=len(data),
                experiment_name=experiment_name,
                distance_metric=metric,
                batch_build=True,
                batch_build_data=data[:i+size],
                batch_build_tags=[j for j in range(1, i+size+1)],
                query_k=query_k,
                build_complexity=build_complexity,
                query_complexity=query_complexity,
                graph_degree=graph_degree,
            )
    

def random_point_recall_improvement_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, build_complexity=64, query_complexity=64, graph_degree=64,
):
    """
    This experiment repeatedly do the following to an index
    add random points
    delete the same random points
    consolidate
    measure recall
    """
    size *= 2
    # data_to_update = data[3*size: 4*size]
    data = data[:size]
    data_to_update = data

    update_batch_size = size
    n_update_batch = 50
    n_queries = len(queries)

    lookup = [(1, i) for i in range(len(queries))]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    gt_neighbors, gt_dists = get_or_create_ground_truth_batch(
        path=Path(gt_data_prefix+'/ann_static_gt/'+dataset_name+"_"+metric+"_"+str(len(data))+suffix).expanduser(),
        data=data,
        start=0,
        end=size,
        queries=queries,
        save=True,
        k=10,
        dataset_name=dataset_name,
        size=size,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )
    assert(len(gt_neighbors) == len(queries))
    assert(len(gt_dists) == len(queries))
    extra_data = np.zeros((n_update_batch * update_batch_size, len(data[0])))
    for i in range(0, n_update_batch * update_batch_size):
        base_idx = np.random.randint(size)
        extra_data[i] = data_to_update[base_idx] + np.random.normal(scale=np.linalg.norm(data_to_update[base_idx]), size=len(data[0]))
    print("The shape of data is ", data.shape)
    print("The shape of extra data is ", extra_data.shape)
    data = np.concatenate((data, extra_data))
    try:
        assert(data.shape[0] == size + n_update_batch * update_batch_size)
    except AssertionError:
        print("Shape mismatch between data and random data to update, the shape of dat is ", data.shape)
        return

    plans = [("Search"+str(i), data, queries, lookup, gt_neighbors, False)]

    for i in range(0, n_update_batch):
        update_plan = []
        for j in range(update_batch_size):
            id = size + i*update_batch_size + j
            update_plan.append((0, id))
        for j in range(update_batch_size):
            id = size + i*update_batch_size + j
            update_plan.append((2, id))
        plans.append(("Update", data, queries, update_plan, None, False))
        plans.append(("Search"+str(i), data, queries, lookup, gt_neighbors, True))
        plans.append(("Search"+str(i), data, queries, lookup, gt_neighbors, False))
        
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name="{}_{}_{}_{}_random_sweep_consolidate".format(dataset_name, size, setting_name, metric),
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )

def sorted_adversarial_data_recall_experiment(data, queries, reverse=False, batch_build=False, metric="l2"):
    medoid_vector = calculate_medoid(data, metric)
    dist_func = get_l2_dist
    if metric == "cosine":
        dist_func = get_cosine_dist
    if metric == "mips":
        dist_func = get_mips_dist
    medoid_distances = [dist_func(medoid_vector, v) for v in data]
    # sort the data from the furthest to the closest to the medoid
    sorted_data = [v for _, v in sorted(zip(medoid_distances, data), key=lambda pair: -pair[0], reverse=False)]
    indexing_plan = [(0, i) for i in range(len(data))]
    lookup = [(1, i) for i in range(len(queries))]
    plans = [] if batch_build else [("Indexing", np.array(sorted_data), queries, indexing_plan, None, False)]
    gt_neighbors, gt_dists = get_or_create_ground_truth_batch(
        None,
        sorted_data,
        0,
        len(sorted_data),
        queries,
        save=False,
        k=10,
        dataset_name="sift",
        size=len(data),
        shuffled_data=False,
        random_queries=False
    )
    plans.append(("Search", np.array(sorted_data), queries, lookup, gt_neighbors, False))
    run_dynamic_test(
        plans, gt_neighbors, gt_dists,
        distance_metric=metric,
        batch_build=batch_build,
        batch_build_data=sorted_data,
        batch_build_tags=[i for i in range(1, len(sorted_data)+1)],
        max_vectors=len(data), experiment_name="sorted_adversarial_10000_")
    indexing_plan = [(0, i) for i in range(len(data))]
    lookup = [(1, i) for i in range(len(queries))]
    plans = [] if batch_build else [("Indexing", data, queries, indexing_plan, None, False)]
    gt_neighbors2, gt_dists2 = get_or_create_ground_truth_batch(None, data, 0, len(data), queries, save=False, k=10, dataset_name="sift", size=len(data))
    print(gt_neighbors[0], gt_neighbors2[0])
    print(gt_dists[0], gt_dists2[0])
    plans.append(("Search", data, queries, lookup, gt_neighbors2, False))
    run_dynamic_test(
        plans, gt_neighbors2, gt_dists2,
        distance_metric=metirc,
        batch_build=batch_build,
        batch_build_data=data,
        batch_build_tags=[i for i in range(1, len(data)+1)],
        max_vectors=len(data), experiment_name="sorted_adversarial_10000_baseline_")


def create_and_save_random_in_distribution_queries(data_prefix_path, dataset_name, dataset, n_queries=10000):
    queries = []
    original_data_suffix = "{}/{}.npy".format(dataset_name, dataset)
    random_queries_suffix = "{}/{}_random_queries.npy".format(dataset_name, dataset)
    try:
        data = np.load(data_prefix_path/original_data_suffix)
    except Exception as ex:
        print(ex)
        data = load_or_create_test_data(data_prefix_path/"{}/{}.hdf5".format(dataset_name, dataset))
    print("Loaded dataset for {} to create in-distribution random queries".format(dataset_name))
    rand_indices = np.random.choice(len(data), n_queries)
    queries = data[rand_indices]
    dimension = len(queries[0])
    apx_min_dist = np.inf
    for i in range(1, len(data)):
        dist = np.linalg.norm(data[i] - data[i-1])
        if dist < apx_min_dist:
            apx_min_dist = dist
    scale = apx_min_dist / 2

    for i in range (n_queries):
        q = np.random.normal(queries[i], scale, size=dimension)
        if q[0] == -np.inf or q[0] == np.inf:
            print(scale)
        queries[i] = q
    np.save(data_prefix_path/random_queries_suffix, queries)


def run_one_experiment_manual():
    # data, queries, _, _ = load_or_create_test_data(path="../data/sift-128-euclidean.hdf5")
    data = np.load("/storage/sylziyuz/new_filtered_ann_datasets/redcaps/redcaps-512-angular.npy")
    data = data[:1000000]
    queries = np.load("/storage/sylziyuz/new_filtered_ann_datasets/redcaps/redcaps-512-angular_queries.npy")
    print(len(queries))
    small_batch_gradual_update_experiment(
        data,
        queries,
        dataset_name="redcaps",
        gt_data_prefix="/storage/sylziyuz",
        setting_name="setting_name",
        size=len(data)/2,
        metric="l2",
        shuffled_data=False,
        random_queries=False
    )

def small_batch_gradual_update_train_test_split_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, query_complexity=64, build_complexity=64, graph_degree=64
):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * 2)
    data = data[:2 * size]
    n_update_batch = 100
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans=[]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"
    all_gt_neighbors, all_gt_dists = get_or_create_rolling_update_ground_truth(
        path=Path(gt_data_prefix +'/ann_rolling_update_gt/'+dataset_name+"_"+metric+"_"+str(size)+"_100"+suffix).expanduser(),
        data=data[:size],
        data_to_update=data[size:2 * size],
        queries=queries,
        save=True,
        k=query_k,
        dataset_name=dataset_name,
        metric=metric,
        shuffled_data=False,
        random_queries=False,
    )
    initial_lookup_gt_neighbors = all_gt_neighbors[0]
    initial_lookup_gt_dists = all_gt_dists[0]
    apx_total_initial_gt = 0.0
    for ans in initial_lookup_gt_dists:
        apx_total_initial_gt += abs(ans[0])
    # apx_average_initial_gt helps perturb the query workload to generating training set
    apx_average_initial_gt = apx_total_initial_gt / len(initial_lookup_gt_dists)

    train_prop = 0.02
    plans = []

    for i in range(0, size, update_batch_size):
        update_plan = []
        for j in range(update_batch_size):
            delete_id = i + j
            insert_id = delete_id + size
            update_plan.append((0, insert_id))
            update_plan.append((2, delete_id))
        plans.append(("Update", data, queries, update_plan, None, False))
        gt_neighbors = all_gt_neighbors[1 + i // update_batch_size]
        gt_dists =all_gt_dists[1 + i // update_batch_size]

        n_train = max(int(queries.shape[0] * train_prop), 100)
        
        train_queries = queries[np.random.choice(queries.shape[0], int(n_train), replace=False)]
        if train_queries.dtype == np.dtype('int8'):
            for j in range(len(train_queries)):
                train_queries[j] += np.random.randint(-math.ceil(apx_average_initial_gt), math.ceil(apx_average_initial_gt), size=train_queries.shape[1], dtype=np.int8)
        else:
            for j in range(len(train_queries)):
                train_queries[j] += np.random.normal(train_queries[j], apx_average_initial_gt, size=train_queries.shape[1])
        
        plans.append(("Train"+str(i), data, train_queries, [(3, i) for i in range(len(train_queries))], gt_neighbors, False))
        plans.append(("Search"+str(i), data, queries, initial_lookup, gt_neighbors, False))
    
    experiment_name = "{}_{}_{}_{}_rolling_update_train_test_split".format(dataset_name, size, setting_name, metric)
    run_dynamic_test(
        plans,
        gt_neighbors,
        gt_dists,
        max_vectors=len(data),
        experiment_name=experiment_name,
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )

def mixed_throughput_experiment(data, queries, dataset_name, gt_data_prefix,
    setting_name="setting_name", size=5000, metric="l2", shuffled_data=False, random_queries=False,
    query_k=10, query_complexity=64, build_complexity=64, graph_degree=64, n_iter=2,
):
    assert(size > 500)
    assert(size % 100 == 0)
    assert(len(data) >= size * n_iter)
    data = data[:n_iter * size]
    n_update_batch = 100 * n_iter
    update_batch_size = size // 100
    n_queries = len(queries)

    indexing_plan = [(0, i) for i in range(size)]
    initial_lookup = [(1, i) for i in range(len(queries))]

    plans=[]
    suffix = ""
    if shuffled_data:
        suffix += "_shuffled"
    if random_queries:
        suffix += "_random_queries"

    for i in range(0, (n_iter - 1) * size, update_batch_size):
        execution_plan = []
        for j in range(update_batch_size):
            delete_id = i + j
            insert_id = delete_id + size
            execution_plan.append((0, insert_id))
            execution_plan.append((2, delete_id))
        # train_queries = queries[np.random.choice(queries.shape[0], min(queries.shape[0] // 10, 200), replace=False)]
        train_queries = queries[np.random.choice(queries.shape[0], queries.shape[0] // 2, replace=False)]
        if train_queries.dtype == np.dtype('int8'):
            for j in range(len(train_queries)):
                train_queries[j] += np.random.randint(-math.ceil(4.9), math.ceil(4.9), size=train_queries.shape[1], dtype=np.int8)
        else:
            for j in range(len(train_queries)):
                train_queries[j] += np.random.normal(train_queries[j], 5, size=train_queries.shape[1])
        all_queries = np.concatenate((queries, train_queries))
        execution_plan += initial_lookup
        execution_plan += [(3, i + len(queries)) for i in range(len(train_queries))]
        random_state = random.Random(i)
        random_state.shuffle(execution_plan)
        consolidate = False
        if (i / update_batch_size) % 10 == 0 and i > 0:
            consolidate = True
        plans.append(("Mixed+Train", data, all_queries, execution_plan, None, consolidate))
        # TODO (SylviaZiyuZhang): Try this first and remove or reduce the synchronization barrier if need be
    
    experiment_name = "{}_{}_{}_{}_mixed_throughput".format(dataset_name, size, setting_name, metric)
    run_dynamic_test(
        plans,
        None,
        None,
        max_vectors=len(data),
        experiment_name=experiment_name,
        distance_metric=metric,
        batch_build=True,
        batch_build_data=data[:size],
        batch_build_tags=[i for i in range(1, size+1)],
        query_k=query_k,
        build_complexity=build_complexity,
        query_complexity=query_complexity,
        graph_degree=graph_degree,
        )

