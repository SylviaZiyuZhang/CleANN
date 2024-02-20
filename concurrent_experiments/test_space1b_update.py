import heapq
import numpy as np
import h5py # FileNotFoundError
import struct
import numpy as np
import os
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test

desired_total_samples = 200000000 # two million
desired_index_size = 100000000
spacev_batch_size = desired_index_size // 100

def parse_space1b_benchmark():
    desired_total_samples = 200000000 # two million
    desired_index_size = 100000000

    part_count = len(os.listdir('../SPACEV1B/vectors.bin'))
    for i in range(1, part_count + 1):
        fvec = open(os.path.join('../SPACEV1B/vectors.bin', 'vectors_%d.bin' % i), 'rb')
        if i == 1:
            vec_count = struct.unpack('i', fvec.read(4))[0]
            vec_dimension = struct.unpack('i', fvec.read(4))[0]
            try:
                vecbuf = bytearray(vec_count * vec_dimension)
            except MemoryError as ex:
                import traceback
                traceback.print_exc()
                print(part_count)
                print(vec_count)
                print(vec_dimension)
                print(vec_count * vec_dimension)
                raise ex
            vecbuf_offset = 0
        while True:
            part = fvec.read(1048576)
            if len(part) == 0: break
            vecbuf[vecbuf_offset: vecbuf_offset + len(part)] = part
            vecbuf_offset += len(part)
        fvec.close()

    X = np.frombuffer(vecbuf, dtype=np.int8).reshape((vec_count, vec_dimension))
    sampled_indices_total = np.random.choice(X.shape[0], size=desired_samples, replace=False)
    sampled_vectors_total = X[sampled_indices, :]
    X_index = sampled_vectors_total[:desired_index_size]
    X_update = sampled_vectors_total[desired_index_size:]

    fq = open('../SPACEV1B/query.bin', 'rb')
    q_count = struct.unpack('i', fq.read(4))[0]
    q_dimension = struct.unpack('i', fq.read(4))[0]
    queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

    return X_index, X_update, queries
    """
    ftruth = open('../SPACEV1B/truth.bin', 'rb')
    t_count = struct.unpack('i', ftruth.read(4))[0]
    topk = struct.unpack('i', ftruth.read(4))[0]
    truth_vids = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
    truth_distances = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))
    """

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
        heapq.heappush(neighbors, (-np.linalg.norm(query-data[i]), i))
        cur_k += 1
        if cur_k > k:
            heapq.heappop(neighbors)
    neighbor_ids = [i for (_, i) in neighbors]
    dists = [-d for (d, _) in neighbors]
    #print(start, end)
    #print(len(neighbor_ids))
    assert len(neighbor_ids) == k
    assert len(dists) == k
    return neighbor_ids, dists

def spacev_update_experiment():
    pass


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

# half_dataset_update_experiment(data, queries, gt_neighbors, gt_dists)
# close_insertion_experiment(data, queries)
parse_space1b_benchmark()