import diskannpy
import os
import time
import h5py
import numpy as np

query_k = 10
alpha = 1.2
build_complexity = 64
graph_degree = 64
query_complexity = 64

dynamic_test = diskannpy._diskannpy.run_dynamic_test


def num_recalled(found, query_gt, k):
    gt_set = set(query_gt[:k])
    found_set = set(found[:k])
    intersection = gt_set.intersection(found_set)
    return len(intersection)


def test_static_index(
    data,
    queries,
    gt_neighbors,
    dataset_short_name,
    index_directory="indices",
    num_threads=0,
    distance_metric="l2",
):
    prefix = f"{dataset_short_name}-{alpha}-{build_complexity}-{graph_degree}"

    if not os.path.exists(index_directory + "/" + prefix):
        diskannpy.build_memory_index(
            data,
            alpha=alpha,
            complexity=build_complexity,
            graph_degree=graph_degree,
            distance_metric=distance_metric,
            index_directory=index_directory,
            num_threads=0,
            use_pq_build=False,
            use_opq=False,
            index_prefix=prefix,
        )

    index = diskannpy.StaticMemoryIndex(
        index_directory=index_directory,
        num_threads=num_threads,
        initial_search_complexity=query_complexity,
        index_prefix=prefix,
    )

    total_recalled = 0
    start = time.time()
    for query, query_gt in zip(queries, gt_neighbors):
        result = index.search(
            query=query, complexity=query_complexity, k_neighbors=query_k
        )
        total_recalled += num_recalled(result.identifiers, query_gt, query_k)
    end = time.time()

    recall = total_recalled / len(queries) / query_k

    print(f"Static index has recall {recall} and took {end - start:.2f} seconds")


def parse_ann_benchmarks_hdf5(data_path):
    with h5py.File(data_path, "r") as file:
        gt_neighbors = np.array(file["neighbors"])
        queries = np.array(file["test"])
        data = np.array(file["train"])

        return data, queries, gt_neighbors


# plans should be a list of pairs of the form (plan_name, data, queries, update_list, Optional[ground_truth])
def run_dynamic_test(plans, max_vectors, threads=[1, 2, 4, 8], distance_metric="l2"):
    time_keys = [plan[0] for plan in plans] + ["Total"]
    all_times = {time_key: [] for time_key in time_keys}

    for num_threads in threads:
        start_overall_time = time.time()

        dynamic_index = diskannpy.DynamicMemoryIndex(
            distance_metric=distance_metric,
            vector_dtype=np.float32,
            dimensions=plans[0][1].shape[1],
            max_vectors=max_vectors,
            complexity=build_complexity,
            graph_degree=graph_degree,
        )

        for plan_name, data, queries, update_list, optional_gt in plans:
            start_plan_time = time.time()

            dynamic_test(
                dynamic_index._index,
                data,
                queries,
                update_list,
                query_k=query_k,
                query_complexity=query_complexity,
                num_threads=num_threads,
            )

            all_times[plan_name].append(time.time() - start_plan_time)

        all_times["Total"].append(time.time() - start_overall_time)

        new_times = [all_times[time_key][-1] for time_key in time_keys]
        firsts_times = [all_times[time_key][0] for time_key in time_keys]
        speedups = [f / n for n, f in zip(new_times, firsts_times)]
        print(
            f"Times with {num_threads} threads: " + str(list(zip(time_keys, new_times)))
        )
        print(
            f"Speedups with {num_threads} threads: "
            + str(list(zip(time_keys, speedups)))
        )
