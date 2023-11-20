import diskannpy
import os
import time


def num_recalled(found, query_gt, k):
    gt_set = set(query_gt[:k])
    found_set = set(found[:k])
    intersection = gt_set.intersection(found_set)
    return len(intersection)


def test_static_index(
    data,
    queries,
    gt_neighbors,
    build_complexity,
    graph_degree,
    alpha,
    query_complexity,
    query_k,
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
        result = index.search(query=query, complexity=query_complexity, k_neighbors=query_k)
        total_recalled += num_recalled(result.identifiers, query_gt, query_k)
    end = time.time()

    recall = total_recalled / len(queries) / query_k

    print("Static", recall, end - start)
