import diskannpy
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json


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
    graph_degree=64,
    build_complexity=64,
    insert_complexity=64,
    query_complexity=64,
    query_k=10,
    alpha=1.2,
    bridge_start_lb=3,
    bridge_start_hb=5,
    bridge_end_lb=9,
    bridge_end_hb=64,
    bridge_prob=0.5,
):
    prefix = f"{dataset_short_name}-{alpha}-{build_complexity}-{graph_degree}"

    if not os.path.exists(index_directory + "/" + prefix):
        diskannpy.build_memory_index(
            data,
            alpha=alpha,
            complexity=build_complexity,
            insert_complexity=64,
            graph_degree=graph_degree,
            distance_metric=distance_metric,
            index_directory=index_directory,
            num_threads=0,
            bridge_start_lb=3,
            bridge_start_hb=5,
            bridge_end_lb=9,
            bridge_end_hb=64,
            bridge_prob=0.5,
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
        queries = np.array(file["test"])
        data = np.array(file["train"])
        gt_neighbors = np.array(file["neighbors"]) if "neighbors" in file.keys() else []
        gt_dists = np.array(file["distances"]) if "distances" in file.keys() else []

        return data, queries, gt_neighbors, gt_dists


# plans should be a list of pairs of the form (plan_name, data, queries, update_list, Optional[ground_truth])
def run_dynamic_test(plans, neighbors, dists, max_vectors,
    experiment_name="trial", threads=[56], distance_metric="l2",
    batch_build=False, batch_build_data=None, batch_build_tags=None,
    build_complexity=64, insert_complexity=64, graph_degree=64, query_complexity=64, query_k=10,
    bridge_start_lb=3, bridge_start_hb=5, bridge_end_lb=9, bridge_end_hb=64, bridge_prob=0.5,
):

    settings = [ # name, alpha, build complexity, insert complexity, query_complexity, bridge_start_lb, bridge_start_hb, bridge_end_lb, bridge_end_hb, bridge_prob
        #('C', 1.2, 64, 32, 128, 3, 6, 9, 64, 0.3),
        #('D', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.1),
        #('E', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.5),
        #('F', 1.2, 64, 32, 64, 3, 6, 9, 12, 0.7),
        #('G', 1.2, 64, 32, 32, 3, 6, 9, 12, 0.7),
        #('H', 1.2, 64, 32, 32, 9, 12, 9, 12, 0.3)
        # param_sweep_6
        # ('C1', 1, 64, 32, 128, 3, 6, 9, 64, 0.1),
        # ('C4', 1.3, 64, 32, 128, 3, 6, 9, 64, 0.1),
        # ('A1', 1, 64, 64, 100, 3, 6, 9, 64, 0.1),
        # ('A4', 1.3, 64, 64, 100, 3, 6, 9, 64, 0.1),
        # ('B4', 1.3, 64, 64, 75, 3, 6, 9, 64, 0.1),
        # ('D1', 1, 64, 64, 64, 3, 6, 9, 64, 0.1),
        # ('D2', 1.1, 64, 64, 64, 3, 6, 9, 64, 0.1),
        # ('D3', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.1),
        # ('D4', 1.3, 64, 64, 64, 3, 6, 9, 64, 0.1),
        # ('E1', 1, 64, 64, 64, 3, 6, 9, 64, 0.5),
        # ('E4', 1.3, 64, 64, 64, 3, 6, 9, 64, 0.5),
        # baseline_sweep_2
        #('baseline_B4', 1.3, 64, 64, 75, 3, 6, 9, 64, 0.1),
        #('baseline_D3', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.1),
        #('baseline_A1', 1, 64, 64, 100, 3, 6, 9, 64, 0.1),
        #('baseline_C3', 1.2, 64, 32, 128, 3, 6, 9, 64, 0.1),
        #('baseline_C4', 1.3, 64, 32, 128, 3, 6, 9, 64, 0.1),

        #('boundary_insert_test', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.1), # mixed_throughput_cleann was measured here, mixed_throughput_consolidate
        # ('static_recompute', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.1),
        ('reverse_freshvamana_naive_trial', 1.2, 64, 64, 64, 3, 6, 9, 64, 0.1),
    ]
    for setting in settings:
        setting_name, alpha, build_complexity, insert_complexity, query_complexity, bridge_start_lb, bridge_start_hb, bridge_end_lb, bridge_end_hb, bridge_prob = setting
        time_keys = [plan[0] for plan in plans] + ["Total"]
        all_times = {time_key: [] for time_key in time_keys}
        recall_keys = [plan[0] for plan in plans] + ["Recall"]
        all_recalls = {recall_key: [] for recall_key in recall_keys}
        build_time = 0

        for num_threads in threads:
            res_file_name = experiment_name+'_'+setting_name+'_t'+str(num_threads)+'_result_data.json'
            start_overall_time = time.time()
            dynamic_index = diskannpy.DynamicMemoryIndex(
                distance_metric=distance_metric,
                alpha=alpha,
                vector_dtype=np.float32,
                dimensions=plans[0][1].shape[1],
                max_vectors=max_vectors,
                complexity=build_complexity,
                insert_complexity=insert_complexity,
                graph_degree=graph_degree,
                bridge_start_lb=bridge_start_lb,
                bridge_start_hb=bridge_start_hb,
                bridge_end_lb=bridge_end_lb,
                bridge_end_hb=bridge_end_hb,
                bridge_prob=bridge_prob,
            )
            if batch_build:
                start_build_time = time.time()
                assert(len(batch_build_data) == len(batch_build_tags))
                dynamic_index._index.build(batch_build_data, len(batch_build_data), batch_build_tags)
                build_time = time.time() - start_build_time
            else:
                dynamic_index._index.set_start_points_at_random(radius=10.0, random_seed=42)

            all_recalls_list = []
            all_mses_list = []
            all_latencies_list = []
            all_num_updates_list = []
            plan_names_list = []
            plan_ids_list = []
            cur_plan = 0
            p99_list = []
            p50_list = []
            p90_list = []
            for plan_name, data, queries, update_list, optional_gt, plan_consolidate in plans:
                print("Starting plan ", plan_name)
                dynamic_index._index.print_status()
                recall_count = 0
                search_count = 0
                test_search_count = 0
                actual_queries = []
                actual_update_list = []
                for it in update_list:
                    if it[0] == 1 or it[0] == 3 and it[1] < len(queries): # 1: test queries, 3: train queries
                        actual_queries.append(queries[it[1]])
                        actual_update_list.append((it[0], len(actual_queries) - 1))
                    else:
                        actual_update_list.append(it)

                consolidate = 1 if plan_consolidate else 0
                start_plan_time = time.time()
                results = dynamic_test(
                    dynamic_index._index,
                    data,
                    actual_queries,
                    actual_update_list,
                    query_k=query_k,
                    query_complexity=query_complexity,
                    num_threads=num_threads,
                    consolidate=consolidate,
                    plan_id=cur_plan,
                )
                plan_total_time = time.time() - start_plan_time
                mse_total = 0
                if optional_gt is not None:
                    for i, it in enumerate(update_list):
                        if it[0] == 1 and it[1] < len(queries): # A search query that does not allow improvements
                            largest_returned = 0
                            largest_true = 0
                            if True:
                                for k in range(query_k):
                                    if results[0][search_count][k] in optional_gt[i][:query_k]:
                                        recall_count += 1
                                    if largest_returned < results[1][search_count][k]:
                                        largest_returned = results[1][search_count][k]
                                    if largest_true < optional_gt[i][k]:
                                        largest_true = optional_gt[i][k]
                                mse_total += np.square(largest_returned - largest_true)
                                test_search_count += 1
                            search_count += 1
                recall = -1 if search_count == 0 else (recall_count / (test_search_count * query_k))
                mse = -1 if search_count == 0 else mse_total / search_count
                all_times[plan_name].append(plan_total_time)
                all_recalls[plan_name].append(recall)
                # Ths following are for generating plots
                all_recalls_list.append(recall)
                all_mses_list.append(mse)
                num_updates = len(update_list) if len(update_list) > 0 else 1
                all_num_updates_list.append(num_updates)
                all_latencies_list.append(plan_total_time / num_updates)
                plan_ids_list.append(cur_plan)
                plan_names_list.append(plan_name)
                p99_list.append(np.percentile(results[2], 99))
                p50_list.append(np.percentile(results[2], 50))
                p90_list.append(np.percentile(results[2], 90))
                cur_plan += 1


            all_times["Total"].append(time.time() - start_overall_time)

            new_times = [all_times[time_key][-1] for time_key in time_keys]
            firsts_times = [all_times[time_key][0] for time_key in time_keys]
            speedups = [f / n for n, f in zip(new_times, firsts_times)]

            result = {
                "num_threads": num_threads,
                "build_time": build_time,
                "plan_names": plan_names_list,
                "recalls": all_recalls_list,
                "mses": all_mses_list,
                "latencies": all_latencies_list,
                "p99_latencies": p99_list,
                "p50_latencies": p50_list,
                "p90_latencies": p90_list,
                "alpha": alpha,
                "query_k": query_k,
                "num_updates": all_num_updates_list,
                "new_times": new_times,
                "speedups": speedups,
                "setting_name": setting_name,
                "build_complexity": build_complexity,
                "query_complexity": query_complexity,
                "insert_complexity": insert_complexity,
                "bridge_prob": bridge_prob,
                "bridge_start_lb": bridge_start_lb,
                "bridge_start_hb": bridge_start_hb,
                "bridge_end_lb": bridge_end_lb,
                "bridge_end_hb": bridge_end_hb,
            }
            with open(res_file_name, 'w') as f:
                json.dump(result, f)
