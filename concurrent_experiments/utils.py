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
    query_complexity=64,
    query_k=10,
    alpha=1.2,
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
        queries = np.array(file["test"])
        data = np.array(file["train"])
        gt_neighbors = np.array(file["neighbors"]) if "neighbors" in file.keys() else []
        gt_dists = np.array(file["distances"]) if "distances" in file.keys() else []

        return data, queries, gt_neighbors, gt_dists


# plans should be a list of pairs of the form (plan_name, data, queries, update_list, Optional[ground_truth])
def run_dynamic_test(plans, neighbors, dists, max_vectors,
    experiment_name="trial", threads=[8], distance_metric="l2",
    batch_build=False, batch_build_data=None, batch_build_tags=None,
    build_complexity=64, graph_degree=64, query_complexity=64, query_k=10,
):
    time_keys = [plan[0] for plan in plans] + ["Total"]
    all_times = {time_key: [] for time_key in time_keys}
    recall_keys = [plan[0] for plan in plans] + ["Recall"]
    all_recalls = {recall_key: [] for recall_key in recall_keys}

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
        if batch_build:
            assert(len(batch_build_data) == len(batch_build_tags))
            dynamic_index._index.build(batch_build_data, len(batch_build_data), batch_build_tags)

        all_recalls_list = []
        all_mses_list = []
        all_latencies_list = []
        all_num_updates_list = []
        plan_names_list = []
        plan_ids_list = []
        cur_plan = 0
        for plan_name, data, queries, update_list, optional_gt, plan_consolidate in plans:
            start_plan_time = time.time()
            recall_count = 0
            search_count = 0
            actual_queries = []
            actual_update_list = []
            for i, it in enumerate(update_list):
                if it[0] == 1 and it[1] < len(queries):
                    actual_queries.append(queries[it[1]])
                    actual_update_list.append((it[0], len(actual_queries) - 1))
                else:
                    actual_update_list.append(it)
            
            consolidate = 1 if plan_consolidate else 0

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
  
            mse_total = 0
            if optional_gt is not None:
                for i, it in enumerate(update_list):
                    if it[0] == 1 and it[1] < len(queries): # A search query
                        largest_returned = 0
                        largest_true = 0
                        for k in range(query_k):
                            if results[0][search_count][k] in optional_gt[i][:query_k]:
                                recall_count += 1
                            if largest_returned < results[1][search_count][k]:
                                largest_returned = results[1][search_count][k]
                            if largest_true < optional_gt[i][k]:
                                largest_true = optional_gt[i][k]
                        mse_total += np.square(largest_returned - largest_true)
                        search_count += 1
            recall = -1 if search_count == 0 else (recall_count / (search_count * query_k))
            mse = -1 if search_count == 0 else mse_total / search_count
            plan_total_time = time.time() - start_plan_time
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
            cur_plan += 1

        all_times["Total"].append(time.time() - start_overall_time)

        new_times = [all_times[time_key][-1] for time_key in time_keys]
        firsts_times = [all_times[time_key][0] for time_key in time_keys]
        speedups = [f / n for n, f in zip(new_times, firsts_times)]

        result = {
            "num_threads": num_threads,
            "plan_names": plan_names_list,
            "recalls": all_recalls_list,
            "mses": all_mses_list,
            "latencies": all_latencies_list,
            "num_updates": all_num_updates_list,
            "new_times": new_times,
            "speedups": speedups
        }
        with open(experiment_name+'_result_data.json', 'w') as f:
            json.dump(result, f)

        start_plotting_index = -1
        # Amputate the recalls
        for i, it in enumerate(all_recalls_list):
            if start_plotting_index == -1 and it != -1: # Only start plotting when recall values are meaningful
                start_plotting_index = i
            if it == -1 and i > 0 and all_recalls_list[i-1] != -1:
                all_recalls_list[i] = all_recalls_list[i-1] # amputate invalid value with the previous value
        
        for i, it in enumerate(all_latencies_list):
            if plan_names_list[i] == "Update":
                all_latencies_list[i] = all_latencies_list[i+1]
        recalls_to_plot = all_recalls_list[start_plotting_index:]
        latencies_to_plot = all_latencies_list[start_plotting_index:]
        plan_ids_to_plot = plan_ids_list[start_plotting_index:]
        """
        if num_threads == 8:
            #plt.plot(plan_ids_to_plot, recalls_to_plot, label='Recall 10@10')

            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('batch')
            ax1.set_ylabel('Recall 10@10', color=color)
            line1, = ax1.plot(plan_ids_to_plot, recalls_to_plot, color=color, label='Recall 10@10')
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Throughput', color=color)
            line2, = ax2.plot(plan_ids_to_plot, latencies_to_plot, color=color, label='Latency')
            ax2.tick_params(axis='y', labelcolor=color)

            # Combine the legend handles and labels from both axes
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]

            # Display legend on the first axis
            ax1.legend(lines, labels, loc='upper left')

            fig.tight_layout()
            plt.title('Recall and Latency Plot on Consolidation')
            plt.savefig(experiment_name + 'recall_latency_plot.png')
            plt.show()
            
            #plt.plot(plan_ids_to_plot, latencies_to_plot, label='latency per query')
            #plt.xlabel('Batch')
            #plt.title('Recall and latency plot on consolidation')
            #plt.legend()
            #plt.savefig(experiment_name+'recall_latency_plot.png')
            
        # plt.show()
        """
