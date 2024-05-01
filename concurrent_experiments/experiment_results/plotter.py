import matplotlib.pyplot as plt
import json
#plt.plot(plan_ids_to_plot, recalls_to_plot, label='Recall 10@10')
baseline_f = open("redcaps_1M_fresh_update_no_consolidation_baseline_result_data.json")
baseline = json.load(baseline_f)
baseline_f.close()
static_f = open("redcaps_no_shuffle_1M_fresh_update_static_recalls.json")
static = json.load(static_f)
static_f.close()
baseline_consolidate_f = open("redcaps_no_shuffle_1M_fresh_update_consolidate_result_data.json")
baseline_consolidate = json.load(baseline_consolidate_f)
baseline_consolidate_f.close()
compress_search_path_f = open("redcaps_no_shuffle_1M_fresh_update_no_consolidate_distance_compression_not_inter_insert_result_data.json")
compress_search_path = json.load(compress_search_path_f)
compress_search_path_f.close()
compress_neighbor_path_f = open("redcaps_no_shuffle_1M_fresh_update_no_consolidate_distance_compression_inter_insert_4_result_data.json")
compress_neighbor_path = json.load(compress_neighbor_path_f)
compress_neighbor_path_f.close()
compress_neighbor_path_consolidate_f = open("redcaps_no_shuffle_1M_fresh_update_consolidate_10_perc_distance_compression_inter_insert_4_result_data.json")
compress_neighbor_path_consolidate = json.load(compress_neighbor_path_consolidate_f)
compress_neighbor_path_consolidate_f.close()
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('batch')
ax1.set_ylabel('Recall 10@10', color=color)
print(baseline["plan_names"])
print(baseline["recalls"])
line1, = ax1.plot(list(filter(lambda s: "Search" in s, baseline["plan_names"])), list(filter(lambda x: x > 0, baseline["recalls"])), color="blue", label='baseline no consolidation')
line2, = ax1.plot(list(filter(lambda s: "Search" in s, baseline_consolidate["plan_names"])), list(filter(lambda x: x > 0, baseline_consolidate["recalls"])), color="black", label='baseline consolidate')
line3, = ax1.plot(list(filter(lambda s: "Search" in s, static["plan_names"])), list(filter(lambda x: x > 0, static["recalls"])), color="grey", label='static')
line4, = ax1.plot(list(filter(lambda s: "Search" in s, compress_search_path["plan_names"])), list(filter(lambda x: x > 0, compress_search_path["recalls"])), color="orange", label='compress search path')
line5, = ax1.plot(list(filter(lambda s: "Search" in s, compress_neighbor_path["plan_names"])), list(filter(lambda x: x > 0, compress_neighbor_path["recalls"])), color="red", label='compress neighbor path')
line6, = ax1.plot(list(filter(lambda s: "Search" in s, compress_neighbor_path_consolidate["plan_names"])), list(filter(lambda x: x > 0, compress_neighbor_path_consolidate["recalls"])), color="pink", label='compress neighbor path+consolidate')
ax1.tick_params(axis='y', labelcolor=color)

"""
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Throughput', color=color)
line2, = ax2.plot(plan_ids_to_plot, latencies_to_plot, color=color, label='Latency')
ax2.tick_params(axis='y', labelcolor=color)
"""

# Combine the legend handles and labels from both axes
lines = [line1, line2, line3, line4, line5, line6]
labels = [line.get_label() for line in lines]

# Display legend on the first axis
ax1.legend(lines, labels, loc='lower left')

fig.tight_layout()
plt.title('Trial plot')
plt.savefig('redcaps_1M_recall_plot.png')
plt.show()

#plt.plot(plan_ids_to_plot, latencies_to_plot, label='latency per query')
#plt.xlabel('Batch')
#plt.title('Recall and latency plot on consolidation')
#plt.legend()
#plt.savefig(experiment_name+'recall_latency_plot.png')

# plt.show()