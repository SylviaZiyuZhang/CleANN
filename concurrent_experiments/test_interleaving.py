import numpy as np
from utils import parse_ann_benchmarks_hdf5, run_dynamic_test, test_static_index

distance_metric = "euclidean"
dataset_name = "sift-128-euclidean.hdf5"
dataset_short_name = dataset_name.split("-")[0]

data_path = f"data/{dataset_name}"
data, queries, gt_neighbors = parse_ann_benchmarks_hdf5(data_path)

data = data[:10000]

num_adversarial_per_data = 16
adversarial_queries = []
for d in data:
    for i in range(num_adversarial_per_data):
        adversarial_queries.append(d + 1 * np.random.normal(size=len(d)))
adversarial_queries = np.array(adversarial_queries)
queries = adversarial_queries


# "Easy", all adversarial queries come at the end
print("Adversarial queries, all at the end:")

indexing_plan = [(0, i) for i in range(len(data))]
querying_plan = [(1, i) for i in range(len(queries))]

plans = [
    ("Indexing", data, [], indexing_plan, None),
    ("Querying", [], queries, querying_plan, None),
]
run_dynamic_test(plans, max_vectors=len(data))


# "Hard", adversarial queries are at the worst position (right after their nearest neighbor)
print("Adversarial queries, mixed in at the worst position:")

indexing_and_querying_plan = []
for i in range(len(data)):
    indexing_and_querying_plan.append((0, i))
    for j in range(num_adversarial_per_data):
        indexing_and_querying_plan.append((1, i * num_adversarial_per_data + j))

plans = [
    ("Adversarially Mixed", data, queries, indexing_and_querying_plan, None),
]
run_dynamic_test(plans, max_vectors=len(data))
