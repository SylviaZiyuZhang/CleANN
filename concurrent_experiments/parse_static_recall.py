import json
import os

def parse_static_recall(experiment_name="redcaps_no_shuffle_10000_fresh_update"):
    # Directory containing JSON files
    directory = './'

    # List to store parsed JSON data
    all_data = {
        "plan_names": [],
        "recalls": [],
        "latencies": [],
        "num_updates": [],
        "new_times": [], # the last item in here is total
        "speedups": [], # the last item in here is total
    }

    # Function to extract the numeric part of the filename
    def extract_number(filename):
        return int(filename.split('_')[8])  # Assuming filename is like experiment_1_data.json

    # Get list of JSON filenames and sort them numerically
    json_files = sorted([filename for filename in os.listdir(directory) if filename.endswith('.json')],
                        key=extract_number)

    # Loop through each sorted JSON file
    for filename in json_files:
        file_path = os.path.join(directory, filename)
        search_start_id = extract_number(filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            all_data["num_threads"] = data["num_threads"]
            all_data["plan_names"].append("Indexing"+str(search_start_id))
            all_data["plan_names"].append("Search"+str(search_start_id))
            all_data["recalls"] += data["recalls"]
            all_data["latencies"] += data["latencies"]
            all_data["num_updates"] += data["num_updates"]
            all_data["new_times"] += data["new_times"][:-1]
            all_data["speedups"] += data["speedups"][:-1]
    
    with open(experiment_name+"_static_recalls.json", 'w') as f:
        json.dump(all_data, f)

parse_static_recall()