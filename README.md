# Overview 

This repository is a concurrent dynamic approximate nearest neighbor search index built on [the Microsoft DiskANN algorithm](https://github.com/microsoft/DiskANN) (commit #35f8cf7).

# Commands

You can build the library and the python binding with with
```
python3 -m pip install --no-build-isolation -ve .
```

You may first need to install pybind11:
```
python3 -m pip install pybind11
```

If you get output saying pip is installing UNKNOWN, first run
```
python3 -m pip install setuptools --upgrade
```

You will also likely need to install the packages the original DiskANN needs:
```
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```
and
```
sudo apt install libmkl-full-dev
```
You may also need to tell CMake where omp is installed in `CMakeLists.txt` by setting `POSSIBLE_OMP_PATHS`.

See the original readme (https://github.com/microsoft/DiskANN/tree/main#readme).


# Experiment Code

Experiments are in `concurrent_experiments`. They can be run as e.g.

You can to configure the system in `concurrent_experiments/utils.py`. Parameters you can configure include the graph sparsity regularizer, graph degree limit, starting memory allocation, number of hyper-threads, and so on. 
To run experiments in batch, refer to `concurrent_experiments/experiment_batch_runner.py`. A sample batch experiment script is batch_run_experiment.sh.

To configure whether to use one or more algorithms proposed in our paper, please modify the compilation macro in `src/index.cpp` for now.
* `LAYER_BASED_PATH_COMPRESSION` corresponds to __GuidedBridgeBuild__.
* `FIXES_DELETES_LOWER_LAYER` corresponds to __on-the-fly consolidation__.
* `MEMORY_COLLECTION` corresponds to __semi-lazy cleaning__.

# Datasets

The existing experiments expect the dataset to be downloaded in a folder named data in hdf5 format named as {dataset-name}-{dimension}-{metric}.hdf5. The hdf5 formatting is consistent with the ANN benchmarks project: in the hdf5 file, indexed data points are under the "train" dataset, and test data points are under the "test" dataset. Since almost all datasets do not come with sliding window update ground truths for the queries, `concurrent_experiments/test_rolling_update.get_or_create_rolling_update_ground_truth` computes and stores the ground truth for the dataset being tested on.

For the current manuscript, we conducted experiments on 7 datasets of different nature on various scales. The datasets used are listed below:

|Dataset | Direct source | d(*)| Sizes| Data Type | Domain | Distribution Shift|
| -| -| -| -| -| -| -|
| Adversarial | synthetic | Euclidean | 10k - 1M | float | Spatial | ✅ |
| GloVe | ann-benchmark | Cosine | 10k - 1M | float | Word Representation | ❌ |
| HuffPost | Huggingface | Cosine | 10k - 15k | float | Short Text Representation | ✅ |
| RedCaps | redcaps.xyz | Cosine | 10k - 1M | float | Text-to-Multimodal Search| ✅ |
| Sift | ann-benchmark | Euclidean | 10k - 1M | float | Image | ❌ |
| SpaceV | Microsoft | Euclidean | 10k - 10M | float | Web Search | ✅ |
| Yandex-tti | Yandex | MIPS | 10k - 1M | float | Text-to-Image | ❌ |

* [RedCaps](https://zenodo.org/records/13137120) was embedded with CLIP and [HuffPost](https://zenodo.org/records/13137331) was embedded with OpenAI's text-embedding-3 API. Both are sorted in timestamp to represent a realistic distribution shift. We linked the curated versions above.
* The script for generating Adversarial can be found in `concurrent_experiments/adversarial-dataset-gen.py`.

# Paper

This repository is the artifact for the manuscript CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search.

# Current Results

![adversarial_50000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/e6901c0c-9251-4cbb-b453-b3490901e052 "Adversarial Recall")
![adversarial_50000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/0856b21f-536c-4fa7-bc94-223d3f526fe9 "Adversarial Throughput")
![glove_50000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/6f00f503-d538-43ed-99ec-790d679ce509 "GloVe Recall")
![glove_50000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/9d8d467e-ac19-4b6a-9f36-6ec5a20c412e "GloVe Throughput")
![huffpost_50000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/d5de2c31-36c4-4e77-ad6b-282e37a7c818 "HuffPost Recall")
![huffpost_50000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/4c513b28-ed79-4941-b5b0-7741c5770ad3 "HuffPost Throughput")
![redcaps_500000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/f81e8838-8d10-412b-8de4-734c35ec699e "RedCaps Recall")
![redcaps_500000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/a820e450-5940-4ab2-99f8-f19fe65038d3 "RedCaps Throughput")
![sift_50000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/5ebd99b6-2474-47af-a79d-002f6621a7ee "Sift Recall")
![sift_50000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/ef1cbdca-fdf1-490a-9b5e-7484afb22861 "Sift Throughput")
![spacev-30m_500000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/74654456-9fa6-4e58-a0ac-d313e28fa9ee "SpaceV Recall")
![spacev-30m_500000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/4f87b3bd-cde7-4ca2-8119-565c5194fe73 "SpaceV Throughput")
![yandextti-10m_500000_recall_plot_param_sweep_1](https://github.com/user-attachments/assets/3c9f89ee-5ea8-49d8-95e8-71787fa04fef "Yandex Recall")
![yandextti-10m_500000_xput_plot_param_sweep_1](https://github.com/user-attachments/assets/d1fefe9f-e114-49d4-bb40-9ef43192810e "Yandex Throughput")


# Acknowledgement
* Some utility code comes from [Josh's fork](https://github.com/JoshEngels/ConcurrentANN)

