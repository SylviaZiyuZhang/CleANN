# Overview 

Concurrent DiskANN project, forked from https://github.com/microsoft/DiskANN, commit #35f8cf7

# Commands

Fast incremental builds can be done with
```
python3 -m pip install --no-build-isolation -ve .
```
If you only change a .cc file, this should be relatively fast.

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
See the original readme (https://github.com/microsoft/DiskANN/tree/main#readme).

To develop in vscode and have syntax highlighting work (tested with the clangd extension), run
```
scripts/generate-compile-commands.sh
```

# Experiment Code

Existing experiments are in concurrent_experiments. They can be run as e.g.

```
python3 concurrent_experiments/test_tough_data.py
```

Excerpts of the output of a clean run of each experiment are in concurrent_experiments/experiment_results.

The existing experiments use a testing framework that should be mostly self explanatory.

# Datasets

The existing experiments expect the sift dataset to be downloaded in a folder named data in hdf5 format. 
The file can be downloaded from the ANN benchmarks project, like so:

```
mkdir data
cd data
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
```

