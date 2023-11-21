# Overview 

Concurrent DiskANN project, forked from https://github.com/microsoft/DiskANN, commit #35f8cf7

# Commands

Fast incremental builds can be done with
```
pip3 install --no-build-isolation -ve .
```
If you only change a .cc file, this should be relatively fast.

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

