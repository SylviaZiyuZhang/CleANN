import numpy as np
import h5py

dim = 128
data = np.zeros((1000000, 128))
queries = np.zeros((10000, 128))

for c in range(10000):
    base_vec = np.random.rand(dim)
    for d in range(dim):
        base_vec[d] *= np.random.randint(-1000, 1000)
    for i in range(100):
        data[c * 100 + i] = base_vec + np.random.normal(0, 10, size=base_vec.shape)

for c in range(10000):
    query_vec = np.random.rand(dim)
    for d in range(dim):
        query_vec[d] *= np.random.randint(-1000, 1000)
    queries[c] = query_vec

with h5py.File("adversarial-128-euclidean.hdf5", 'w') as file:
    file.create_dataset("train", data=data)
    file.create_dataset("test", data=queries)

