import struct
import numpy as np
import os

desired_total_samples = 200000000 # two million
desired_index_size = 100000000

part_count = len(os.listdir('vectors.bin'))
for i in range(1, part_count + 1):
    fvec = open(os.path.join('vectors.bin', 'vectors_%d.bin' % i), 'rb')
    if i == 1:
        vec_count = struct.unpack('i', fvec.read(4))[0]
        vec_dimension = struct.unpack('i', fvec.read(4))[0]
        vecbuf = bytearray(vec_count * vec_dimension)
        vecbuf_offset = 0
    while True:
        part = fvec.read(1048576)
        if len(part) == 0: break
        vecbuf[vecbuf_offset: vecbuf_offset + len(part)] = part
        vecbuf_offset += len(part)
    fvec.close()

X = np.frombuffer(vecbuf, dtype=np.int8).reshape((vec_count, vec_dimension))
sampled_indices_total = np.random.choice(X.shape[0], size=desired_samples, replace=False)
sampled_vectors_total = X[sampled_indices, :]
X_index = sampled_vectors_total[:desired_index_size]
X_update = sampled_vectors_total[desired_index_size:]

fq = open('query.bin', 'rb')
q_count = struct.unpack('i', fq.read(4))[0]
q_dimension = struct.unpack('i', fq.read(4))[0]
queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

"""
ftruth = open('truth.bin', 'rb')
t_count = struct.unpack('i', ftruth.read(4))[0]
topk = struct.unpack('i', ftruth.read(4))[0]
truth_vids = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
truth_distances = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))
"""