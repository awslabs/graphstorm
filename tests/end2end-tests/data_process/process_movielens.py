#!/usr/bin/env python

import pandas
import numpy as np
import h5py
import pyarrow.parquet as pq
import pyarrow as pa

# process user data
user = pandas.read_csv('u.user', delimiter='|', header=None,
        names=["id", "age", "gender", "occupation", "zipcode"])

age = np.array(user['age'])
age = np.expand_dims(age/np.max(user['age']), axis=1)

gender = user['gender']
gender_dict = {'M': 0, 'F': 1}
gender = np.expand_dims(np.array([gender_dict[g] for g in gender]), axis=1)

occupations = np.unique(np.array(user['occupation']))
occ_dict = {occ: i for i, occ in enumerate(occupations)}
occupation = np.array([occ_dict[o] for o in user['occupation']])
occ_one_hot = np.zeros((len(occupation), len(occ_dict)), dtype=np.float32)
for x, y in zip(np.arange(len(occupation)), occupation):
    occ_one_hot[x,y] = 1

feat = np.concatenate([age, gender, occ_one_hot], axis=1)

with h5py.File("user.hdf5", "w") as f:
    ids = user['id']
    arr = f.create_dataset("id", ids.shape, dtype=ids.dtype)
    arr[:] = ids
    arr = f.create_dataset("feat", feat.shape, dtype=feat.dtype)
    arr[:] = feat

# process movie data
movie = pandas.read_csv('u.item', delimiter='|', encoding="ISO-8859-1", header=None)
title = movie[1]
attrs = []
for i in range(5, 24):
    attrs.append(np.array(movie[i]))
attrs = np.stack(attrs, axis=1)
ids = np.array(movie[0])

def write_data_parquet(data, data_file):
    arr_dict = {}
    for key in data:
        arr = data[key]
        assert len(arr.shape) == 1 or len(arr.shape) == 2, \
                "We can only write a vector or a matrix to a parquet file."
        if len(arr.shape) == 1:
            arr_dict[key] = arr
        else:
            arr_dict[key] = [arr[i] for i in range(len(arr))]
    table = pa.Table.from_arrays(list(arr_dict.values()), names=list(arr_dict.keys()))
    pq.write_table(table, data_file)

movie_data = {'id': ids, 'feat': attrs, 'title': title}
write_data_parquet(movie_data, 'movie.parquet')

# process edges
edges = pandas.read_csv('u.data', delimiter='\t', header=None)
edge_data = {'src_id': edges[0], 'dst_id': edges[1], 'rate': edges[2]}
write_data_parquet(edge_data, 'edges.parquet')
