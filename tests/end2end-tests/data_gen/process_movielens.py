"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Graphstorm package.
"""
#!/usr/bin/env python

import pandas
import numpy as np
import h5py
import pyarrow.parquet as pq
import pyarrow as pa

# process user data
user = pandas.read_csv('/data/ml-100k/u.user', delimiter='|', header=None,
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

with h5py.File("/data/ml-100k/user.hdf5", "w") as f:
    ids = user['id']
    arr = f.create_dataset("id", ids.shape, dtype=ids.dtype)
    arr[:] = ids
    arr = f.create_dataset("feat", feat.shape, dtype=feat.dtype)
    arr[:] = feat

# process movie data
movie = pandas.read_csv('/data/ml-100k/u.item', delimiter='|',
        encoding="ISO-8859-1", header=None)
title = movie[1]
labels = []
for i in range(5, 24):
    labels.append(np.array(movie[i]))
labels = np.stack(labels, axis=1)

# Get the first non zero value and consider it as primary genre as there are multiple genre labels from column 5 to 23
label_list = []
for i in range(labels.shape[0]):
    label_list.append(np.nonzero(labels[i])[0][0])
labels = np.array(label_list)
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

movie_data = {'id': ids, 'label': labels, 'title': title}
write_data_parquet(movie_data, '/data/ml-100k/movie.parquet')

# process edges
edges = pandas.read_csv('/data/ml-100k/u.data', delimiter='\t', header=None)
edge_data = {'src_id': edges[0], 'dst_id': edges[1], 'rate': edges[2]}
write_data_parquet(edge_data, '/data/ml-100k/edges.parquet')