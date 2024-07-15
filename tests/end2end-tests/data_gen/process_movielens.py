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
import os
import pandas
import numpy as np
import h5py
import pyarrow.parquet as pq
import pyarrow as pa

from graphstorm.gconstruct.id_map import IdMap

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
num_users = feat.shape[0]

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

# Get the first non zero value and consider it as primary genre as there are multiple genre labels
# from column 5 to 23
label_list = []
for i in range(labels.shape[0]):
    label_list.append(np.nonzero(labels[i])[0][0])
labels = np.array(label_list)
ids = np.array(movie[0])

user_ids = np.arange(num_users).astype(str)
user_id_map = IdMap(user_ids)
map_prefix = os.path.join("/data/ml-100k/", "raw_id_mappings", "user")
user_id_map.save(map_prefix)
movie_id_map = IdMap(ids.astype(str))
map_prefix = os.path.join("/data/ml-100k/", "raw_id_mappings", "movie")
movie_id_map.save(map_prefix)

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

user_data = {'id': user['id'], 'feat': feat, 'occupation': user['occupation']}
write_data_parquet(user_data, '/data/ml-100k/users.parquet')

movie_data = {'id': ids,
              'title': title,
              'label': labels,
              'label2': labels } # label2 for multi-task learning test
write_data_parquet(movie_data, '/data/ml-100k/movie.parquet')

# process edges
edges = pandas.read_csv('/data/ml-100k/u.data', delimiter='\t', header=None)
# Set the rate to start from 0 to fit evaluation metrics, e.g., roc_auc or p_r
edge_data = {'src_id': edges[0],
             'dst_id': edges[1],
             'rate': edges[2]-1,
             'rate_class': edges[2]} # rate_class for multi-task learning test
write_data_parquet(edge_data, '/data/ml-100k/edges.parquet')

# generate data for homogeneous optimization test
edges = pandas.read_csv('/data/ml-100k/u.data', delimiter='\t', header=None)
# Set rate to start from 0 to fit evaluation metrics, e.g., roc_auc or p_r
edge_data = {'src_id': edges[1], 'dst_id': edges[1], 'rate': edges[2]-1}
write_data_parquet(edge_data, '/data/ml-100k/edges_homogeneous.parquet')

# generate hard negatives
num_movies = len(ids)
neg_movie_idx = np.random.randint(0, num_movies, (edges.shape[0], 5))
neg_movie_0 = ids[neg_movie_idx]
neg_movie_1 = []
for idx, neg_movie in enumerate(neg_movie_0):
    if idx < 10:
        neg_movie_1.append(list(neg_movie.astype(str))[0])
    else:
        neg_movie_1.append(",".join(list(neg_movie.astype(str))))
neg_movie_1 = np.array(neg_movie_1)
neg_movie_idx = np.random.randint(0, num_movies, (edges.shape[0], 10))
neg_movie_2 = ids[neg_movie_idx]

neg_edge_data = {
    "hard_0": neg_movie_0,
    "hard_1": neg_movie_1,
    "fixed_eval": neg_movie_2
}
write_data_parquet(neg_edge_data, '/data/ml-100k/hard_neg.parquet')

# generate synthetic user data with label
user_labels = np.random.randint(11, size=feat.shape[0])
user_data = {'id': user['id'].values, 'feat': feat, 'occupation': user['occupation'], 'label': user_labels}
write_data_parquet(user_data, '/data/ml-100k/users_with_synthetic_labels.parquet')