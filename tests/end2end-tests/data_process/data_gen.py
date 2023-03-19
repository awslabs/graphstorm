import copy
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

def write_data_parquet(data, data_file):
    df = {}
    for key in data:
        arr = data[key]
        assert len(arr.shape) == 1 or len(arr.shape) == 2
        if len(arr.shape) == 1:
            df[key] = arr
        else:
            for i in range(arr.shape[1]):
                df[key + "@" + str(i)] = arr[:,i]
    table = pa.Table.from_arrays(list(df.values()), names=list(df.keys()))
    pq.write_table(table, data_file)

node_data1 = {
    'id': np.random.randint(0, 1000000000, 10000),
    'data': np.random.randint(0, 1000000000, 10000),
}
node_data1['data_id'] = copy.deepcopy(node_data1['id'])

node_data2 = {
    'id': np.random.randint(0, 1000000000, 20000),
    'data': np.random.random((20000, 5)),
}
node_data2['data_id'] = copy.deepcopy(node_data2['id'])

edge_data1 = {
    'src': node_data1['id'][np.random.randint(0, 10000, 100000)],
    'dst': node_data2['id'][np.random.randint(0, 20000, 100000)],
}
edge_data2 = {
    'src': node_data1['id'][np.random.randint(0, 10000, 50000)],
    'dst': node_data1['id'][np.random.randint(0, 10000, 50000)],
}
edge_data3 = {
    'src': node_data2['id'][np.random.randint(0, 20000, 100000)],
    'dst': node_data2['id'][np.random.randint(0, 20000, 100000)],
}

in_dir = '/tmp/test_data/'
out_dir = '/tmp/test_out/'

os.makedirs(in_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def split_data(data, num):
    new_data_list = [{} for _ in range(num)]
    for key, val in data.items():
        for i, val in enumerate(np.split(val, num)):
            new_data_list[i][key] = val
    return new_data_list

for i, node_data in enumerate(split_data(node_data1, 5)):
    write_data_parquet(node_data, os.path.join(in_dir, f'node_data1_{i}.parquet'))
for i, node_data in enumerate(split_data(node_data2, 10)):
    write_data_parquet(node_data, os.path.join(in_dir, f'node_data2_{i}.parquet'))
for i, edge_data in enumerate(split_data(edge_data1, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data1_{i}.parquet'))
for i, edge_data in enumerate(split_data(edge_data2, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data2_{i}.parquet'))
for i, edge_data in enumerate(split_data(edge_data3, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data3_{i}.parquet'))
node_id1 = {'id': np.sort(node_data1['id'])}
node_id2 = {'id': np.sort(node_data2['id'])}
write_data_parquet(node_id1, os.path.join(in_dir, 'node_id1.parquet'))
write_data_parquet(node_id2, os.path.join(in_dir, 'node_id2.parquet'))

transform_conf = [
    {
        "in_format": {"name": "parquet"},
        "out_format": {"name": "numpy"},
        "type": "node_data",
        "name": "node_data1",
        "in_files": os.path.join(in_dir, "node_data1_*.parquet"),
        "ops": {
            "id": { "op": "create_id_map", "map_file": os.path.join(in_dir, "node_id1.npy") },
            "data": { "op": "identity", },
            "data_id" : { "op": "identity", },
        }
    },
    {
        "in_format": {"name": "parquet"},
        "out_format": {"name": "numpy"},
        "type": "node_data",
        "name": "node_data2",
        "in_files": os.path.join(in_dir, "node_data2_*.parquet"),
        "ops": {
            "id": { "op": "create_id_map", "map_file": os.path.join(in_dir, "node_id2.npy") },
            "data_id": { "op": "identity", }
        }
    },
    {
        "in_format":{"name":  "parquet"},
        "out_format": {"name": "csv", "delimiter": ",", "order": "src,dst", "format": "%d"},
        "type": "edge",
        "name": "node_data1:edge_data1:node_data2",
        "in_files": os.path.join(in_dir, "edge_data1_*.parquet"),
        "ops": {
            "src": { "op": "map_id", "map_file": os.path.join(in_dir, "node_id1.npy") },
            "dst": { "op": "map_id", "map_file": os.path.join(in_dir, "node_id2.npy") },
        }
    },
    {
        "in_format": {"name": "parquet"},
        "out_format": {"name": "csv", "delimiter": ",", "order": "src,dst", "format": "%d"},
        "type": "edge",
        "name": "node_data1:edge_data2:node_data1",
        "in_files": os.path.join(in_dir, "edge_data2_*.parquet"),
        "ops": {
            "src": { "op": "map_id", "map_file": os.path.join(in_dir, "node_id1.npy") },
            "dst": { "op": "map_id", "map_file": os.path.join(in_dir, "node_id1.npy") },
        }
    },
    {
        "in_format": {"name": "parquet"},
        "out_format": {"name": "csv", "delimiter": ",", "order": "src,dst", "format": "%d"},
        "type": "edge",
        "name": "node_data2:edge_data3:node_data2",
        "in_files": os.path.join(in_dir, "edge_data3_*.parquet"),
        "ops": {
            "src": { "op": "map_id", "map_file": os.path.join(in_dir, "node_id2.npy") },
            "dst": { "op": "map_id", "map_file": os.path.join(in_dir, "node_id2.npy") },
        }
    },
]
json.dump(transform_conf, open(os.path.join(in_dir, 'test_data_transform.conf'), 'w'), indent=4)
