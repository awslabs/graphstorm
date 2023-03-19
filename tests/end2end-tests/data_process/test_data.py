import glob
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

out_dir = '/tmp/test_out/'
conf_file = '/tmp/test_data/test_data_transform.conf'

def read_data_parquet(data_file):
    table = pq.read_table(data_file)
    pd = table.to_pandas()
    return {key: np.array(pd[key]) for key in pd}
    
transform_conf = json.load(open(conf_file, 'r'))

# Test the first node data
in_files = glob.glob(transform_conf[0]["in_files"])
out_prefix = os.path.join(out_dir, transform_conf[0]['type'] + '-' + transform_conf[0]['name'])
in_files.sort()
for feat_name in ['data', 'data_id']:
    out_files = glob.glob(out_prefix + f'-*{feat_name}.npy')
    out_files.sort()
    assert len(in_files) == len(out_files)
    for in_file, out_file in zip(in_files, out_files):
        node_data = read_data_parquet(in_file)
        data = np.load(out_file)
        assert np.all(node_data[feat_name] == data)

# Test the second node data
in_files = glob.glob(transform_conf[1]["in_files"])
out_prefix = os.path.join(out_dir, transform_conf[1]['type'] + '-' + transform_conf[1]['name'])
in_files.sort()
for feat_name in ['data_id']:
    out_files = glob.glob(out_prefix + f'-*{feat_name}.npy')
    out_files.sort()
    assert len(in_files) == len(out_files)
    for in_file, out_file in zip(in_files, out_files):
        node_data = read_data_parquet(in_file)
        data = np.load(out_file)
        assert np.all(node_data[feat_name] == data)

# Test the edge data
for i in range(2, 5):
    in_files = glob.glob(transform_conf[i]["in_files"])
    in_files.sort()
    output_prefix = os.path.join(out_dir,
                                 transform_conf[i]['type'] + '-' +
                                 transform_conf[i]['name'].replace(':', '_'))
    out_files = glob.glob(output_prefix + '-*.csv')
    out_files.sort()
    assert len(in_files) == len(out_files)
    src_id_map = np.load(transform_conf[i]['ops']['src']['map_file'])
    dst_id_map = np.load(transform_conf[i]['ops']['dst']['map_file'])
    for in_file, out_file in zip(in_files, out_files):
        in_edge_data = read_data_parquet(in_file)
        out_edge_data = np.loadtxt(out_file, dtype=np.int64, delimiter=',')
        assert np.all(src_id_map[out_edge_data[:,0]] == in_edge_data['src'])
        assert np.all(dst_id_map[out_edge_data[:,1]] == in_edge_data['dst'])
