import os
import dgl
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch as th

out_dir = '/tmp/test_out/'
conf_file = '/tmp/test_data/test_data_transform.conf'

def read_data_parquet(data_file):
    table = pq.read_table(data_file)
    pd = table.to_pandas()
    return {key: np.array(pd[key]) for key in pd}
    
g = dgl.load_graphs(os.path.join(out_dir, "test.dgl"))[0][0]
node1_map = read_data_parquet(os.path.join(out_dir, "node1_id_remap.parquet"))
reverse_node1_map = {val: key for key, val in zip(node1_map['orig'], node1_map['new'])}

# Test the first node data
data = g.nodes['node1'].data['feat'].numpy()
label = g.nodes['node1'].data['label'].numpy()
assert label.dtype == np.int32
orig_ids = np.array([reverse_node1_map[new_id] for new_id in range(g.number_of_nodes('node1'))])
assert np.all(data == orig_ids)
assert np.all(label == orig_ids % 100)
assert th.sum(g.nodes['node1'].data['train_mask']) == int(g.number_of_nodes('node1') * 0.8)
assert th.sum(g.nodes['node1'].data['val_mask']) == int(g.number_of_nodes('node1') * 0.2)
assert th.sum(g.nodes['node1'].data['test_mask']) == 0

# Test the second node data
data = g.nodes['node2'].data['feat'].numpy()
orig_ids = np.arange(g.number_of_nodes('node2'))
assert data.shape[1] == 5
for i in range(data.shape[1]):
    assert np.all(data[:,i] == orig_ids)

# Test the edge data
src_ids, dst_ids = g.edges(etype=('node1', 'relation1', 'node2'))
label = g.edges[('node1', 'relation1', 'node2')].data['label'].numpy()
assert label.dtype == np.int32
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['train_mask']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.8)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['val_mask']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.2)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['test_mask']) == 0
