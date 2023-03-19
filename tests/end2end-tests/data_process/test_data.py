import os
import dgl
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

out_dir = '/tmp/test_out/'
conf_file = '/tmp/test_data/test_data_transform.conf'

def read_data_parquet(data_file):
    table = pq.read_table(data_file)
    pd = table.to_pandas()
    return {key: np.array(pd[key]) for key in pd}
    
g = dgl.load_graphs(os.path.join(out_dir, "test.dgl"))[0][0]
node1_map = read_data_parquet(os.path.join(out_dir, "node1_id_remap.parquet"))
reverse_node1_map = {val: key for key, val in zip(node1_map['orig'], node1_map['new'])}
node2_map = read_data_parquet(os.path.join(out_dir, "node2_id_remap.parquet"))
reverse_node2_map = {val: key for key, val in zip(node2_map['orig'], node2_map['new'])}

# Test the first node data
data = g.nodes['node1'].data['feat'].numpy()
label = g.nodes['node1'].data['label'].numpy()
orig_ids = np.array([reverse_node1_map[new_id] for new_id in range(g.number_of_nodes('node1'))])
assert np.all(data == orig_ids)
assert np.all(label == orig_ids % 100)

# Test the second node data
data = g.nodes['node2'].data['feat'].numpy()
orig_ids = np.array([reverse_node2_map[new_id] for new_id in range(g.number_of_nodes('node2'))])
assert data.shape[1] == 5
for i in range(data.shape[1]):
    assert np.all(data[:,i] == orig_ids)

# Test the edge data
