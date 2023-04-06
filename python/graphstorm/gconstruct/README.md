To load a user-defined graph to GraphStorm, a user needs to format their data as follows.
Specifially, each node/edge type should be stored in separate folders. The data associated
with a node/edge type can be stored in multiple files. Below shows an example of the graph
data with two node types and two edge types. In this example, the node/edge data are stored
in the parquet format. We support three input formats: parquet, CSV and JSON.
```
data_root_dir/
  |-- input_data.json
  |-- node1/
  |   |-- node_1_1.parquet
  |   |-- node_1_2.parquet
  |-- node2/
  |   |-- node_2_1.parquet
  |   |-- node_2_2.parquet
  |-- edge1/
  |   |-- edge_1_1.parquet
  |   |-- edge_1_2.parquet
  |-- edge2/
  |   |-- edge_2_1.parquet
  |   |-- edge_2_2.parquet
```

The graph data folder should contains a JSON file that describes the graph data.
It defines where to get node data
and edge data to construct a graph. "nodes" contains a list of node types and
it contains a blob for each node type. Similarly, "edges" contains a list of
edge types and each blob defines an edge type.
Inside a blob, it contains the "features" field that defines where to get
node/edge features and how to transform features if specified. It contains
the "labels" field that defines where to get node/edge labels and how
to split nodes/edges into training/validation/test set if specified.
Below shows an example that contains one node type and an edge type.
```
{
	nodes: [
		{
			"node_id_col":  "<column name>",
			"node_type":    "<node type>",
			"format":       {"name": "csv", "separator": ","},
			"files":        ["<paths to files>", ...],
			"features":     [
				{
					"feature_col":  ["<column name>", ...],
					"feature_name": "<feature name>",
					"data_type":    "<feature data type>",
					"transform":    {"name": "<operator name>", ...}
				},
			],
			"labels":       [
				{
					"label_col":    "<column name>",
					"task_type":    "<task type: e.g., classification>",
					"split_type":   [0.8, 0.2, 0.0],
				},
			],
		}
	],
	edges: [
		{
			"source_id_col":    "<column name>",
			"dest_id_col":      "<column name>",
			"relation":         "<src type, relation type, dest type>",
			"format":           {"name": "csv", "separator": ","},
			"files":            ["<paths to files>", ...],
			"features":         [
				{
					"feature_col":  ["<column name>", ...],
					"feature_name": "<feature name>",
					"data_type":    "<feature data type>",
					"transform":    {"name": "<operator name>", ...}
				},
			],
			"labels":           [
				{
					"label_col":    "<column name>",
					"task_type":    "<task type: e.g., classification>",
					"split_type":   [0.8, 0.2, 0.0],
				},
			],
		}
	]
}
```

GraphStorm contains a script `construct_graph.py` that constructs a graph
from the user's input data with the format described above. The script will save
the constructed graph in a file with the DGL or DistDGL graph format. If users' data have
string node IDs or the node IDs in the node files are not stored in an ascending order
starting from 0, `construct_graph.py` will remap the node IDs and save the ID map
into files. The node ID map of each node type is saved in separate files.

The command line below shows an example of how to use `construct_graph.py` to
construct a graph and save it in DistDGL graph format directly.
```
python3 -m graphstorm.gconstruct.construct_graph \
			--conf_file test_data/test_data.json \
			--num_processes 2 \
			--output_dir /tmp/test_out \
			--graph_name test
```
