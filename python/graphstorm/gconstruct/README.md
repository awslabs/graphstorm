To load a user-defined graph to GraphStorm, a user needs to provide a JSON file to describe
their data. Users need to store the node/edge data of different types in separate files.
The data associated
with a node/edge type can be stored in multiple files. Below shows an example of the graph
data with two node types and two edge types. In this example, the node/edge data are stored
in the parquet format.

The JSON file that describes the graph data defines where to get node data
and edge data to construct a graph. Below shows an example of such a JSON file.
In the highest level, it contains two fields: `nodes` and `edges`.

`nodes` contains a list of node types and the information of a node type
is stored in a dictionary. A node dictionary contains multiple fields and
most fields are optional:
* `node_type` specifies the node type. This field is mandatory.
* `files` specifies the input files for the node data. This field is mandatory.
There are multiple options to specify the input files.
For a single input file, it contains the path of a single file.
For multiple files, it contains the path of a directory
(all files in the directory are considered as the input files),
the path of files with a wildcard, or a list of file paths.
* `node_id_col` specifies the column that contains the node IDs. This field is optional.
* `format` specifies the input file format. This field is optional.
If this is not provided, the input file format is determined by the extension name of the input files.
Currently, the pipeline supports two formats: parquet and JSON.
The detailed format information is specified in the format section.
* `features` is a list of dictionaries that define how to get features
and transform features. This is optional. The format of a feature directionary
is defined below.
* `labels` is a list of dictionaries that define where to get labels
and how to split the data into training/validation/test set. This is optional.
The format of a label directionary is defined below.

Similarly, `edges` contains a list of edge types and the information of
an edge type is stored in a dictionary. An edge dictionary also contains
the same fields of `files`, `format`, `features` and `labels` as nodes.
In addition, it contains the following fields:
* `source_id_col` specifies the column name of the source node IDs.
* `dest_id_col` specifies the column name of the destination node IDs.
* `relation` is a list of three elements that contains the node type of
the source nodes, the relation type of the edges and the node type of
the destination nodes.

A feature dictionary is defined:
* 

A label dictionary is defined:
*

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
					"split_pct":   [0.8, 0.2, 0.0],
				},
			],
		}
	],
	edges: [
		{
			"source_id_col":    "<column name>",
			"dest_id_col":      "<column name>",
			"relation":         ["<src node type>", "<relation type>", "<dest node type>"],
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
					"split_pct":   [0.8, 0.2, 0.0],
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

## Input formats

## Feature transformation

