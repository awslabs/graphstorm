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
For multiple files, it contains the paths of files with a wildcard,
or a list of file paths.
* `format` specifies the input file format. This field is mandatory.
Currently, the pipeline supports two formats: parquet and JSON.
The detailed format information is specified in the format section.
* `node_id_col` specifies the column that contains the node IDs. This field is optional.
If a node type contains multiple blocks to specify the node data, only
one of the blocks require to specify the node ID column.
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
* `feature_col` specifies the column name in the input file that contains the feature.
* `feature_name` specifies the prefix of the column features. This is optional.
If `feature_name` is not provided, `feature_col` is used as the feature name.
If the feature transformation generates multiple tensors, `feature_name` becomes
the prefix of the names of the generated tensors.
* `out_dtype` specifies the data type of the transformed feature. `out_dtype` is
optional. If it is not set, no data type casting is applied to the transformed feature.
If it is set, the output feature will be cast into the corresponding data type.
Now only flaot16 and float32 are supported.
* `transform` specifies the actual feature transformation. This is a dictionary
and its `name` field indicates the feature transformation. Each transformation
has its own argument. The list of feature transformations supported by the pipeline
are listed in the section of Feature Transformation.

A label dictionary is defined:
* `task_type` specifies the task defined on the nodes or edges. The field is mandatory.
Currently, its value can be `classification`, `regression` and `link_prediction`.
* `label_col` specifies the column name in the input file that contains the label.
This has to be specified for classification and regression tasks.
`label_col` is used as the label name.
* `split_pct` specifies how to split the data into training/validation/test.
This is optional. If it's not specified, the data is split into 80% for training
10% for validation and 10% for testing.
The pipeline constructs three additional vectors indicating
the training/validation/test masks. For classification and regression tasks,
the names of the mask tensors are "train_mask", "val_mask"
and "test_mask".
* `custom_split_filenames` customizes the data split and specifies individual
nodes are used in training/validation/test sets. It specifies a dict with
file names that contains training/validation/test node IDs. The keys
of the dict are "train", "valid" and "test". A node ID in the files
is stored as a JSON object. Custom data split currently only works for
nodes.

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
			--conf-file test_data/test_data.json \
			--num-processes 2 \
			--num-parts 2 \
			--output-dir /tmp/test_out \
			--graph-name test
```

## Input formats
Currently, the graph construction pipeline supports two input formats: Parquet and JSON.

For the Parquet format, each column defines a node/edge feature, label or node/edge IDs.
For multi-dimensional features, currently the pipeline requires the features to be stored
as a list of vectors. The pipeline will reconstruct multi-dimensional features and store
them in a matrix.

For JSON format, each line of the JSON file is a JSON object. The JSON object can only
have one level. The value of each field can only be primitive values, such as integers,
strings and floating points, or a list of integers or floating points.

## Feature transformation
Currently, the graph construction pipeline only supports the following feature transformation:
* tokenize the text string with a HuggingFace tokenizer.
* compute the BERT embeddings with HuggingFace.
* compute the min-max normalization.
* convert the data into categorial values.

To tokenize text data, the `name` field in the feature transformation dictionary
is `tokenize_hf`. The dict should contain two additional fields. `bert_model`
specifies the BERT model used for tokenization. `max_seq_length` specifies
the maximal sequence length.

To compute BERT embeddings, the `name` field is `bert_hf`. The dict should
contain the following fields: `bert_model` specifies the BERT model;
`max_seq_length` specifies the maximal sequence length; `infer_batch_size`
specifies the batch size for inference.

To compute min-max normalization, the `name` field is `max_min_norm`.
The dict should contain the following fields: `max_bound` and `min_bound`
defines the upper bound and lower bound.

To convert the data to categorial values, the `name` field is `to_categorical`.
This assumes that the input data are strings.
The dict should contain the following fields: `separator` specifies how to
split the string into multiple categorical values (this is only used to define
multiple categorical values). If `separator` is not specified, the entire
string is a categorical value. `mapping` specifies how to map a string to
an integer value that defines a categorical value.

## Output
Currently, the graph construction pipeline outputs two output formats: DistDGL and DGL.
By Specifying the `output_format` as "DGL", the output will be
an [DGLGraph] (https://docs.dgl.ai/en/1.0.x/generated/dgl.save_graphs.html).
By Specifying the `output_format` as "DistDGL", the output will be a partitioned
graph named DistDGL graph. (See https://doc.dgl.ai/guide/distributed-preprocessing.html#partitioning-api for more details.)
It contains the partitioned graph, a JSON config
describing the meta-information of the partitioned graph, and the mappings for the
edges and nodes after partition which maps each node and edge in the partitoined
graph into the original node and edge id space.
The node ID mapping is stored as a dictionary of 1D tensors whose key is
the node type and value is a 1D tensor mapping between shuffled node IDs and the original node IDs.
The edge ID mapping is stored as a dictionary of 1D tensors whose key is
the edge type and value is a 1D tensor mapping between shuffled edge IDs and the original edge IDs.
