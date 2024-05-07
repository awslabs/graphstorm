.. _configurations-gconstruction:

Graph Construction
============================

`construct_graph.py <https://github.com/zhjwy9343/graphstorm/blob/main/python/graphstorm/gconstruct/construct_graph.py>`_ arguments
--------------------------------------------------------------------------------------------------------------------------------------

* **-\-conf-file**: (**Required**) the path of the configuration JSON file.
* **-\-num-processes**: the number of processes to process the data simulteneously. Default is 1. Increase this number can speed up data processing.
* **-\-num-processes-for-nodes**: the number of processes to process node data simulteneously. Increase this number can speed up node data processing.
* **-\-num-processes-for-edges**: the number of processes to process edge data simulteneously. Increase this number can speed up edge data processing.
* **-\-output-dir**: (**Required**) the path of the output data files.
* **-\-graph-name**: (**Required**) the name assigned for the graph.
* **-\-remap-node-id**: boolean value to decide whether to rename node IDs or not. Default is true.
* **-\-add-reverse-edges**: boolean value to decide whether to add reverse edges for the given graph. Default is true.
* **-\-output-format**: the format of constructed graph, options are ``DGL``,  ``DistDGL``.  Default is ``DistDGL``. It also accepts multiple graph formats at the same time separated by an space, for example ``--output-format "DGL DistDGL"``. The output format is explained in the :ref:`Output <output-format>` section below.
* **-\-num-parts**: the number of partitions of the constructed graph. This is only valid if the output format is ``DistDGL``.
* **-\-skip-nonexist-edges**: boolean value to decide whether skip edges whose endpoint nodes don't exist. Default is true.
* **-\-ext-mem-workspace**: the directory where the tool can store data during graph construction. Suggest to use high-speed SSD as the external memory workspace.
* **-\-ext-mem-feat-size**: the minimal number of feature dimensions that features can be stored in external memory. Default is 64.

.. _gconstruction-json:

Configuration JSON Explanations
---------------------------------

The JSON file that describes the graph data defines where to get node data and edge data to construct a graph. Below shows an example of such a JSON file. In the highest level, it contains three fields: ``version``, ``nodes`` and ``edges``.

``version``
...........
``version`` marks the version of the configuration file schema, allowing its identification to be self-contained for downstream applications. The current (and expected) version is ``gconstruct-v0.1``.

``nodes``
...........
``nodes`` contains a list of node types and the information of a node type is stored in a dictionary. A node dictionary contains multiple fields and most fields are optional.

* ``node_type``: (**Required**) specifies the node type. Think this as a name given to one type of nodes, e.g. `author` and `paper`.
* ``files``: (**Required**) specifies the input files for the node data. There are multiple options to specify the input files. For a single input file, it contains the path of a single file. For multiple files, it contains the paths of files with a wildcard, or a list of file paths, e.g., `file_name*.parquet`.
* ``format``: (**Required**) specifies the input file format. Currently, the pipeline supports three formats: ``parquet``, ``HDF5``, and ``JSON``. The value of this field is a dictionary, where the key is ``name`` and the value is either ``parquet`` or ``JSON``, e.g., `{"name":"JSON"}`. The detailed format information is specified in the format section.
* ``node_id_col``: specifies the column name that contains the node IDs. This field is optional. If a node type contains multiple blocks to specify the node data, only one of the blocks require to specify the node ID column.
* ``features`` is a list of dictionaries that define how to get features and transform features. This is optional. The format of a feature dictionary is defined :ref:`below <feat-format>`.
* ``labels`` is a list of dictionaries that define where to get labels and how to split the data into training/validation/test set. This is optional. The format of a label dictionary is defined :ref:`below<label-format>`.

``edges``
...........
Similarly, ``edges`` contains a list of edge types and the information of an edge type is stored in a dictionary. An edge dictionary also contains the same fields of ``files``, ``format``, ``features`` and ``labels`` as ``nodes``. In addition, it contains the following fields:

* ``source_id_col``: (**Required**) specifies the column name of the source node IDs.
* ``dest_id_col``: (**Required**) specifies the column name of the destination node IDs.
* ``relation``: (**Required**) is a list of three elements that contains the node type of the source nodes, the relation type of the edges and the node type of the destination nodes. Values of node types should be same as the corresponding values specified in the ``node_type`` fields in ``nodes`` objects, e.g., `["author", "write", "paper"]`.

.. _feat-format:

**A feature dictionary is defined:**

* ``feature_col``: (**Required**) specifies the column name in the input file that contains the feature. The ``feature_col`` can accept either a string or a list. When ``feature_col`` is specified as a list with multiple columns, the same feature transformation operation will be applied to each column, and then the transformed feature will be concatenated to form the final feature.
* ``feature_name``: specifies the prefix of the column feature name. This is optional. If feature_name is not provided, ``feature_col`` is used as the feature name. If the feature transformation generates multiple tensors, ``feature_name`` becomes the prefix of the names of the generated tensors. If there are multiple columns defined in ``feature_col``, ``feature_name`` is required.
* ``out_dtype`` specifies the data type of the transformed feature. ``out_dtype`` is optional. If it is not set, no data type casting is applied to the transformed feature. If it is set, the output feature will be cast into the corresponding data type. Now only `float16`, `float32`, and `float64` are supported.
* ``transform``: specifies the actual feature transformation. This is a dictionary and its name field indicates the feature transformation. Each transformation has its own argument. The list of feature transformations supported by the pipeline are listed in the section of :ref:`Feature Transformation <feat-transform>` below.

.. _label-format:

**A label dictionary is defined:**

* ``task_type``: (**Required**) specifies the task defined on the nodes or edges. Currently, its value can be ``classification``, ``regression`` and ``link_prediction``.
* ``label_col``: specifies the column name in the input file that contains the label. This has to be specified for ``classification`` and ``regression`` tasks. ``label_col`` is used as the label name.
* ``split_pct``: (Optional) specifies how to split the data into training/validation/test. If it's not specified, the data is split into 80% for training 10% for validation and 10% for testing. The pipeline constructs three additional vectors indicating the training/validation/test masks. For ``classification`` and ``regression`` tasks, the names of the mask tensors are ``train_mask``, ``val_mask`` and ``test_mask``.
* ``custom_split_filenames``: (Optional) specifies the customized training/validation/test mask. It has field named ``train``, ``valid``, and ``test`` to specify the path of the mask files. It is possible that one of the subfield here leaves empty and it will be treated as none. It will override the ``split_pct`` once provided. Refer to :ref:`Use Your Own Graphs Tutorial <use-own-data>` for an example.

.. _input-format:

Input formats
..............
Currently, the graph construction pipeline supports three input formats: ``Parquet``, ``HDF5``, and ``JSON``.

For the Parquet format, each column defines a node/edge feature, label or node/edge IDs. For multi-dimensional features, currently the pipeline requires the features to be stored as a list of vectors. The pipeline will reconstruct multi-dimensional features and store them in a matrix.

The HDF5 format is similar as the parquet format, but have larger capacity. Therefore suggest to use HDF5 format if users' data is large.

For JSON format, each line of the JSON file is a JSON object. The JSON object can only have one level. The value of each field can only be primitive values, such as integers, strings and floating points, or a list of integers or floating points.

.. _feat-transform:

Feature transformation
.........................
Currently, the graph construction pipeline supports the following feature transformation:

* **HuggingFace tokenizer transformation** tokenizes text strings with a HuggingFace tokenizer. The ``name`` field in the feature transformation dictionary is ``tokenize_hf``. The dict should contain two additional fields. ``bert_model`` specifies the LM model used for tokenization. Users can choose any `HuggingFace LM models <https://huggingface.co/models>`_ from one of the following types: ``"bert", "roberta", "albert", "camembert", "ernie", "ibert", "luke", "mega", "mpnet", "nezha", "qdqbert","roc_bert"``. ``max_seq_length`` specifies the maximal sequence length.
* **HuggingFace LM transformation** encodes text strings with a HuggingFace LM model.  The ``name`` field in the feature transformation dictionary is ``bert_hf``. The dict should contain two additional fields. ``bert_model`` specifies the LM model used for embedding text. Users can choose any `HuggingFace LM models <https://huggingface.co/models>`_ from one of the following types: ``"bert", "roberta", "albert", "camembert", "ernie", "ibert", "luke", "mega", "mpnet", "nezha", "qdqbert","roc_bert"``. ``max_seq_length`` specifies the maximal sequence length.
* **Numerical MAX_MIN transformation** normalizes numerical input features with `val = (val-min)/(max-min)`, where `val` is the feature value, `max` is the maximum number in the feature and `min` is the minimum number in the feature. The ``name`` field in the feature transformation dictionary is ``max_min_norm``. The dict can contain four optional fields: ``max_bound``, ``min_bound``, ``max_val`` and ``min_val``. ``max_bound`` specifies the maximum value allowed in the feature. Any number larger than ``max_bound`` will be set to ``max_bound``. Here, `max` = min(np.amax(feats), ``max_bound``). ``min_bound`` specifies the minimum value allowed in the feature. Any number smaller than ``min_bound`` will be set to ``min_bound``. Here, `min` = max(np.amin(feats), ``min_bound``). ``max_val`` defines the `max` in the transformation formula. When ``max_val`` is provided, `max` is always equal to ``max_val``. ``min_val`` defines the `min` in the transformation formula.  When ``min_val`` is provided, `min` is always equal to ``min_val``. ``max_val`` and ``min_val`` are mainly used in the inference stage, where we want to use the max & min values computed in the training stage to normalize inference data.
* **Numerical Rank Gauss transformation** normalizes numerical input features with rank gauss normalization. It maps the numeric feature values to gaussian distribution based on ranking. The method follows https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927. The ``name`` field in the feature transformation dictionary is ``rank_gauss``. The dict can contains one optional field, i.e., ``epsilon`` which is used to avoid INF float during computation.
* **Convert to categorical values** converts text data to categorial values. The `name` field is `to_categorical`. `separator` specifies how to split the string into multiple categorical values (this is only used to define multiple categorical values). If `separator` is not specified, the entire string is a categorical value. `mapping` is a dict that specifies how to map a string to an integer value that defines a categorical value.
* **Numerical Bucket transformation** normalizes numerical input features with buckets. The input features are divided into one or multiple buckets. Each bucket stands for a range of floats. An input value can fall into one or more buckets depending on the transformation configuration. The ``name`` field in the feature transformation dictionary is ``bucket_numerical``. Users need to provide ``range`` and ``bucket_cnt`` field, which ``range`` defines a numerical range, and ``bucket_cnt`` defines number of buckets among the range. All buckets will have same length, and each of them is left included. e.g, bucket ``(a, b)`` will include a, but not b. All input feature column data are categorized into respective buckets using this method. Any input data lower than the minimum value will be assigned to the first bucket, and any input data exceeding the maximum value will be assigned to the last bucket. For example, with range=`[10,30]` and bucket_cnt=`2`, input data `1` will fall into the bucket `[10, 20]`, input data `11` will be mapped to `[10, 20]`, input data `21` will be mapped to `[20, 30]`, input data `31` will be mapped to `[20, 30]`. Finally we use one-hot-encoding to encode the feature for each numerical bucket. If a user wants to make numeric values fall into more than one bucket, it is preferred to use the `slide_window_size`: `"slide_window_size": s` , where `s` is a number. Then each value `v` will be transformed into a range from `v - s/2` through `v + s/2` , and assigns the value `v` to every bucket that the range covers.

.. _output-format:

Output
..........
Currently, the graph construction pipeline outputs two output formats: ``DistDGL`` and ``DGL``. If select ``DGL``, the output is a file, named `<graph_name>.dgl` under the folder specified by the **-\-output-dir** argument, where `<graph_name>` is the value of argument **-\-graph-name**. If select ``DistDGL``, the output is a JSON file, named `<graph_name>.json`, and a set of `part*` folders under the folder specified by the **-\-output-dir** argument, where the `*` is the number specified by the **-\-num-parts** argument.

By Specifying the output_format as ``DGL``, the output will be an `DGLGraph <https://docs.dgl.ai/en/1.0.x/generated/dgl.save_graphs.html>`_. By Specifying the output_format as ``DistDGL``, the output will be a partitioned graph named `DistDGL graph <https://doc.dgl.ai/guide/distributed-preprocessing.html#partitioning-api>`_. It contains the partitioned graph, a JSON config describing the meta-information of the partitioned graph, and the mappings for the edges and nodes after partition, ``node_mapping.pt`` and ``edge_mapping.pt``, which maps each node and edge in the partitoined graph into the original node and edge id space. The node ID mapping is stored as a dictionary of 1D tensors whose key is the node type and value is a 1D tensor mapping between shuffled node IDs and the original node IDs. The edge ID mapping is stored as a dictionary of 1D tensors whose key is the edge type and value is a 1D tensor mapping between shuffled edge IDs and the original edge IDs.

.. note:: The two mapping files are used to record the mapping between the ogriginal node and edge ids in the raw data files and the ids of nodes and edges in the constructed graph. They are important for mapping the training and inference outputs. Therefore, DO NOT move or delete them.

An example
............
Below shows an example that contains one node type and an edge type. For a real example, please refer to the :ref:`input JSON file <input-config>` used in the :ref:`Use Your Own Graphs Tutorial <use-own-data>`.

.. code-block:: json

    {
        "version": "gconstruct-v0.1",
        "nodes": [
            {
                "node_id_col":  "paper_id",
                "node_type":    "paper",
                "format":       {"name": "parquet"},
                "files":        "/tmp/dummy/paper_nodes*.parquet",
                "features":     [
                    {
                        "feature_col":  ["paper_title"],
                        "feature_name": "title",
                        "transform":    {"name": "tokenize_hf",
                                         "bert": "huggingface-basic",
                                         "max_seq_length": 512}
                    },
                ],
                "labels":       [
                    {
                        "label_col":    "labels",
                        "task_type":    "classification",
                        "split_pct":   [0.8, 0.2, 0.0],
                    },
                ],
            }
        ],
        "edges": [
            {
                "source_id_col":    "src_paper_id",
                "dest_id_col":      "dest_paper_id",
                "relation":         ["paper", "cite", "paer"],
                "format":           {"name": "parquet"},
                "files":            ["/tmp/edge_feat.parquet"],
                "features":         [
                    {
                        "feature_col":  ["citation_time"],
                        "feature_name": "feat",
                    },
                ]
            }
        ]
    }
