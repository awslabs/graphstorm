.. _single-machine-gconstruction:

Single Machine Graph Construction
-----------------------------------

Prerequisites
**************

1. A machine with Linux operation system that has proper CPU memory according to the raw data size.
2. Following the :ref:`Setup GraphStorm with pip Packages <setup_pip>` guideline to install GraphStorm and its dependencies.
3. Following the :ref:`Input Raw Data Explanations <input_raw_data>` guideline to prepare the input raw data.

Graph consturction command
****************************

GraphStorm provides a ``gconstruct.construct_graph`` module for graph construction in a signle machine. Users can run the ``gconstruct.construct_graph`` command by following the command template below.

.. code:: python

    python -m graphstorm.gconstruct.construct_graph \
          --conf-file config.json \
          --output-dir /a_path \
          --num-parts 1 \
          --graph-name a_name

This template provides the actual Python command, and it also indicates the three required command arguments, i.e., ``--conf-file`` specifies a JSON file containing graph construction configurations, ``--output-dir`` specifies the directory for outputs, and ``--graph-name`` specifies a string as a name given to the constructed graph. The ``--num-parts`` whose default given value is ``1`` is also an important argument. It determines how many partitions to be constructed. In distrusted model training and inference, the number of machines is determined by the number of partitions.

.. _gconstruction-json:

Configuration JSON Object Explanations
**************************************

The configuration JSON file is the **key** input argument for graph construction. The file contains a JSON object that defines the overall graph schema in terms of node type and edge type. For each node and edge type, it defines where the node and edge data are stored and in what file format. When a type of node or edge has features, it defines which columns in the data table are features and what feature transformation operations will be used to encode the features. When a type of node or edge has labels, it defines which columns in the data table are labels and how to split the labels into the training, validation, and testing sets.

In the highest level, the JSON object contains three fields: ``version``, ``nodes`` and ``edges``.

``version`` (**Optional**)
..........................
``version`` marks the version of the configuration file schema, allowing its identification to be self-contained for downstream applications. The current (and expected) version is ``gconstruct-v0.1``.

``nodes`` (**Required**)
........................
``nodes`` contains a list of node types and the information of a node type is stored in a dictionary. A node dictionary contains multiple fields and most fields are optional.

* ``node_type``: (**Required**) specifies the node type. Think this as a name given to one type of nodes, e.g. `"author"` and `"paper"`.
* ``files``: (**Required**) specifies the input files for the node type. There are multiple options to specify the input files. For a single input file, it contains the path of a single file. For multiple files, it could contain the paths of files with a wildcard, e.g., `file_name*.parquet`, or a list of file paths, e.g., `["file_name001.parquet", "file_name002.parquet", ...]`.
* ``format``: (**Required**) specifies the input file format. Currently, the construction command supports three input file formats: ``csv``, ``parquet``, and ``HDF5``. The value of this field is a dictionary, where the key is ``name`` and the value is either ``csv``, ``parquet`` or ``HDF5``, e.g., `{"name":"csv"}`. The detailed format information could be found in the :ref:`Input Raw Data Explanations <input_raw_data>` guideline.
* ``node_id_col``: specifies the column name that contains the node IDs. This field is optional. If not provided, the construction command will create node IDs according to the total number of rows and consider each row in the node table is a unique node. If user choose to store columns of a node type in multiple sets of tables, only one of the set of tables require to specify the node ID column. For example of this multiple sets of tables, please refer to :ref:`the simple input data example <multi-set-table-examle>` document.
* ``features`` is a list of dictionaries that define how to get features and transform features. This is optional. The format of a feature dictionary is defined in the :ref:`Feature dictionary format <feat-format>` section below.
* ``labels`` is a list of dictionaries that define where to get labels and how to split the labels into training/validation/test set. This is optional. The format of a label dictionary is defined in the :ref:`Label dictionary format <label-format>` section below.

``edges`` (**Required**)
........................
Similarly, ``edges`` contains a list of edge types and the information of an edge type is stored in a dictionary. An edge dictionary also contains the same fields of ``files``, ``format``, ``features`` and ``labels`` as the ``nodes`` field. In addition, it contains the following unique fields:

* ``source_id_col``: (**Required**) specifies the column name of the source node IDs.
* ``dest_id_col``: (**Required**) specifies the column name of the destination node IDs.
* ``relation``: (**Required**) is a list of three elements that contains the node type of the source nodes, the relation type of the edges, and the node type of the destination nodes. Values of node types should be same as the corresponding values specified in the ``node_type`` fields in ``nodes`` objects, e.g., `["author", "write", "paper"]`.

.. _feat-format:

**Feature dictionary format**

* ``feature_col``: (**Required**) specifies the column name in the input file that contains the feature. The ``feature_col`` can accept either a string or a list. When ``feature_col`` is specified as a list with multiple columns, the same feature transformation operation will be applied to each column, and then the transformed feature will be concatenated to form the final feature.
* ``feature_name``: specifies the prefix of the column feature name. This is optional. If feature_name is not provided, ``feature_col`` is used as the feature name. If the feature transformation generates multiple tensors, ``feature_name`` becomes the prefix of the names of the generated tensors. If there are multiple columns defined in ``feature_col``, ``feature_name`` is required.
* ``out_dtype`` specifies the data type of the transformed feature. ``out_dtype`` is optional. If it is not set, no data type casting is applied to the transformed feature. If it is set, the output feature will be cast into the corresponding data type. Now only `float16`, `float32`, and `float64` are supported.
* ``transform``: specifies the actual feature transformation. This is a dictionary and its name field indicates the feature transformation operation. Each transformation operation has its own argument(s). The list of feature transformations supported by the pipeline are listed in the section of :ref:`Feature Transformation <feat-transform>` below.

.. _label-format:

**Label dictionary format**

* ``task_type``: (**Required**) specifies the task defined on the nodes or edges. Currently, its value can be one of ``classification``, ``regression``, ``link_prediction``, and ``reconstruct_node_feat``.
* ``label_col``: specifies the column name in the input file that contains the labels. This has to be specified for ``classification`` and ``regression`` tasks. ``label_col`` is also used as the label name.
* ``split_pct``: specifies how to split the data into training/validation/test. If it's not specified, the data is split into 80% for training 10% for validation and 10% for testing. The pipeline constructs three additional vectors indicating the training/validation/test masks. For ``classification`` and ``regression`` tasks, the names of the mask tensors are ``train_mask``, ``val_mask`` and ``test_mask``.
* ``custom_split_filenames``: specifies the customized training/validation/test mask. It has field named ``train``, ``valid``, and ``test`` to specify the path of the mask files. It is possible that one of the subfield here leaves empty and it will be treated as none. It will override the ``split_pct`` once provided. Refer to :ref:`Label split files <customized-split-labels>` for detailed explanations.
* ``label_stats_type``: specifies the statistic type used to summarize labels. So far, only support one value, i.e., ``frequency_cnt``.

.. _feat-transform:

Feature transformation
.........................
GraphStorm provides a set of transformation operations for different types of feautures.

* **HuggingFace tokenizer transformation** tokenizes text strings with a HuggingFace tokenizer. The ``name`` field in the feature transformation dictionary is ``tokenize_hf``. The dict should contain two additional fields.

  1. ``bert_model`` specifies the LM model used for tokenization. Users can choose any `HuggingFace LM models <https://huggingface.co/models>`_ from one of the following types: ``"bert", "roberta", "albert", "camembert", "ernie", "ibert", "luke", "mega", "mpnet", "nezha", "qdqbert","roc_bert"``, such as ``"bert-base-uncased" and "roberta-base"``
  2. ``max_seq_length`` specifies the maximal sequence length.

  Example:

  .. code:: json

    "transform": {"name": "tokenize_hf",
                  "bert_model": "bert-base-uncased",
                  "max_seq_length": 16},

* **HuggingFace LM transformation** encodes text strings with a HuggingFace LM model.  The ``name`` field in the feature transformation dictionary is ``bert_hf``. The dict should contain two additional fields.

  1. ``bert_model`` specifies the LM model used for embedding text. Users can choose any `HuggingFace LM models <https://huggingface.co/models>`_ from one of the following types: ``"bert", "roberta", "albert", "camembert", "ernie", "ibert", "luke", "mega", "mpnet", "nezha", "qdqbert","roc_bert"``, such as ``"bert-base-uncased" and "roberta-base"``
  2. ``max_seq_length`` specifies the maximal sequence length.

  Example:

  .. code:: json

    "transform": {"name": "bert_hf",
                  "bert_model": "roberta-base",
                  "max_seq_length": 256},

* **Numerical MAX_MIN transformation** normalizes numerical input features with `val = (val-min)/(max-min)`, where `val` is the feature value, `max` is the maximum value in the feature and `min` is the minimum value in the feature. The ``name`` field in the feature transformation dictionary is ``max_min_norm``. The dictionary can contain four optional fields: ``max_bound``, ``min_bound``, ``max_val`` and ``min_val``. 

  - ``max_bound`` specifies the maximum value allowed in the feature. Any number larger than ``max_bound`` will be set to ``max_bound``. Here, `max = min(np.amax(feats), ``max_bound``)`.
  - ``min_bound`` specifies the minimum value allowed in the feature. Any number smaller than ``min_bound`` will be set to ``min_bound``. Here, `min` = max(np.amin(feats), ``min_bound``). 
  - ``max_val`` defines the `max` in the transformation formula. When ``max_val`` is provided, `max` is always equal to ``max_val``.
  - ``min_val`` defines the `min` in the transformation formula.  When ``min_val`` is provided, `min` is always equal to ``min_val``.
  
  ``max_val`` and ``min_val`` are mainly used in the inference stage, where we want to use the same `max` and `min` values computed in the training stage to normalize inference data.

  Example:

  .. code:: json

    "transform": {"name": "max_min_norm",
                  "max_bound": 2.,
                  "min_bound": -2.}

* **Numerical Rank Gauss transformation** normalizes numerical input features with rank gauss normalization. It maps the numeric feature values to gaussian distribution based on ranking. The method follows the description in the normalization section of `the Porto Seguro's Safe Driver Prediction kaggle competition <https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927>`_. The ``name`` field in the feature transformation dictionary is ``rank_gauss``. The dict can contains two optional fields, i.e., ``epsilon`` which is used to avoid ``INF`` float during computation and ``uniquify`` which controls whether deduplicating input features before computing rank gauss norm.

  Example:

  .. code:: json

    "transform": {"name": "rank_gauss",
                  "epsilon": 1e-5,
                  "uniquify": True, }

* **Convert to categorical values** converts text data to categorial values. The ``name`` field is ``to_categorical``, and ``separator`` specifies how to split the string into multiple categorical values (this is only used to define multiple categorical values). If ``separator`` is not specified, the entire string is considered as a single categorical value. ``mapping`` (optional) is a dictionary that specifies how to map a string to an integer value that defines a categorical value. If ``mapping`` is provided, any string value which is not in the ``mapping`` will be ignored. The ``mapping`` field is mainly used in the inference stage when we want to keep the same categorical mapping as in the training stage.

  Example:

  .. code:: json

    "transform": {"name": "to_categorical"},

* **Numerical Bucket transformation** normalizes numerical input features with buckets. The input features are divided into one or multiple buckets. Each bucket stands for a range of floats. An input value can fall into one or more buckets depending on the transformation configuration. The ``name`` field in the feature transformation dictionary is ``bucket_numerical``. Users can to provide ``range`` and ``bucket_cnt`` fields, where ``range`` defines a numerical range, and ``bucket_cnt`` defines number of buckets among the range. All buckets will have same length, and each of them is left included. e.g, bucket ``[a, b)`` will include ``a``, but not ``b``. All input feature column data are categorized into respective buckets using this method. Any input data lower than the minimum value will be assigned to the first bucket, and any input data exceeding the maximum value will be assigned to the last bucket. For example, with ``range: [10,30]`` and ``bucket_cnt: 2``, input data ``1`` will fall into the bucket ``[10, 20]``, input data ``11`` will be mapped to ``[10, 20]``, input data ``21`` will be mapped to ``[20, 30]``, input data ``31`` will be mapped to ``[20, 30]``. Finally GraphStorm uses one-hot-encoding to encode the feature for each numerical bucket. If a user wants to make numeric values fall into more than one bucket, it is suggested to use the ``slide_window_size`` field. ``slide_window_size`` defines a number, e.g., ``s``. Then each value ``v`` will be transformed into a range from ``v - s/2`` through ``v + s/2`` , and assigns the value ``v`` to every bucket that the range covers.

  Example:

  .. code:: json

    "transform": {"name": "bucket_numerical",
                  "range": [10, 50],
                  "bucket_cnt": 2,
                  "slide_window_size": 10},

* **No-op vector truncation (experimental)** truncates feature vectors to the length requested. The ``name`` field can be empty (e.g., ``{name: }``), and an integer ``truncate_dim`` value will determine the length of the output vector. This can be useful when experimenting with input features that were trained using `Matryoshka Representation Learning <https://arxiv.org/abs/2205.13147>`_.

  Example:

  .. code:: json

    "transform": {"name": ,
                  "truncate_dim": 24},

.. _gcon-output-format:

Outputs of the graph consturction command
............................................
The graph construction command outputs two formats: ``DistDGL`` or ``DGL`` specified by the argument **-\-output-format**. 

If select ``DGL``, the output includes an `DGLGraph <https://docs.dgl.ai/en/1.0.x/generated/dgl.save_graphs.html>`_ file, named ``<graph_name>.dgl`` under the folder specified by the **-\-output-dir** argument, where `<graph_name>` is the value of argument **-\-graph-name**.

If select ``DistDGL``, the output will be a partitioned `DistDGL graph <https://doc.dgl.ai/guide/distributed-preprocessing.html#partitioning-api>`_. It includes a JSON file, named `<graph_name>.json` that describes the meta-information of the partitioned graph, a set of ``part*`` folders under the folder specified by the **-\-output-dir** argument, where the `*` is the number specified by the **-\-num-parts** argument.

Besides the graph data, the graph construction command also generate other files that contain related metadata information associated with the graph data, including a set of node and edge ID mapping files, a new construction configuration JSON file that records the details of feature transformation operations, and lable statistic summary files if required in the ``label_stats_type`` field.

.. _gs-id-mapping-files:

    - **Node and Edge Mapping Files:**
      There are two node/edge id mapping stages during graph construction. The first mapping occurs when GraphStorm converts the original user provided node ids into integer-based node ids, and the second mapping happends when graph partition operation shuffles these integer-based node ids to each partition with new node ids. Meanwhile, graph construction also saves two sets of node id mapping files as parts of its outputs.

      Outputs of the first mapping stage are stored at the ``raw_id_mappings`` folder under the path specified by the **-\-output-dir** argument. For each node type, there is a dedicated folder named after the ``node_type`` filed, in which contains parquet format files named after ``part-*****.parquet``, where ``*****`` represents five digit numbers starting from ``00000``.

      Outputs of the second mapping stage are two PyTorch tensor files, i.e., ``node_mapping.pt`` and ``edge_mapping.pt``, each of which maps the node and edge in the partitoined graph into the integer original node and edge id space. The node ID mapping is stored as a dictionary of 1D tensors whose key is the node type and value is a 1D tensor mapping between shuffled node IDs and the original node IDs. The edge ID mapping is stored as a dictionary of 1D tensors whose key is the edge type and value is a 1D tensor mapping between shuffled edge IDs and the original edge IDs.

    - **New Construction Configuration JSON:**
      By default, GraphStorm will regenerate a construction configuration JSON file that copies the contents in the given JSON file specified by the **--conf-file** argument. In addition if there are transformations of features occurred, this newly generated JSON file will include some additional information. For example, if the original configuration JSON file requires to perform a **Convert to categorical values** transformation without giving the ``mapping`` dictionary, the newly generated configuration JSON file will add this ``mapping`` dictionary with the actual values and their mapping ids. This added information could help construct new graphs for fine-tunning saved models or doing inference with saved models.

      If users provide a value of the **-\-output-conf-file** argument, the newly generated configuration file will use this value as the file name. Otherwise GraphStorm will save the configuration JSON file in the **-\-output-dir** with name ``data_transform_new.json``.

    - **Label Statistic Summary JSONs:**
      If required in the ``label_stats_type`` field, the graph construction command will compute statistics of labels and save them in a ``node_label_stats.json`` or a ``edge_label_stats.json``. 

.. note:: These mapping files are important for mapping the training and inference outputs. Therefore, DO NOT move or delete them.

A construction configuration JSON example
..........................................

This section provides a construction configuration JSON associated to the :ref:`simple raw data example <simple-input-raw-data-example>` as an example for refernece.

.. code:: yaml

    {
        "version": "gconstruct-v0.1",
        "nodes": [
            {
                "node_id_col":  "nid",
                "node_type":    "paper",
                "format":       {"name": "parquet"},
                "files":        "paper_nodes.parquet",
                "features":     [
                    {
                        "feature_col":  ["aff"],
                        "feature_name": "aff_feat",
                        "transform":    {"name": "to_categorical",
                                         "mapping": {"NE": 0, "MT": 1,"UL": 2, "TT": 3,"UC": 4}}
                    },
                    {
                        "feature_col":  "abs",
                        "feature_name": "abs_bert",
                        "out_dtype": "float32",
                        "transform": {"name": "bert_hf",
                                     "bert_model": "roberta",
                                     "max_seq_length": 16}
                    },
                ],
                "labels":       [
                    {
                        "label_col":    "class",
                        "task_type":    "classification",
                        "custom_split_filenames": {
                                            "train": "train.json",
                                            "valid": "val.json",
                                            "test":  "test.json"},
                        "label_stats_type": "frequency_cnt",
                    },
                ],
            },
            {
                "node_id_col":  "domain",
                "node_type":    "subject",
                "format":       {"name": "parquet"},
                "files":        "subject_nodes.parquet",
            },
            {
                "node_id_col":  "n_id",
                "node_type":    "author",
                "format":       {"name": "parquet"},
                "files":        "author_nodes.parquet",
                "features":     [
                    {
                        "feature_col":  ["hdx"],
                        "feature_name": "feat",
                        "out_dtype": 'float16',
                        "transform": {"name": "max_min_norm",
                                      "max_bound": 1000.,
                                      "min_val":   0.}
                    },
                ],
            },
            {
                "node_type":    "author",
                "format":       {"name": "hdf5"},
                "files":        "author_node_embeddings.h5",
                "features":     [
                    {
                        "feature_col":  ["embedding"],
                        "feature_name": "embed",
                        "out_dtype": 'float16',
                    },
                ],

            },
        ],
        "edges": [
            {
                "source_id_col":    "nid",
                "dest_id_col":      "domain",
                "relation":         ["paper", "has", "subject"],
                "format":           {"name": "parquet"},
                "files":            ["paper_has_subject_edges.parquet"],
                "labels":       [
                    {
                        "label_col": "cnt",
                        "task_type": "regression",
                        "custom_split_filenames": {
                                            "train": "train_edges.json",
                                            "valid": "val_edges.json",
                                            },
                    },
                ],
            },
            {
                "source_id_col":    "nid",
                "dest_id_col":      "n_id",
                "relation":         ["paper", "written-by", "author"],
                "format":           {"name": "parquet"},
                "files":            ["paper_written-by_author_edges.parquet"],
            }
        ]
    }

.. note:: For a real runnable example, please refer to the :ref:`input JSON file <input-config>` used in the :ref:`Use Your Own Graphs Tutorial <use-own-data>`.

A full argument list of the ``gconstruct.construct_graph`` command
...................................................................

* **-\-conf-file**: (**Required**) the path of the configuration JSON file.
* **-\-num-processes**: the number of processes to process the data simulteneously. Default is 1. Increase this number can speed up data processing, but will also increase the CPU memory consumption.
* **-\-num-processes-for-nodes**: the number of processes to process node data simulteneously. Increase this number can speed up node data processing.
* **-\-num-processes-for-edges**: the number of processes to process edge data simulteneously. Increase this number can speed up edge data processing.
* **-\-output-dir**: (**Required**) the path of the output data files.
* **-\-graph-name**: (**Required**) the name assigned for the graph.
* **-\-remap-node-id**: boolean value to decide whether to rename node IDs or not. Adding this argument will set it to be true, otherwise false.
* **-\-add-reverse-edges**: boolean value to decide whether to add reverse edges for the given graph. Adding this argument sets it to true; otherwise, it defaults to false. It is **strongly** suggested to include this argument for graph construction, as some nodes in the original data may not have in-degrees, and thus cannot update their presentations by aggregating messages from their neighbors. Adding this arugment helps prevent this issue.
* **-\-output-format**: the format of constructed graph, options are ``DGL``,  ``DistDGL``.  Default is ``DistDGL``. It also accepts multiple graph formats at the same time separated by an space, for example ``--output-format "DGL DistDGL"``. The output format is explained in the :ref:`Output <gcon-output-format>` section above.
* **-\-num-parts**: an integer value that specifies the number of graph partitions to produce. This is only valid if the output format is ``DistDGL``.
* **-\-skip-nonexist-edges**: boolean value to decide whether skip edges whose endpoint nodes don't exist. Default is true.
* **-\-ext-mem-workspace**: the directory where the tool can store intermediate data during graph construction. Suggest to use high-speed SSD as the external memory workspace.
* **-\-ext-mem-feat-size**: the minimal number of feature dimensions that features can be stored in external memory. Default is 64.
* **-\-output-conf-file**: The output file with the updated configurations that records the details of data transformation, e.g., convert to categorical value mappings, and max-min normalization ranges. If not specified, will save the updated configuration file in the **-\-output-dir** with name `data_transform_new.json`.

.. _configurations-partition:

Graph Partition for DGL Graphs
********************************

.. warning:: The two graph partition tools in this section were originally implemented for quick code debugging and are no longer maintained. It is **strongly** suggested to use the ``gconstruct.construct_graph`` command or the :ref:`Distributed Graph Construction <distributed-gconstruction>` guideline for graph construction.

For users who are already familiar with DGL and know how to construct DGL graphs, GraphStorm provides two graph partition tools to split DGL graphs into the required input format for GraphStorm model training and inference.

* `partition_graph.py <https://github.com/awslabs/graphstorm/blob/main/tools/partition_graph.py>`_: for Node/Edge Classification/Regress task graph partition.
* `partition_graph_lp.py <https://github.com/awslabs/graphstorm/blob/main/tools/partition_graph_lp.py>`_: for Link Prediction task graph partition.

`partition_graph.py <https://github.com/awslabs/graphstorm/blob/main/tools/partition_graph.py>`_ arguments
...........................................................................................................

- **-\-dataset**: (**Required**) the graph dataset name defined for the saved DGL graph file.
- **-\-filepath**: (**Required**) the file path of the saved DGL graph file.
- **-\-target-ntype**: the node type for making prediction, required for node classification/regression tasks. This argument is associated with the node type having labels. Current GraphStorm supports **one** prediction node type only.
- **-\-ntype-task**: the node type task to perform. Only support ``classification`` and ``regression`` so far. Default is ``classification``.
- **-\-nlabel-field**: the field that stores labels on the prediction node type, **required** if **target-ntype** is set. The format is ``nodetype:labelname``, e.g., `"paper:label"`.
- **-\-target-etype**: the canonical edge type for making prediction, **required** for edge classification/regression tasks. This argument is associated with the edge type having labels. Current GraphStorm supports **one** prediction edge type only. The format is ``src_ntype,etype,dst_ntype``, e.g., `"author,write,paper"`.
- **-\-etype-task**: the edge type task to perform. Only allow ``classification`` and ``regression`` so far. Default is ``classification``.
- **-\-elabel-field**: the field that stores labels on the prediction edge type, required if **target-etype** is set. The format is ``src_ntype,etype,dst_ntype:labelname``, e.g., `"author,write,paper:label"`.
- **-\-generate-new-node-split**: a boolean value, required if need the partition script to split nodes for training/validation/test sets. If this argument is set to ``true``, the **target-ntype** argument **must** also be set.
- **-\-generate-new-edge-split**: a boolean value, required if need the partition script to split edges for training/validation/test sets. If this argument is set to ``true``, the **target-etype** argument **must** also be set.
- **-\-train-pct**: a float value (\>0. and \<1.) with default value ``0.8``. If you want the partition script to split nodes/edges for training/validation/test sets, you can set this value to control the percentage of nodes/edges for training.
- **-\-val-pct**: a float value (\>0. and \<1.) with default value ``0.1``. You can set this value to control the percentage of nodes/edges for validation. 

.. Note::
    The sum of the **train-pct** and **val-pct** should be less than 1. And the percentage of test nodes/edges is the result of 1-(train_pct + val_pct).

- **-\-add-reverse-edges**: if add this argument, will add reverse edges to the given graph.
- **-\-num-parts**: (**Required**) an integer value that specifies the number of graph partitions to produce. Remember this number because we will need to set it in the model training step.
- **-\-output**: (**Required**) the folder path that the partitioned DGL graphs will be saved.

`partition_graph_lp.py <https://github.com/awslabs/graphstorm/blob/main/tools/partition_graph_lp.py>`_ arguments
..................................................................................................................
- **-\-dataset**: (**Required**) the graph name defined for the saved DGL graph file.
- **-\-filepath**: (**Required**) the file path of the saved DGL graph file.
- **-\-target-etypes**: (**Required**) the canonical edge types for making prediction. GraphStorm supports multiple predict edge types that are separated by a white space. The format is ``src_ntype1,etype1,dst_ntype1 src_ntype2,etype2,dst_ntype2``, e.g., `"author,write,paper paper,citing,paper"`.
- **-\-train-pct**: a float value (\>0. and \<1.) with default value ``0.8``. If you want the partition script to split edges for training/validation/test sets, you can set this value to control the percentage of edges for training.
- **-\-val-pct**: a float value (\>0. and \<1.) with default value ``0.1``. You can set this value to control the percentage of edges for validation. 

.. Note:: 
    The sum of the **train-pct** and **val-pct** should less than 1. And the percentage of test edges is the result of 1-(train_pct + val_pct).

- **-\-add-reverse-edges**: if add this argument, will add reverse edges to the given graphs.
- **-\-num-parts**: (**Required**) an integer value that specifies the number of graph partitions to produce. Remember this number because we will need to set it in the model training step.
- **-\-output**: (**Required**) the folder path that the partitioned DGL graph will be saved.