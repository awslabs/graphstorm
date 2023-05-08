.. _use-own-data:

Use Your Own Data Tutorial
============================
It is easy for users to prepare their own graphs data and leverage GraphStorm's built-in GNN models, e.g., RGCN and RGAT, to perform graph machine learning tasks.  It takes three steps:

- Step 1: Prepare your own graph data.
- Step 2: Modify configuration YAML file.
- Step 3: Launch GraphStorm command for training/inference.

.. Note:: 
    All commands below should be run within the GraphStorm container environment. 

Step 1: Prepare Your Own Graph Data
-------------------------------------
There are two options to prepare your own graph data for using GraphStorm:

- Put your graph in the required raw data format, and use GraphStorm's construction tools to automatically generate the input files. This is the preferred way.
- Prepare your data as a DGL heterogeneous graph following the required format, and then use GraphStorm's partition tool to generate the input files. This option is for advanced DGL users.

Option 1: Required raw data format
.......................................
GraphStorm provides a graph construction tool to generate input files for using the training/inference commands. The general information about the raw data format can be found in the `gconstruct README <https://github.com/awslabs/graphstorm/tree/main/python/graphstorm/gconstruct#readme>`_. 

In general, the graph construction tool need three set of files.

* A configuration JSON file, which describes the graph data, the tasks to perform, the node features, and data file paths.
* A set of files of nodes. Each type of nodes must have one file associated. If the file is too big, users can split this one file into multiple files that have the same columns and different rows.
* A set of files of edges. Each type of edges must have one file associated. If the file is too big, users can split this one file into multiple files that have the same columns and different rows.

Here use the `ACM publication graph <https://data.dgl.ai/dataset/ACM.mat>`_ for node classification as an demonstration to show how to prepare your own graph data, and what these files and their contents are like.

First go the ``/graphstorm/examples/`` folder.

.. code-block:: bash

    cd /graphstorm/examples 

Then run the command to create the ACM data with the required raw format.

.. code-block:: bash
    
    python3 -m /graphstorm/examples/acm_data.py --output-path /tmp/acm_raw 

Once succeeds, the command will create a set of folders and files under the ``/tmp/acm_raw/`` folder, as shown below:

.. code-block:: bash
    
    /tmp/acm_raw
    config.json
    |- edges
        author_writing_paper.parquet
        paper_cited_paper.parquet
        paper_citing_paper.parquet
        paper_is-about_subject.parquet
        paper_written-by_author.parquet
        subject_has_paper.parquet
    |- nodes
        author.parquet
        paper.parquet
        subject.parquet

.. _input-config:

Input Configuration JSON
```````````````````````````
GraphStorm's graph construction tool relies on the configuration JSON to provide graph information. The explain the format of the configuration JSON contents is in the `gconstruct README <https://github.com/awslabs/graphstorm/tree/main/python/graphstorm/gconstruct#readme>`_. Below show the contents of the examplar ACM config.json file.

.. code-block:: json

    {
        "nodes": [
            {
                "node_type": "author",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/author.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat"
                    }
                ]
            },
            {
                "node_type": "paper",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/paper.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat"
                    }
                ],
                "labels": [
                    {
                        "label_col": "label",
                        "task_type": "classification",
                        "split_pct": [
                            0.8,
                            0.1,
                            0.1
                        ]
                    }
                ]
            },
            {
                "node_type": "subject",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/subject.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat"
                    }
                ]
            }
        ],
        "edges": [
            {
                "relation": [
                    "author",
                    "writing",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/author_writing_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "cited",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_cited_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "citing",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_citing_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "is-about",
                    "subject"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_is-about_subject.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "written-by",
                    "author"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_written-by_author.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "subject",
                    "has",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/subject_has_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            }
        ]
    }

Based on the original ACM dataset, this example build a simple heterogenous graph that contains three types of nodes and six types of edges as shown in the diagram below.

.. figure:: ../../../tutorial/ACM_schema.png
    :align: center

.. _raw-data-files:

Input raw node/edge data files
```````````````````````````````
The raw node and edge data files are both in parquet format, whose contents are demonstrated as the diagram below.

.. figure:: ../../../tutorial/ACM_raw_parquet.png
    :align: center

In this example, only the ``paper`` nodes have labels and the task is node classification. So, in the JSON file, the ``paper`` node has the ``labels`` field, and the ``task_type`` is specified as ``classification``. All edges do not have feature associated. Therefore, there only have two columns in these parquet files for edges, the ``source_id`` and the ``dest_id``.

The configuration JSON file along with these node and edge parquet files are the required inputs of the GraphStorm's construction tool. Then we can use the tool to create the partition graph data as the following command does.

.. code-block:: bash

    python3 -m graphstorm.gconstruct.construct_graph \
               --conf_file /tmp/acm_raw/config.json \
               --output_dir /tmp/acm_nc \
               --num_partitions 1 \
               --graph_name acm

.. _output-graph-construction:

Outputs of graph construction
```````````````````````````````
Outputs of the command are under the ``/tmp/acm_nc/`` folder like the followings:

.. code-block:: bash

    /tmp/acm_nc
    acm.json 
    node_mapping.pt
    edge_mapping.pt 
    |- part0
        edge_feat.dgl
        graph.dgl
        node_feat.dgl 

These files become the inputs of GraphStorm's launch scripts and APIs.

Option 2: Required DGL graph
................................
For some users who are already familiar with `DGL <https://www.dgl.ai/>`_, they can convert their graph data into the required DGL graph format. And then use GraphStorm's partition tools to create the inputs of GraphStorm's launch scripts and APIs.

Required DGL graph format:

- a `dgl.heterograph <https://docs.dgl.ai/generated/dgl.heterograph.html#dgl.heterograph>`_.
- All nodes/edges features are set in nodes/edges' data field, and remember the feature names, which will be used in the later steps.
    - For nodes' features, the common way to set features is like ``g.nodes['nodetypename'].data['featurename']=nodefeaturetensor``, The formal explanation of DGL's node feature could be found in the `Using node features <https://docs.dgl.ai/generated/dgl.DGLGraph.nodes.html>`_.
    - For edges' features, the common way to set features is like ``g.edges['edgetypename'].data['featurename']=edgefeaturetensor``, The formal explanation of DGL's edge feature could be found in the `Using edge features <https://docs.dgl.ai/generated/dgl.DGLGraph.edges.html>`_.
- Save labels (for node/edge tasks) into the target nodes/edges as a feature, and remember the label feature names, which will be used in the later steps.
    - The common way to set node-related labels as a feature is like ``g.nodes['predictnodetypename'].data['labelname']=nodelabeltensor``.
    - The common way to set edge-related labels as a feature is like ``g.nodes['predictedgetypename'].data['labelname']=edgelabeltensor``.
    - For link prediction task, a common way to extract labels is to use existing edges as the positive edges and use negative sampling method to extract non-exist edges as negative edges. So in this step, we do not need to set the labels. The GraphStorm has implemented this function.
- (Optional) if you have your own train/validation/test split on nodes/edges, you can put the train/validation/test nodes/edges index tensors as three nodes/edges features with the feature names as ``train_mask``, ``val_mask``, and ``test_mask``. If you do not have nodes/edges split, you can use the split functions provided in the GSF partition scripts to create them in the next step.
    - For training nodes, the setting is like ``g.nodes['predictnodetypename'].data['train_mask']=trainingnodeindexetensor``.
    - For validation nodes, the setting is like ``g.nodes['predictnodetypename'].data['val_mask']=validationnodeindexetensor``. Make sure you use 'val_mask' as the feature name because the GSF uses this name by default.
    - For validation nodes, the setting is like ``g.nodes['predictnodetypename'].data['test_mask']=testnodeindexetensor``.
    - Similar to nodes splits, you can use the same feature names, ``train_mask``, ``val_mask``, and ``test_mask``, to assign the edge index tensors. 
    - The index tensor is either a boolean tensor, or an integer tensor including only 0s and 1s.

Once this DGL graph is constructed, you can use DGL's `save_graphs() <https://docs.dgl.ai/generated/dgl.save_graphs.html?highlight=save_graphs#dgl.save_graphs>`_ function to save it into a local file. The file name must follow GraphStorm convention: ``<datasetname>.dgl``. You can give your graph dataset a name, e.g., ``acm`` or ``ogbn_mag``. 

The ACM graph data example
`````````````````````````````
For the ACM data, the following command can create a DGL graph as the input for GraphStorm’s partition tools.

.. code-block:: bash

    python3 -m /graphstorm/examples/acm_data.py \
               --output-type dgl \
               --output-path /tmp/acm_dgl 

The below image show how the built DGL ACM data looks like.

.. figure:: ../../../tutorial/ACM_graph_schema.png
    :align: center

.. figure:: ../../../tutorial/ACM_LabelAndMask.png
    :align: center

Partition the DGL ACM graph for node classification
```````````````````````````````````````````````````````
GraphStorm provides two graph partition scripts, the `partition_graph.py <https://github.com/awslabs/graphstorm/blob/main/tools/partition_graph.py>`_ for node/edge prediction graph partition, and the `partition_graph_lp.py <https://github.com/awslabs/graphstorm/blob/main/tools/partition_graph_lp.py>`_ for the link prediction graph partition.

The below command partition the DGL ACM graph, the ``acm.dgl`` in the ``/tmp/acm_dgl`` folder into one partition, and save the partitioned data to ``/tmp/acm_nc/`` folder.

.. code-block:: bash

    python3 /graphstorm/tools/partition_graph.py \
            --dataset acm\
            --filepath /tmp/acm_dgl \
            --num_parts 1 \
            --predict_ntype paper \
            --nlabel_field paper:label \
            --output /tmp/acm_nc

Outputs of the command are under the ``/tmp/acm_nc/`` folder with the same contents as the Option 1.

Step 2: Modify the YAML configuration file to include your own data's information
-----------------------------------------------------------------------------------
It is common that users will copy and reuse GraphStorm's built-in scripts and yaml files to run on their own graph data, but forget to change the contents of yaml files to match their own data. Below are some configurations that users need to double check and make changes accordingly.

- **part_config**: please change value of this configure to where you store the partitioned graph's JSON file. It is better to use an absolute path to avoid path mis-match.
- **ip_config**: please make sure ip_list.txt created and the path of the ip_list.txt file is correct.

If you conduct Classification/Regression tasks,

- **label_field**: please change value of this field to fit to the field name of labeled data in your graph data.
- **num_classes**: please change value of this filed to fit to the number of classes to be predicted in your graph data if doing a Classification task.

If you conduct Node Classification/Regression tasks,

- **predict_ntype**: please change value of this field to the node type that the label is associated, which should be the same node type for prediction.

If you conduct Edge Classification/Regression tasks,

- **target_etype**: please change value of this field to the edge type that the label is associated, which should be the same edge type for prediction.

If you conduct Link Prediction tasks,

- **train_etype**: please specify value of this field for the edge type that you want to do link prediction for the downstream task, e.g. recommendation or search. Although if not specified, i.e. put None as the value, all edge types will be used for training, this might not commonly used practice for most Link Prediction related tasks.
- **eval_etype**: it is highly recommended that set this value to be the same as the value of train_etype, so that the evaluation metric can truly demonstrate the performance of models.

Besides these configuration, it is also important for you to use the correct format to configure node/edge types in the yaml files. For example, in an edge-related task, you should provide a canonical edge type, e.g. `**user,write,paper**` (no white spaces in this string), for edge types, rather than the edge name only, e.g. the `**write**`. 

For more detailed information of these options, please refer to the :ref:`GraphStorm Configurations <configurations>` page.

An example ACM  YAML file for node classification
..................................................
Below is an example YAML configuration file for the ACM data, which sets to use GraphStorm's built-in RGCN model for node classification on the ``paper`` type nodes. The YAML file can also be found at the ``/graphstorm/examples/use_your_own_data/`` folder.

.. code-block:: yaml

    ---
    version: 1.0
    gsf:
    basic:
        model_encoder_type: rgcn
        backend: gloo
        verbose: false
    gnn:
        fanout: "50,50"
        num_layers: 2
        hidden_size: 256
        use_mini_batch_infer: false
    input:
        restore_model_path: null
    output:
        save_model_path: /tmp/acm_nc/models
        save_embeds_path: /tmp/acm_nc/embeds
    hyperparam:
        dropout: 0.
        lr: 0.0001
        lm_tune_lr: 0.0001
        num_epochs: 200
        batch_size: 1024
        bert_infer_bs: 128
        wd_l2norm: 0
        alpha_l2norm: 0.
    rgcn:
        num_bases: -1
        use_self_loop: true
        sparse_optimizer_lr: 1e-2
        use_node_embeddings: false
    node_classification:
        target_ntype: "paper"
        label_field: "label"
        multilabel: false
        num_classes: 14


Step 3: Launch training script on your own graphs
---------------------------------------------------

With the partitioned data and configuration YAML file available, it is easy to use GraphStorm's training scripts to launch the training job. 

.. Note:: we assume an ip_list.txt file has been created in the ``/tmp/`` folder. Users can use the following commands to create this file.

    .. code-block:: bash

        touch /tmp/ip_list.txt
        echo 127.0.0.1 > /tmp/ip_list.txt

Below is a simple launch script example that train a GraphStorm built-in RGCN model on the ACM data for node classification.

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
            --workspace /graphstorm/examples/use_your_own_data \
            --part_config /tmp/acm_nc/acm.json \
            --ip_config /tmp/ip_list.txt \
            --num_trainers 4 \
            --num_servers 1 \
            --num_samplers 0 \
            --ssh_port 2222 \
            --cf /graphstorm/examples/use_your_own_data/acm_nc.yaml \
            --node-feat-name paper:feat author:feat subject:feat

Similar to the Quick-Start tutorial, users can launch the inference script on thier own data. Below is the customized scripts for predicting the classes of nodes in the test set of the ACM graph.

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
               --inference \
               --workspace /graphstorm/examples/use_your_own_data \
               --part_config /tmp/acm_nc/acm.json \
               --ip_config /tmp/ip_list.txt \
               --num_trainers 4 \
               --num_servers 1 \
               --num_samplers 0 \
               --ssh_port 2222 \
               --cf /graphstorm/examples/use_your_own_data/acm_nc.yaml \
               --node-feat-name paper:feat author:feat subject:feat \
               --restore-model-path /data/acm_nc/models/epoch-0 \
               --save-prediction-path  /data/acm_nc/predictions

