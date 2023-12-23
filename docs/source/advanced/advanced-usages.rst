.. _advanced_usages:

GraphStorm Advanced Usages
===========================

Prevent Information Leakage in Link Prediction
-----------------------------------------------

Link prediction is widely employed as a pre-training technique to generate high-quality entity representations applicable to diverse business applications. However, implementing a training loop for link prediction tasks needs to carefully handle the information leakage problems caused by 1) including target edges in message passing, and 2) including validation/test edges in message passing during training. (`This paper <https://arxiv.org/pdf/2306.00899.pdf>`_ provides more details.)

GraphStorm provides supports to avoid theses problems:

* To avoid including target edges in message passing, users need to set ``exclude_training_targets`` to `True` and provide ``reverse_edge_types_map`` when launching link prediction training tasks. These two arguments tell GraphStorm to exclude the training target edges and the corresponding reverse edges when doing message passing. More explanation of the two arguments can be found on the :ref:`Training and Inference Configurations<configurations-run>`.

* To avoid including validation/test edges in message passing during model training, users need to mask validation edges and test edges with ``val_mask`` and ``test_mask`` respectively. Users also need to mask all the other edges with ``train_mask``.

Speedup Link Prediction Training
---------------------------------------------
GraphStorm relies on ``dgl.dataloading.MultiLayerNeighborSampler`` and ``train_mask`` to avoid sampling validation and test edges during training. Basically, it only samples edges with ``train_mask`` set to be `True`. However, the implementation is not efficient. To speedup graph sampling during link prediction training, GraphStorm provides four link prediction dataloaders (i.e., ``fast_uniform``, ``fast_joint``, ``fast_localuniform`` and ``fast_localjoint``) with more efficient implementation but less precise neighbor sampling behavior.

To be more specific, these dataloaders will do neighbor sampling regardless of any masks in the beginning, and later remove edges with  ``val_mask`` or ``test_mask`` set to be `True`. In theory, a sampled subgraph may have less neighbor nodes than expected as some of them would be removed. However, with a graph having hundreds of millions of edges (or more) and small validation and test sets, e.g., each with less than 10% edges, the impact is negligible.

With DGL 1.0.4, ``fast_localuniform`` dataloader can speedup 2.4X over ``localuniform`` dataloader on training a 2 layer RGCN on MAG dataset on four g5.48x instances.

Hard Negative sampling in Link Prediction Training
-------------------------------------
GraphStorm provides support for users to define hard negative edges for a positive edge during Link Prediction Training.
Currently, hard negative edges are constructed by replacing the destination nodes of edges with pre-defined hard negatives.
For example, given an edge (``src_pos``, ``dst_pos``) and its hard negative destination nodes ``hard_0`` and ``hand_1``, GraphStorm will construct two hard negative edges, i.e., (``src_pos``, ``hard_0``) and (``src_pos``, ``hand_1``).

The hard negatives are stored as edge features of the target edge type.
Users can provide the hard negatives for each edge type through ``train_etypes_negative_dstnode`` in the training config yaml.
For example, the following yaml block defines the hard negatives for edge type (``src_type``,``rel_type0``,``dst_type``) as the edge feature ``negative_nid_field_0`` and the hard negatives for edge type (``src_type``,``rel_type1``,``dst_type``) as the edge feature ``negative_nid_field_1``.

  .. code-block:: yaml

    train_etypes_negative_dstnode:
      - src_type,rel_type0,dst_type:negative_nid_field_0
      - src_type,rel_type1,dst_type:negative_nid_field_1

Users can also define the number of hard negatives to sample for each edge type during training though ``num_train_hard_negatives`` in the training config yaml.
For example, the following yaml block defines the number of hard negatives for edge type (``src_type``,``rel_type0``,``dst_type``) is 5 and the number of hard negatives for edge type (``src_type``,``rel_type1``,``dst_type``) is 10.

  .. code-block:: yaml

    num_train_hard_negatives:
      - src_type,rel_type0,dst_type:5
      - src_type,rel_type1,dst_type:10

Hard negative sampling can be used together with any link prediction negative sampler, such as ``uniform``, ``joint``, ``inbatch_joint``, etc.
By default, GraphStorm will sample hard negatives first to fulfill the requirement of ``num_train_hard_negatives`` and then sample random negatives to fulfill the requirement of ``num_negative_edges``.
In general, GraphStorm covers following cases:

- **Case 1** ``num_train_hard_negatives`` is larger or equal to ``num_negative_edges``. GraphStorm will only sample hard negative nodes.
- **Case 2** ``num_train_hard_negatives`` is smaller than ``num_negative_edges``. GraphStorm will randomly sample ``num_train_hard_negatives`` hard negative nodes from the hard negative set and then randomly sample ``num_negative_edges - num_train_hard_negatives`` negative nodes.
- **Case 3** GraphStorm supports cases when some edges do not have enough hard negatives provided by users. For example, the expected ``num_train_hard_negatives`` is 10, but an edge only have 5 hard negatives. In certain cases, GraphStorm will use all the hard negatives first and then randomly sample negative nodes to fulfill the requirement of ``num_train_hard_negatives``. Then GraphStorm will go back to **Case 1** or **Case 2**.

** Preparing graph data for hard negative sampling **

The gconstruct pipeline of GraphStorm provides support to load hard negative data from raw input.
Hard destination negatives can be defined through ``edge_dst_hard_negative`` transformation.
The ``feature_col`` field of ``edge_dst_hard_negative`` must stores the raw node ids of hard destination nodes.
GraphStorm accepts two types of hard negative inputs:

- **An array of strings or integers** When the input format is ``Parquet``, the ``feature_col`` can store string or integer arrays. In this case, each row stores a string/integer array representing the hard negative node ids of the corresponding edge. For example, the ``feature_col`` can be a 2D string array, like ``[["e0_hard_0", "e0_hard_1"],["e1_hard_0"], ..., ["en_hard_0", "en_hard_1"]]`` or a 2D integer array (for integer node ids) like ``[[10,2],[3],...[4,12]]``. It is not required for each row to have the same dimension size. GraphStorm will automatically handle the case when some edges do not have enough pre-defined hard negatives.

- **A single string** The ``feature_col`` stores strings instead of string arrays. (When the input format is ``Parquet`` or ``CSV``) In this case, a ``separator`` must be provided to split the strings into node ids. The ``feature_col`` will be a 1D string list, for example ``["e0_hard_0;e0_hard_1", "e1_hard_1", ..., "en_hard_0;en_hard_1"]``. The string length, i.e., number of hard negatives, can vary from row to row. GraphStorm will automatically handle the case when some edges do not have enough hard negatives.

GraphStorm will automatically translate the Raw Node IDs of hard negatives into Partition Node IDs in a DistDGL graph.

Multiple Target Node Types Training
-------------------------------------

When training on a hetergenious graph, we often need to train a model by minimizing the objective function on more than one node type. GraphStorm provides supports to achieve this goal.

- Train on multiple node types: The users only need to edit the ``target_ntype`` in model config YAML file to minimize the objective function defined on mutiple target node types. For example, by setting ``target_ntype`` as following, we can jointly optimize the objective function defined on "movie" and "user" node types.

  .. code-block:: yaml

    target_ntype:
    -  movie
    -  user

  During evuation, the users can set a single node type for evaluation. For example, by setting ``eval_target_ntype:  movie``, we will only perform evaluation on "movie" node type.

- Evaluate on single node type: During evuation, the users can set a single node type for evaluation. For example, by setting ``eval_target_ntype:  movie``, we will only perform evaluation on "movie" node type. Our current implementation only support evaluating on a single node type.

- Per target node type decoder: The users may also want to use a different decoder on each node type, where the output dimension for each decoder maybe different. We can achieve this by setting ``num_classes`` in model config YAML file. For example, by setting ``num_classes`` as following, GraphStorm will create a decoder with output dimension as 3 for movie node type, and a decoder with output dimension as 7 for user node type.

  .. code-block:: yaml

    num_classes:
      movie:  3
      user:  7

- Reweighting on loss function: The users may also want to use a customized loss function reweighting on each node type, which can be achieved by setting ``multilabel``, ``multilabel_weights``, and ``imbalance_class_weights``. Examples are illustrated as following. Our current implementation does not support different node types with different ``multilabel`` setting.

  .. code-block:: yaml

    multilabel:
      movie:  true
      user:  true
    multilabel_weights:
      movie:  0.1,0.2,0.3
      user:  0.1,0.2,0.3,0.4,0.5,0.0

    multilabel:
      movie:  false
      user:  false
    imbalance_class_weights:
      movie:  0.1,0.2,0.3
      user:  0.1,0.2,0.3,0.4,0.5,0.0
