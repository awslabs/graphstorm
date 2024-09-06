.. _link_prediction_usage:

Link Prediction Learning in GraphStorm
=========================================
Link prediction is widely used in the industry as a pre-training method to produce high-quality entity representations. However, performing link
prediction training on large graphs is non-trivial both in terms of model
performance and efficiency. GraphStorm offers a wide array of options for users
to customize their link prediction model training from both model performance
and training efficiency standpoints.

Optimizing model performance
----------------------------
GraphStorm incorporates three ways of improving model performance of link
prediction. Firstly, GraphStorm avoids information leak in model training.
Secondly, to better handle heterogeneous graphs, GraphStorm provides three ways
to compute link prediction scores: dot product, DistMult and RotatE.
Thirdly, GraphStorm provides two options to compute training losses, i.e.,
cross entropy loss and contrastive loss. The following sub-sections provide more details.

Prevent Information Leakage in Link Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implementing a training loop for link prediction tasks needs to carefully handle the information leakage problems caused by 1) including target edges in message passing, and 2) including validation/test edges in message passing during training. (`This paper <https://arxiv.org/pdf/2306.00899.pdf>`_ provides more details.)

GraphStorm provides supports to avoid theses problems:

* To avoid including target edges in message passing, users need to set ``exclude_training_targets`` to `True` and provide ``reverse_edge_types_map`` when launching link prediction training tasks. These two arguments tell GraphStorm to exclude the training target edges and the corresponding reverse edges when doing message passing. More explanation of the two arguments can be found on the :ref:`Training and Inference Configurations<configurations-run>`.

* To avoid including validation/test edges in message passing during model training, users need to mask validation edges and test edges with ``val_mask`` and ``test_mask`` respectively. Users also need to mask all the other edges with ``train_mask``.


.. _link-prediction-score-func:

Computing Link Prediction Scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GraphStorm provides three ways to compute link prediction scores: Dot Product, DistMult and RotatE.

* **Dot Product**: The Dot Product score function is as:

    .. math::
            score = sum(head\_emb * tail\_emb)

    where the ``head_emb`` is the node embedding of the head node and
    the ``tail_emb`` is the node embedding of the tail node.

* **DistMult**: The DistMult score function is as:

    .. math::
        score = sum(head\_emb * relation\_emb * tail\_emb)

    where the ``head_emb`` is the node embedding of the head node,
    the ``tail_emb`` is the node embedding of the tail node and
    the ``relation_emb`` is the relation embedding of the specific edge type.
    The ``relation_emb`` values are initialized from a uniform distribution
    within the range of ``(-gamma/hidden_size, gamma/hidden_size)``,
    where ``gamma`` and ``hidden_size`` are hyperparameters defined in
    :ref:`Model Configurations<configurations-model>`。

* **RotatE**: The RotatE score function is as:

    .. math::
        score = gamma - \|head\_emb \circ relation\_emb - tail\_emb\|^2

    where the ``head_emb`` is the node embedding of the head node,
    the ``tail_emb`` is the node embedding of the tail node,
    the ``relation_emb`` is the relation embedding of the specific edge type,
    and :math:`\circ` is the element-wise product.
    The ``relation_emb`` values are initialized from a uniform distribution
    within the range of ``(-gamma/(hidden_size/2), gamma/(hidden_size/2))``,
    where ``gamma`` and ``hidden_size`` are hyperparameters defined in
    :ref:`Model Configurations<configurations-model>`.
    To learn more information about RotatE, please refer to `the DGLKE doc <https://dglke.dgl.ai/doc/kg.html#rotatee>`__.

.. _link_prediction_loss:

Link Prediction Loss Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GraphStorm provides four options to compute training losses:

* **Cross Entropy Loss**: The cross entropy loss turns a link prediction task into a binary classification task. We treat positive edges as 1 and negative edges as 0. The loss of an edge ``e`` is as:

    .. math::

        loss = - y \cdot \log score + (1 - y) \cdot \log (1 - score)

    where ``y`` is 1 when ``e`` is a positive edge and 0 when it is a negative edge. ``score`` is the score value of ``e`` computed by the score function.

* **Weighted Cross Entropy Loss**: The weighted cross entropy loss is similar to **Cross Entropy Loss** except that it allows users to set a weight for each positive edge. The loss function of an edge ``e`` is as:

    .. math::

        loss = - w\_e \left[ y \cdot \log score + (1 - y) \cdot \log (1 - score) \right]

    where ``y`` is 1 when ``e`` is a positive edge and 0 when it is a negative edge. ``score`` is the score value of ``e`` computed by the score function, ``w_e`` is the weight of ``e`` and is defined as

    .. math::

        w\_e = \left \{
        \begin{array}{lc}
            1,  & \text{ if } e \in G, \\
            0,  & \text{ if } e \notin G
        \end{array}
        \right.

    where ``G`` is the training graph.


* **Adversarial Cross Entropy Loss**: The adversarial cross entropy loss turns a link prediction task into a binary classification task. We treat positive edges as 1 and negative edges as 0. In addition, adversarial cross-entropy loss adjusts the loss value of each negative sample based on its degree of difficulty. This is enabled by setting the ``adversarial_temperature`` config.

    The loss of positive edges is as:

    .. math::

        loss_{pos} = - \log score

    where ``score`` is the score value of the positive edges computed by the score function.

    The loss of negative edges is as:

    .. math::

        \begin{gather*}
        loss_{neg} = \log (1 - score) \\
        loss_{neg} = \mathrm{softmax}(score * adversarial\_temperature) * loss_{neg}
        \end{gather*}

    where ``score`` is the score value of the negative edges computed by the score function and ``adversarial_temperature`` is a hyper-parameter.

    The final loss is as:

    .. math::

        loss = \dfrac{\mathrm{avg}(loss_{pos}) + \mathrm{avg}(loss_{neg})}{2}

* **Weighted Adversarial Cross Entropy Loss**  The weighted cross entropy loss is similar to **Adversarial Cross Entropy Loss** except that it allows users to set a weight for each positive edge. The loss function of a positive edge ``e`` is as:

    .. math::

        loss_{pos} = - w * \log score

    where ``score`` is the score value of the positive edges computed by the score function, ``w`` is the weight of each positive edge. The loss of the negative edges is the same as **Adversarial Cross Entropy Loss**.

    The final loss is as:

    .. math::

        loss = \dfrac{\mathrm{avg}(loss_{pos}) + \mathrm{avg}(loss_{neg})}{2}

* **Contrastive Loss**: The contrastive loss compels the representations of connected nodes to be similar while forcing the representations of disconnected nodes remains dissimilar. In the implementation, we use the score computed by the score function to represent the distance between nodes. When computing the loss, we group one positive edge with the ``N`` negative edges corresponding to it.The loss function is as follows:

    .. math::

        loss = -\log \left( \dfrac{\exp(pos\_score)}{\sum_{i=0}^N \exp(score\_i)} \right)

    where ``pos_score`` is the score of the positive edge. ``score_i`` is the score of the i-th edge. In total, there are ``N+1`` edges, within which there is 1 positive edge and ``N`` negative edges.

Selecting the Negative Sampler
------------------------------
GraphStorm provides a wide list of negative samplers:

* **Uniform negative sampling**: Given ``N`` training edges under edge type ``(src_t, rel_t, dst_t)`` and the number of negatives set to ``K``, uniform negative sampling randomly samples ``K`` nodes from ``dst_t`` for each training edge. It corrupts the training edge to form ``K`` negative edges by replacing its destination node with sampled negative nodes. In total, it will sample ``N * K`` negative nodes.

    * ``uniform``: Uniformly sample ``K`` negative edges for each positive edge.

    * ``fast_uniform``: same as ``uniform`` except that the sampled subgraphs
    will not exclude edges with ``val_mask`` and ``test_mask``.

    * ``all_etype_uniform``: same as ``uniform``, but it ensures that each
    training edge type appears in every mini-batch.

* **Local uniform negative sampling**: Local uniform negative sampling samples negative edges in the same way as uniform negative sampling except that all the negative nodes are sampled from the local graph partition.

    * ``localuniform``: Uniformly sample ``K`` negative edges for each positive edge.
    However the negative nodes are sampled from the local graph partition
    instead of being sampled globally.

    * ``fast_localuniform``: same as ``localuniform`` except that the sampled subgraphs
    will not exclude edges with ``val_mask`` and ``test_mask``. Please see the details in :ref:`speedup_lp_training_label`.

* **Joint negative sampling**: Given ``N`` training edges under edge type ``(src_t, rel_t, dst_t)`` and the number of negatives set to ``K``, joint negative sampling randomly samples ``K`` nodes from ``dst_t`` for every ``K`` training edges. For these ``K`` training edges, it corrupts each edge to form ``K`` negative edges by replacing its destination node with the same set of negative nodes. In total, it only needs to sample $N$ negative nodes. (We suppose ``N`` is dividable by ``K`` for simplicity.)

    * ``joint``: Sample ``K`` negative nodes for every ``K`` positive edges.
    The ``K`` positive edges will share the same set of negative nodes

    * ``fast_joint``: same as ``joint`` except that the sampled subgraphs
    will not exclude edges with ``val_mask`` and ``test_mask``.
    Please see the details in :ref:`speedup_lp_training_label`.

    * ``all_etype_joint``: same as ``joint``, but it ensures that each
    training edge type appears in every mini-batch.

* **Local joint negative sampling**: Local joint negative sampling samples negative edges in the same way as joint negative sampling except that all the negative nodes are sampled from the local graph partition.

    * ``localjoint``: Sample ``K`` negative nodes for every ``K`` positive edges.
    However the negative nodes are sampled from the local graph partition
    instead of being sampled globally.

    * ``fast_localjoint``: same as ``localjoint`` except that the sampled subgraphs
    will not exclude edges with ``val_mask`` and ``test_mask``.

* **In-batch negative sampling**: In-batch negative sampling creates negative edges by exchanging destination nodes between training edges. For example, suppose there are three training edges ``(u_1, v_1), (u_2, v_2), (u_3, v_3)``, In-batch negative sampling will create two negative edges ``(u_1, v_2)`` and ``(u_1, v_3)`` for ``(u_1, v_1)``, two negative edges ``(u_2, v_1)`` and ``(u_2, v_3)`` for ``(u_2, v_2)`` and two negative edges ``(u_3, v_1)`` and ``(u_3, v_2)`` for ``(u_3, v_3)``. If the batch size is smaller than the number of negatives, either of the above three negative sampling methods can be used to sample extra negative edges.

    * ``inbatch_joint``: In-batch joint negative sampling.

.. _speedup_lp_training_label:

Speedup Link Prediction Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GraphStorm relies on ``dgl.dataloading.MultiLayerNeighborSampler`` and
``train_mask`` to avoid sampling validation and test edges during training.
Basically, it only samples edges with ``train_mask`` set to be `True`. However,
the implementation is not efficient. To speedup graph sampling during link
prediction training, GraphStorm provides four link prediction dataloaders
(i.e., ``fast_uniform``, ``fast_joint``, ``fast_localuniform`` and
``fast_localjoint``) with more efficient implementation but less precise
neighbor sampling behavior. To be more specific, these dataloaders will do
neighbor sampling regardless of any masks in the beginning, and later remove
edges with  ``val_mask`` or ``test_mask`` set to be `True`. In theory, a sampled
subgraph may have less neighbor nodes than expected as some of them would be
removed. However, with a graph having hundreds of millions of edges (or more)
and small validation and test sets, e.g., each with less than 10% edges, the
impact is negligible.

With DGL 1.0.4, ``fast_localuniform`` dataloader can speedup 2.4X over ``localuniform`` dataloader on training a 2 layer RGCN on MAG dataset on four g5.48x instances.

Hard Negative sampling
-----------------------
GraphStorm provides support for users to define hard negative edges for a positive edge during Link Prediction training.
Currently, hard negative edges are constructed by replacing the destination nodes of edges with pre-defined hard negatives.
For example, given an edge (``src_pos``, ``dst_pos``) and its hard negative destination nodes ``hard_0`` and ``hard_1``, GraphStorm will construct two hard negative edges, i.e., (``src_pos``, ``hard_0``) and (``src_pos``, ``hard_1``).

The hard negatives are stored as edge features of the target edge type.
Users can provide the hard negatives for each edge type through ``train_etypes_negative_dstnode`` in the training config yaml.
For example, the following yaml block defines the hard negatives for edge type ``(src_type,rel_type0,dst_type)`` as the edge feature ``negative_nid_field_0`` and the hard negatives for edge type ``(src_type,rel_type1,dst_type)`` as the edge feature ``negative_nid_field_1``.

  .. code-block:: yaml

    train_etypes_negative_dstnode:
      - src_type,rel_type0,dst_type:negative_nid_field_0
      - src_type,rel_type1,dst_type:negative_nid_field_1

Users can also define the number of hard negatives to sample for each edge type during training though ``num_train_hard_negatives`` in the training config yaml.
For example, the following yaml block defines the number of hard negatives for edge type ``(src_type,rel_type0,dst_type)`` is 5 and the number of hard negatives for edge type ``(src_type,rel_type1,dst_type)`` is 10.

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

**Preparing graph data for hard negative sampling**

The gconstruct pipeline of GraphStorm provides support to load hard negative data from raw input.
Hard destination negatives can be defined through ``edge_dst_hard_negative`` transformation.
The ``feature_col`` field of ``edge_dst_hard_negative`` must stores the raw node ids of hard destination nodes.
The follwing example shows how to define a hard negative feature for edges with the relation ``(node1, relation1, node1)``:

  .. code-block:: json

    {
        ...
        "edges": [
            ...
            {
                "source_id_col":    "src",
                "dest_id_col":      "dst",
                "relation": ("node1", "relation1", "node1"),
                "format":   {"name": "parquet"},
                "files":    "edge_data.parquet",
                "features": [
                    {
                        "feature_col": "hard_neg",
                        "feature_name": "hard_neg_feat",
                        "transform": {"name": "edge_dst_hard_negative",
                                                "separator": ";"},
                    }
                ]
            }
        ]
    }

The hard negative data is stored in the column named ``hard_neg`` in the ``edge_data.parquet`` file.
The edge feature to store the hard negative will be ``hard_neg_feat``.

GraphStorm accepts two types of hard negative inputs:

- **An array of strings or integers** When the input format is ``Parquet``, the ``feature_col`` can store string or integer arrays. In this case, each row stores a string/integer array representing the hard negative node ids of the corresponding edge. For example, the ``feature_col`` can be a 2D string array, like ``[["e0_hard_0", "e0_hard_1"],["e1_hard_0"], ..., ["en_hard_0", "en_hard_1"]]`` or a 2D integer array (for integer node ids) like ``[[10,2],[3],...[4,12]]``. It is not required for each row to have the same dimension size. GraphStorm will automatically handle the case when some edges do not have enough pre-defined hard negatives.
For example, the file storing hard negatives should look like the following:

.. code-block:: yaml

      src    |   dst    | hard_neg
    "src_0"  | "dst_0"  | ["dst_10", "dst_11"]
    "src_0"  | "dst_1"  | ["dst_5"]
    ...
    "src_100"| "dst_41" | [dst0, dst_2]

- **A single string** The ``feature_col`` stores strings instead of string arrays (When the input format is ``Parquet`` or ``CSV``). In this case, a ``separator`` must be provided int the transformation definition to split the strings into node ids. The ``feature_col`` will be a 1D string list, for example ``["e0_hard_0;e0_hard_1", "e1_hard_1", ..., "en_hard_0;en_hard_1"]``. The string length, i.e., number of hard negatives, can vary from row to row. GraphStorm will automatically handle the case when some edges do not have enough hard negatives.
For example, the file storing hard negatives should look like the following:

.. code-block:: yaml

      src    |   dst    | hard_neg
    "src_0"  | "dst_0" | "dst_10;dst_11"
    "src_0"  | "dst_1" | "dst_5"
    ...
    "src_100"| "dst_41"| "dst0;dst_2"

GraphStorm will automatically translate the Raw Node IDs of hard negatives into Partition Node IDs in a DistDGL graph.
