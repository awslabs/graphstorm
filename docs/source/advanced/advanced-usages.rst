.. _advanced_usages:

GraphStorm Advanced Usages
===========================

Prevent Information Leakage in Link Prediction
-----------------------------------------------

Link prediction is widely employed as a pre-training technique to generate high-quality entity representations applicable to diverse business applications. However, implementing a training loop for link prediction tasks needs to carefully handle the information leakage problems caused by 1) including target edges in message passing, and 2) including validation/test edges in message passing during training. (`This paper <https://arxiv.org/pdf/2306.00899.pdf>`_ provides more details.) 

GraphStorm provides supports to avoid theses problems:

* To avoid including target edges in message passing, users need to set ``exclude_training_targets`` to `True`` and provide ``reverse_edge_types_map`` when launching link prediction training tasks. These two arguments tell GraphStorm to exclude the training target edges and the corresponding reverse edges when doing message passing. More explanation of the two arguments can be found on the :ref:`Training and Inference Configurations<configurations-run>`.

* To avoid including validation/test edges in message passing during model training, users need to mask validation edges and test edges with ``val_mask`` and ``test_mask`` respectively. Users also need to mask all the other edges with ``train_mask``.

Speedup Link Prediction Training
---------------------------------------------
GraphStorm relies on ``dgl.dataloading.MultiLayerNeighborSampler`` and ``train_mask`` to avoid sampling validation and test edges during training. Basically, it only samples edges with ``train_mask`` set to be `True`. However, the implementation is not efficient. To speedup graph sampling during link prediction training, GraphStorm provides four link prediction dataloaders (i.e., ``fast_uniform``, ``fast_joint``, ``fast_localuniform`` and ``fast_localjoint``) with more efficient implementation but less precise neighbor sampling behavior. 

To be more specific, these dataloaders will do neighbor sampling regardless of any masks in the beginning, and later remove edges with  ``val_mask`` or ``test_mask`` set to be `True`. In theory, a sampled subgraph may have less neighbor nodes than expected as some of them would be removed. However, with a graph having hundreds of millions of edges (or more) and small validation and test sets, e.g., each with less than 10% edges, the impact is negligible. 

With DGL 1.0.4, ``fast_localuniform`` dataloader can speedup 2.4X over ``localuniform`` dataloader on training a 2 layer RGCN on MAG dataset on four g5.48x instances.
