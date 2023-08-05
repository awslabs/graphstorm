.. _advanced_usages:

GraphStorm Advanced Usages
===========================

Link prediction on GraphStorm 

Link prediction is widely employed as a pre-training technique to generate high-quality entity representations applicable to diverse business applications. However, implementing a training loop for link prediction tasks needs to carefully handle the information leakage problems caused by 1) including target edges in message passing, and 2) including validation/test edges in message passing during training. ([This paper](https://arxiv.org/pdf/2306.00899.pdf) provides more details.) GraphStorm provides supports to avoid theses problems:

* To avoid including target edges in message passing, we need to set ``exclude_training_targets`` to True and provide ``reverse_edge_types_map`` when launching link prediction training tasks. (See https://github.com/awslabs/graphstorm/wiki/configuration-configuration-run#configurations-run for more details.) These two arguments tell GraphStorm to exclude the training target edges and the corresponding reverse edges when doing message passing.
* To avoid including validation/test edges in message passing during model training, we need to mask validation edges and test edges with ``val_mask`` and ``test_mask`` respectively. We also need to mask all the other edges with ``train_mask``.

