.. _advanced_usages:

GraphStorm Advanced Usages
===========================

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
