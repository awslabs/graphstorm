.. _advanced_usages:

GraphStorm Advanced Usages
===========================

Deal with Imbalance Labels in Classification/Regression
---------------------------------------------------------

In some cases, the number of labels of different classes could be imbalanced, i.e., some classes have either too large or too small numbers. For example, most fraud detection tasks only have a small number of fraudulent activities (positive labels) versus a huge number of legitimate activities (negative labels). Even in regression tasks, it is possible to encounter many dominant values that can cause imbalanced labels. If not handle properly, these imbalanced labels could impact classification/regression model performance a lot. For example, because too many negative labels are fit into models, models may learn to classify all unseen samples as negative. To tackle the imbalance label problem, GraphStorm provides several built-in features.

For classification tasks, users can configure two arguments in command line interfaces (CLIs), the ``imbalance_class_weights`` and ``class_loss_func``.

The ``imbalance_class_weights`` allows users to give scale weights for each class, hence forcing models to learn more on the classes with higher scale weight. For example, if there are 10 positive labels versus 90 negative labels, you can set ``imbalance_class_weights`` to be ``0.1, 0.9``, meaning class 0 (usually for negative labels) has weight ``0.1``, and class 1 (usually for positive labels) has weight ``0.9``. This help models to be able to detect more positive samples.

You can also set ``focal`` as the ``class_loss_func`` configuration's value, which will use the `focal loss function <https://arxiv.org/abs/1708.02002>`_ in binary classification tasks. The focal loss function is designed for imbalanced classes. Its formula is :math:`loss(p_t) = -\alpha_t(1-p_t)^{\gamma}log(p_t)`, where :math:`p_t = p` if :math:`y=1`, otherwise :math:`p_t = 1-p`. Here :math:`p` is the probability of output in a binary classification. This function has two hyperparameters, :math:`\alpha` and :math:`\gamma`, corresponding to the ``alpha`` and ``gamma`` configuration in GraphStorm. Larger values of ``gamma`` will help update models on harder cases so as to detect more positive samples if the positive to negative ratio is small. There is no clear guideline for values of ``alpha``. You can use its default value(``0.25``) first, and then search for optimal values.

Besides the two configurations, you can output the classification results as probabilities of positive and negative classes by setting the value of ``return_prob`` configuration to be ``true``. By default GraphStorm output classification results using the argmax values, e.g., either 0s or 1s in binary tasks, which equals to using ``0.5`` as the threshold to classify negative from positive samples. With probabilities as outputs, you can use different thresholds, hence being able to achieve desired outcomes. For example, if you need higher recall to catch more suspecious positive samples, a smaller threshold, e.g., 0.25, could classify more positive cases. Or you may use methods like `ROC curve` or `Precision-Recall curve` to determine the optimal threshold.

For regression tasks where there are some dominant values, e.g., 0s, in labels, GraphStorm provides the `shrinkage loss function <https://openaccess.thecvf.com/content_ECCV_2018/html/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.html>`_, which can be set by using ``shrinkage`` as value of the ``regression_loss_func`` configuration. Its formula is :math:`loss = l^2/(1 + \exp \left( \alpha \cdot (\gamma - l)\right))`, where :math:`l` is the absolute difference between predictions and labels. The shrinkage loss function also has the :math:`\alpha` and :math:`\gamma` hyperparameters. You can use the same ``alpha`` and ``gamma`` configuration as the focal loss function to modify their values. The shrinkage loss only penalizes the importance
of easy samples (when :math:`l < 0.5``) and keeps the loss of hard samples unchanged.

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
