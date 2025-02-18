.. _imbalanced_labels:

Deal with Imbalance Labels in Classification/Regression
=======================================================

In some cases, the number of labels of different classes could be imbalanced, i.e., some classes
have either too large or too small numbers. For example, most fraud detection tasks only have a
small number of fraudulent activities (positive labels) versus a huge number of legitimate activities
(negative labels). Even in regression tasks, it is possible to encounter many dominant values that
can cause imbalanced labels. If not handled properly, these imbalanced labels could impact classification/regression
model performance a lot. For example, because too many negative labels are fit into models, models
may learn to classify all unseen samples as negative. To tackle the imbalance label problem, GraphStorm
provides several built-in features.

For classification tasks, users can configure two arguments in command line interfaces (CLIs), the
``imbalance_class_weights`` and ``class_loss_func``.

The ``imbalance_class_weights`` allows users to give scale weights for each class, hence forcing models
to learn more on the classes with higher scale weight. For example, if there are 10 positive labels versus
90 negative labels, you can set ``imbalance_class_weights`` to be ``0.1, 0.9``, meaning class 0 (usually
for negative labels) has weight ``0.1``, and class 1 (usually for positive labels) has weight ``0.9``.
This helps models to detect more positive samples. Below is an example how to set the
``imbalance_class_weights`` in a YAML configuration file.

  .. code-block:: yaml

    imbalance_class_weights: 0.1,0.9

You can also set ``focal`` as the ``class_loss_func`` configuration's value, which will use the
`focal loss function <https://arxiv.org/abs/1708.02002>`_ in binary classification tasks. The focal loss
function is designed for imbalanced classes. Its formula is :math:`loss(p_t) = -\alpha_t(1-p_t)^{\gamma}log(p_t)`,
where :math:`p_t=p`, if :math:`y=1`, otherwise, :math:`p_t = 1-p`. Here :math:`p` is the probability of output
in a binary classification. This function has two hyperparameters, :math:`\alpha` and :math:`\gamma`,
corresponding to the ``alpha`` and ``gamma`` configuration in GraphStorm. Larger values of ``gamma`` will help
update models on harder cases so as to detect more positive samples if the positive to negative ratio is small.
There is no clear guideline for values of ``alpha``. You can use its default value(``0.25``) first, and then
search for optimal values. Below is an example how to set the `focal loss funciton` in a YAML configuration file.

  .. code-block:: yaml

    class_loss_func: focal

    gamma: 10.0
    alpha: 0.5

Besides the two configurations, you can output the classification results as probabilities of positive and negative
classes by setting the value of ``return_proba`` configuration to be ``true``. By default GraphStorm output
classification results using the argmax values, e.g., either 0s or 1s in binary tasks, which equals to using
``0.5`` as the threshold to classify negative from positive samples. With probabilities as outputs, you can use
different thresholds, hence being able to achieve desired outcomes. For example, if you need higher recall to catch
more suspicious positive samples, a smaller threshold, e.g., "0.25", could classify more positive cases. Or you may
use methods like `ROC curve` or `Precision-Recall curve` to determine the optimal threshold. Below is an example how
to set the ``return_proba`` in a YAML configuration file.

  .. code-block:: yaml

    return_proba: true

For regression tasks where there are some dominant values, e.g., 0s, in labels, GraphStorm provides the
`shrinkage loss function <https://openaccess.thecvf.com/content_ECCV_2018/html/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.html>`_,
which can be set by using ``shrinkage`` as value of the ``regression_loss_func`` configuration. Its formula is
:math:`loss = l^2/(1 + \exp \left( \alpha \cdot (\gamma - l)\right))`, where :math:`l` is the absolute difference
between predictions and labels. The shrinkage loss function also has the :math:`\alpha` and :math:`\gamma` hyperparameters.
You can use the same ``alpha`` and ``gamma`` configuration as the focal loss function to modify their values. The shrinkage
loss penalizes the importance of easy samples (when :math:`l < 0.5`) and keeps the loss of hard samples unchanged. Below is
an example how to set the `shrinkage loss function` in a YAML configuration file.

  .. code-block:: yaml

    regression_loss_func: shrinkage

    gamma: 0.2
    alpha: 5
