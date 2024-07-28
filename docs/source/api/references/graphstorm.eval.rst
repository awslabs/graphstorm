.. _apieval:

graphstorm.eval
=======================

    GraphStorm provides built-in evaluators for different Graph Machine Learning tasks.
    Each evaluator can have multiple task associated metrics. For example, the
    ``GSgnnClassificationEvaluator`` uses ``accuracy`` as its default evaluation metric.
    Users can also specify other metrics for it, e.g., ``precision_recall``, ``roc_auc``,
    ``f1_score``, etc.
    
    Users can find all of the metrics for different tasks in the :ref:`Evaluation
    metrics <eval_metrics>` section.

    If users want to implement customized evaluators or evaluation methods, a best practice is to
    extend the base evaluator, i.e., the ``GSgnnBaseEvaluator``, and the corresponding evaluation
    interfaces, e.g., ``GSgnnPredictionEvalInterface`` for prediction evaluation, and
    ``GSgnnLPRankingEvalInterface`` for ranking based link prediction evaluation, and then
    implement the abstract methods defined in those interface classes.

.. currentmodule:: graphstorm.eval

Base Evaluators
----------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: evaltemplate.rst

    GSgnnBaseEvaluator
    GSgnnPredictionEvalInterface
    GSgnnLPRankingEvalInterface

Evaluators
-----------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: evaltemplate.rst

    GSgnnClassificationEvaluator
    GSgnnRegressionEvaluator
    GSgnnMrrLPEvaluator
    GSgnnPerEtypeMrrLPEvaluator
    GSgnnRconstructFeatRegScoreEvaluator
