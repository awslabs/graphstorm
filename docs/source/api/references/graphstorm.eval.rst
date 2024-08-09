.. _apieval:

graphstorm.eval
=======================

    GraphStorm provides built-in evaluators and interfaces for different Graph Machine Learning
    tasks. Each evaluator can have multiple task specific metrics for model evaluation. For
    example, ``GSgnnClassificationEvaluator`` uses ``accuracy`` as its default
    evaluation metric. However, users can also set other metrics, e.g., ``precision_recall``,
    ``roc_auc``, and ``f1_score`` in ``GSgnnClassificationEvaluator``.

    Users can find the information about metrics for different tasks in the :ref:`Evaluation
    Metrics <eval_metrics>` section.

    If users want to implement customized evaluators, a best practice is to extend the base
    evaluator, i.e., ``GSgnnBaseEvaluator``, and the corresponding evaluation
    interfaces, e.g., ``GSgnnPredictionEvalInterface`` for prediction evaluation and
    ``GSgnnLPRankingEvalInterface`` for ranking-based link prediction evaluation, and then
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
    GSgnnHitsLPEvaluator
    GSgnnPerEtypeHitsLPEvaluator
    GSgnnRconstructFeatRegScoreEvaluator
