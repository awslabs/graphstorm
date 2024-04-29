.. _apieval:

graphstorm.eval
=======================

    GraphStorm provides built-in evaluation methods for different Graph Machine
    Learning (GML) tasks.

    If users want to implement customized evaluators or evaluation methods, a best practice is to
    extend the base evaluator, i.e., the ``GSgnnBaseEvaluator``, and the corresponding evaluation
    interfaces, e.g., ``GSgnnPredictionEvalInterface``` for prediction evaluation, and
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
