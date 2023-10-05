.. _apieval:

graphstorm.eval
=======================

    GraphStorm provides built-in evaluation methods for different Graph Machine
    Learning (GML) tasks.

    If users want to implement customized evaluators or evaluation methods, a best practice is to
    extend base evaluators, i.e., the ``GSgnnInstanceEvaluator`` class for node or edge prediction
    tasks, and ``GSgnnLPEvaluator`` for link prediction tasks, and then implement the abstract methods.

.. currentmodule:: graphstorm.eval

Base Evaluators
----------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: evaltemplate.rst

    GSgnnInstanceEvaluator
    GSgnnLPEvaluator

Evaluators
-----------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: evaltemplate.rst

    GSgnnLPEvaluator
    GSgnnMrrLPEvaluator
    GSgnnPerEtypeMrrLPEvaluator
    GSgnnAccEvaluator
    GSgnnRegressionEvaluator
