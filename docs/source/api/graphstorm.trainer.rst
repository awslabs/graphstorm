.. _apitrainer:

graphstorm.trainer
=====================

    GraphStorm trainers assemble the distributed training pipeline for different tasks or
    different training methods.

    If possible, users should always use these trainers to avoid handling the distributed
    processing and tasks.

.. currentmodule:: graphstorm.trainer


Base class
--------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    GSgnnTrainer

Task classes
-----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    GSgnnLinkPredictionTrainer
    GSgnnNodePredictionTrainer
    GSgnnEdgePredictionTrainer

Method classes
-----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    GLEMNodePredictionTrainer
