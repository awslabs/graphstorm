.. _apitrainer:

graphstorm.trainer
=====================

    GraphStorm trainers assemble the distributed training pipeline for different tasks or
    different training methods.

    If possible, users should always use these trainers to avoid handling the distributed
    processing and tasks.

.. currentmodule:: graphstorm.trainer

Trainers
--------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: trainertemplate.rst

    GSgnnLinkPredictionTrainer
    GSgnnNodePredictionTrainer
    GSgnnEdgePredictionTrainer
    GLEMNodePredictionTrainer
