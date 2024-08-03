.. _apitrainer:

graphstorm.trainer
=====================

    GraphStorm Trainers assemble the distributed training pipeline for different tasks,
    e.g., node classification, link prediction, and multi-task learning by using
    different training methods such as mini-batch sampling or layer-wise sampling.

    If possible, users should always use these trainers to avoid handling the complexities
    involved with the distributed data and model processing tasks.

.. currentmodule:: graphstorm.trainer

Trainers
--------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: trainertemplate.rst

    GSgnnNodePredictionTrainer
    GSgnnEdgePredictionTrainer
    GSgnnLinkPredictionTrainer
    GLEMNodePredictionTrainer
