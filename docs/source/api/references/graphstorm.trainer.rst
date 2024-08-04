.. _apitrainer:

graphstorm.trainer
=====================

    GraphStorm Trainers assemble the distributed training pipeline for different tasks,
    e.g., node classification, and link prediction by using different training methods
    such as mini-batch sampling or layer-wise sampling.

    If possible, users should always use these Trainers to avoid handling the complexities
    involved with the distributed data processing and model training.

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
