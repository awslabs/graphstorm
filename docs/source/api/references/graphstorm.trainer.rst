.. _apitrainer:

graphstorm.trainer
=====================

    GraphStorm training modules assemble the distributed training pipeline for different
    tasks, e.g., node classification and link prediction.

    If possible, users should always use these GraphStorm training modules to avoid handling
    the complexities involved with the distributed data loading and model training.

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
