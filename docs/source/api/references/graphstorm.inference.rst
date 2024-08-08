.. _apiinference:

graphstorm.inference
====================

    GraphStorm nference modules assemble the distributed inference pipelines for different
    tasks, e.g., node classification, and link prediction.

    If possible, users should always use these GraphStorm inference modules to avoid handling
    the complexities involved with the distributed data processing and model inference.

.. currentmodule:: graphstorm.inference

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: inferencetemplate.rst

    GSgnnNodePredictionInferrer
    GSgnnEdgePredictionInferrer
    GSgnnLinkPredictionInferrer
    GSgnnEmbGenInferer
