.. _apimodel:

graphstorm.model
===================

    A GraphStorm model may contain three components:

    * Input layer: a set of modules to convert input data for different use cases,
      e.g., embedding texture features.
    * Encoder: a set of Graph Neural Network modules 
    * Decoder: a set of modules to convert results from encoders for different tasks,
      e.g., classification, regression, or link prediction.

    Currently GraphStorm releases the first two set of components.

    If users would like to implement their own model, the best practice is to extend the corresponding ``***ModelBase``, and implement the abstract methods.

.. currentmodule:: graphstorm.model

Base models
------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: modeltemplate.rst

    GSgnnNodeModelBase
    GSgnnEdgeModelBase
    GSgnnLinkPredictionModelBase

Input Layers
-------------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    GSNodeEncoderInputLayer
    GSLMNodeEncoderInputLayer
    GSPureLMNodeInputLayer

Encoders and GNN Layers
--------------------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    RelationalGCNEncoder
    RelGraphConvLayer
    RelationalGATEncoder
    RelationalAttLayer
    SAGEEncoder
    SAGEConv
    HGTEncoder
    HGTLayer