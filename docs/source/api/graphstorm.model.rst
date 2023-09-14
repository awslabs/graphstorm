.. _apimodel:

graphstorm.model
=================

    A GraphStorm model normally contains three components:

    * Input layer: a set of modules to convert input data for different use cases,
      e.g., embedding texture features.
    * Encoder: a set of Graph Neural Network modules 
    * Decoder: a set of modules to convert results from encoders for different tasks,
      e.g., classification, regression, or link prediction.

.. currentmodule:: graphstorm.model

Model input layers
-------------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    GSNodeEncoderInputLayer
    GSLMNodeEncoderInputLayer
    GSPureLMNodeInputLayer

Model encoders and layers
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