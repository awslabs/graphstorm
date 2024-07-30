.. _apimodel:

graphstorm.model
===================

    GraphStorm provides a set of Graph Neural Network (GNN) modules. By combining them
    in proper ways, users can build various GNN models for different tasks.

    A GraphStorm GNN model normally contains four components:

    * Input layer: an input encoder that converts input node/edge features into embeddings
      with the given hidden dimensions. The output of an input layer will become the input of
      the GNN layer, or the decoder layer if no need of GNN computation.
    * GNN layer: a GNN encoders that performs the message passing computation. The outputs of
      a GNN layer are embeddings of nodes that wil be used in the decoder layer.
    * Decoder layer: a task specific module that converts results from either a GNN layer or 
      an input layer for different GML tasks, e.g., classification, regression, or
      link prediction.
    * Model optimizer: GraphStorm model classes have a built-in model optimizer, which should
      be initialized during model object construction.

    Currently GraphStorm releases the APIs of the three layers.

    If users would like to implement their own GNN models, a best practice is to extend the
    a base model class and its corresponding interface, e.g., ``GSgnnModelBase`` and
    ``GSgnnNodeModelInterface``, and implement the required abstract methods.

    If users just want to build their own message passing methods, a best practice is to create
    their own GNN encoders by extending the ``GraphConvEncoder`` base class, and implementing
    the ``forward(self, blocks, h)`` function, which will be called by GraphStorm model classes
    within their own ``forward()`` function.

    For examples of how to use the GraphStorm APIs, please refer to GraphStorm :ref:`API
    Programming Examples <programming-examples>`.

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

Input Layer
----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    GSNodeEncoderInputLayer
    GSLMNodeEncoderInputLayer
    GSPureLMNodeInputLayer

GNN Layer
----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    RelationalGCNEncoder
    RelGraphConvLayer
    RelationalGATEncoder
    RelationalAttLayer
    HGTEncoder
    HGTLayer
    SAGEEncoder
    SAGEConv
    GATEncoder
    GATConv
    GATv2Encoder
    GATv2Conv

Decoder Layer
----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    EntityClassifier
    EntityRegression
    DenseBiDecoder
    MLPEdgeDecoder
    MLPEFeatEdgeDecoder
    LinkPredictDotDecoder
    LinkPredictDistMultDecoder
    LinkPredictWeightedDotDecoder
    LinkPredictWeightedDistMultDecoder
    LinkPredictContrastiveDotDecoder
    LinkPredictContrastiveDistMultDecoder
