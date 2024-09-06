.. _apimodel:

graphstorm.model
===================

    GraphStorm provides a set of Graph Neural Network (GNN) modules. By combining them
    in proper ways, users can build various GNN models for different tasks.

    A GNN model in GraphStorm normally contains four components:

    * Input layer: an input encoder that converts input node/edge features into embeddings
      with the given hidden dimensions. The output of an input layer will become the input of
      the GNN layer, or the decoder layer if GNN is not defined.
    * GNN layer (Optional): a GNN encoder that performs the message passing computation.
      The outputs of a GNN layer are embeddings of nodes that wil be used in the decoder layer.
    * Decoder layer: a task specific decoder that converts results from either a GNN layer or
      an input layer into loss values for different GML tasks, e.g., classification, regression,
      or link prediction.
    * Model optimizer: GraphStorm model classes have a built-in model optimizer, which should
      be initialized during GraphStorm GNN model object construction.

    If users would like to implement their own GNN models, a suggested practice is to extend a
    base GNN model class and its corresponding interface, e.g., ``GSgnnNodeModelBase`` and
    ``GSgnnNodeModelInterface``, and implement the required abstract methods.

    If users just want to build their own message passing methods, a suggested practice is to
    create their own GNN encoders by extending the ``GraphConvEncoder`` base class, and
    implement the ``forward(self, blocks, h)`` function, which will be called by GraphStorm GNN
    model classes within their own ``forward()`` function.

    For examples of how to use these GraphStorm APIs to form training/inference pipelines,
    to switch different GNN encoders to implement various GNN models, and to build a customized
    GNN encoder, please refer to
    :ref:`GraphStorm API Programming Examples <programming-examples>`.

.. currentmodule:: graphstorm.model

Base GNN models
---------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: modeltemplate.rst

    GSgnnModelBase
    GSgnnNodeModelBase
    GSgnnNodeModelInterface
    GSgnnEdgeModelBase
    GSgnnEdgeModelInterface
    GSgnnLinkPredictionModelBase
    GSgnnLinkPredictionModelInterface

Input Layer
----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: modeltemplate.rst

    GSNodeEncoderInputLayer
    GSLMNodeEncoderInputLayer
    GSPureLMNodeInputLayer

GNN Layer
----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: modeltemplate.rst

    RelGraphConvLayer
    RelationalGCNEncoder
    RelationalAttLayer
    RelationalGATEncoder
    HGTLayer
    HGTEncoder
    SAGEConv
    SAGEEncoder
    GATConv
    GATEncoder
    GATv2Conv
    GATv2Encoder

Decoder Layer
----------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: modeltemplate.rst

    EntityClassifier
    EntityRegression
    DenseBiDecoder
    MLPEdgeDecoder
    MLPEFeatEdgeDecoder
    LinkPredictMultiRelationLearnableDecoder
    LinkPredictDotDecoder
    LinkPredictContrastiveDotDecoder
    LinkPredictDistMultDecoder
    LinkPredictContrastiveDistMultDecoder
    LinkPredictRotatEDecoder
    LinkPredictContrastiveRotatEDecoder
