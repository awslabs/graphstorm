.. _apidataloading:

graphstorm.dataloading
==========================

    GraphStorm dataloading module includes a set of graph DataSets and DataLoaders for different
    graph machine learning tasks.

    If users would like to customize DataLoaders, please extend those classes in the
    :ref:`Base DataLoaders <basedataloaders>` section and customize their abstract methods.

.. currentmodule:: graphstorm.dataloading

.. _basedataloaders:

Base DataLoaders
-------------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: dataloadertemplate.rst

    GSgnnNodeDataLoaderBase
    GSgnnEdgeDataLoaderBase
    GSgnnLinkPredictionDataLoaderBase

DataSets
------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: datasettemplate.rst

    GSgnnNodeTrainData
    GSgnnNodeInferData
    GSgnnEdgeTrainData
    GSgnnEdgeInferData

DataLoaders
------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: dataloadertemplate.rst

    GSgnnNodeDataLoader
    GSgnnEdgeDataLoader
    GSgnnLinkPredictionDataLoader
