.. _apidataloading:

graphstorm.dataloading
==========================

    GraphStorm dataloading module includes a set of graph DataSets and DataLoaders for different
    graph machine learning tasks.

    If users would like to customize DataLoaders, please extend those ``***DataLoaderBase`` classes
    and implement the abstract methods with customized means.

.. currentmodule:: graphstorm.dataloading

Base DataLoaders
-------------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: datatemplate.rst

    .. GSgnnNodeDataLoaderBase
    .. GSgnnEdgeDataLoaderBase
    .. GSgnnLinkPredictionDataLoaderBase

DataSets
------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: datatemplate.rst

    GSgnnNodeTrainData
    GSgnnNodeInferData
    GSgnnEdgeTrainData
    GSgnnEdgeInferData

DataLoaders
------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: datatemplate.rst

    GSgnnNodeDataLoader
    GSgnnEdgeDataLoader
    GSgnnLinkPredictionDataLoader
