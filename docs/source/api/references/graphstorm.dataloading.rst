.. _apidataloading:

graphstorm.dataloading
==========================

    GraphStorm dataloading module includes a unified graph Data and a set of different
    DataLoaders for different graph machine learning tasks.

    If users would like to customize DataLoaders, please extend those dataloader base 
    classes in the **Base DataLoaders** section and customize their abstract methods.

.. currentmodule:: graphstorm.dataloading

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

    GSgnnData

DataLoaders
------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: dataloadertemplate.rst

    GSgnnNodeDataLoader
    GSgnnNodeSemiSupDataLoader
    GSgnnEdgeDataLoader
    GSgnnLinkPredictionDataLoader
    GSgnnLinkPredictionTestDataLoader
    GSgnnLinkPredictionPredefinedTestDataLoader
