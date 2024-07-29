.. _apidataloading:

graphstorm.dataloading.dataset
===============================

    GraphStorm dataset provides one unified dataset class, i.e., ``GSgnnData``, for all graph
    machine learning tasks. Users can build a ``GSgnnData`` object by giving the path of
    the JSON file created by the :ref:`GraphStorm Graph Construction<graph_construction>`
    operations. The ``GSgnnData`` will load the related graph artifacts specified in the JSON
    file. It provides a set of APIs for users to extract information of the graph data for
    model training and inference.

.. currentmodule:: graphstorm.dataloading

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: datasettemplate.rst

    GSgnnData

graphstorm.dataloading.dataloading
===================================

    GraphStorm dataloading module includes a set of different DataLoaders for
    different graph machine learning tasks.

    If users would like to customize DataLoaders, please extend those dataloader base 
    classes in the **Base DataLoaders** section and customize their abstract functions.

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
