.. _programming-examples:

API Programming Examples
=========================

Besides the Command Line Interfaces (CLIs), GraphStorm releases a set of Application Programming Interfaces (APIs) for users to build their own Graph Machine Learning (GML) models and pipelines. To help users to better use these APIs, we provide a set of Jupyter notebooks as examples. Users can download these notebooks and related python files, and then run them locally in the GraphStorm standalone mode.

.. note:: All runnable Jupyter notebooks and related python files can be downloaded from the `GraphStorm Github notebook repository <https://github.com/awslabs/graphstorm/tree/main/docs/source/api/notebooks>`_.

These notebooks all use the same ACM data as discussed in the :ref:`User Your Own Data Tutorial<use-own-data>`. Users can create the required dataset and explore details of it by running the `Notebook 0: Data Preparation <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_0_Data_Prepare.ipynb>`_.

The `Notebook 1: Use GraphStorm APIs for Building a Node Classification Pipeline <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_1_NC_Pipeline.ipynb>`_ provides an example that demonstrates how to reproduce the node classification training and inference pipeline that is identical as the :ref:`Step 3: Launch training and inference scripts on your own graphs<launch_training_oyog>` does. While the `Notebook 2: Use GraphStorm APIs for Building a Link Prediction Pipeline <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_2_LP_Pipeline.ipynb>`_ provides another example that use the same dataset, but conduct pipelines for link prediction training and inference. 

GraphStorm provides a variaty of built-in neural network model components that can be easily combined to form different GNN models, e.g., RGCN and HGT. `Notebook 3: Use GraphStorm APIs for Implementing Built-in GNN Models <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_3_Model_Variants.ipynb>`_. demostrates how to leverage the built-in neural network components through GraphStorm APIs to implement models like RGCN, RGAT and HGT.

Besides built-in model components, following GraphStorm APIs users can develop customized components for their specific requirements. `Notebook 4: Use GraphStorm APIs for Customizing Model Components <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_4_Customized_Models.ipynb>`_ provides an example of a customized `RGAT` encoder and how to integrate it into the GraphStorm model architecture.

These examples utilize GraphStorm APIs, such as ``graphstorm``, ``graphstorm.dataloading.GSgnnDataset``, ``graphstorm.dataloading.GSgnnNodeDataLoader``, ``graphstorm.trainer.GSgnnNodePredictionTrainer``, and ``graphstorm.inference.GSgnnNodePredictionInferrer``, to form the training and infernece pipeline. In terms of the GNN models, users can refer to the `demo_model.py <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/demo_models.py>`_ file in which all models are created by using GraphStorm APIs.

More example notebooks will be released.

.. toctree::
  :maxdepth: 2
  :titlesonly:

  Notebook_0_Data_Prepare
  Notebook_1_NC_Pipeline
  Notebook_2_LP_Pipeline
  Notebook_3_Model_Variants
  Notebook_4_Customized_Models
