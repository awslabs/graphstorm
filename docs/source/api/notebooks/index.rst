.. _programming-examples:

API Programming Examples
=========================

Besides the Command Line Interfaces (CLIs), GraphStorm releases a set of Application Programming Interfaces (APIs) for users to build their own Graph Machine Learning (GML) models and pipelines. To help users to better use these APIs, we provide a set of Jupyter notebooks as examples. Users can download these notebooks and related python files, and then run them locally in the GraphStorm standalone mode.

.. note:: All runnable Jupyter notebooks and related python files can be downloaded from the `GraphStorm Github notebook repository <https://github.com/awslabs/graphstorm/tree/main/docs/source/api/notebooks>`_.

These notebooks all use the same ACM data as discussed in the :ref:`User Your Own Data Tutorial<use-own-data>`. Users can create the required dataset and explore details of it by running the `Notebook 0: Data Preparation <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_0_Data_Prepare.ipynb>`_.

The `Notebook 1: Use GraphStorm APIs for Building a Node Classification Pipeline <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_1_NC_Pipeline.ipynb>`_ provides an example that demonstrates how to reproduce the node classification training and inference pipeline that is identical as the :ref:`Step 3: Launch training and inference scripts on your own graphs<launch_training_oyog>` does. While the `Notebook 2: Use GraphStorm APIs for Building a Link Prediction Pipeline <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_2_LP_Pipeline.ipynb>`_ provides another example that use the same dataset, but conduct pipelines for link prediction training and inference. 

GraphStorm provides a variaty of built-in neural network model components that can be easily combined to form different GNN models, e.g., RGCN and HGT. `Notebook 3: Use GraphStorm APIs for Implementing Built-in GNN Models <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_3_Model_Variants.ipynb>`_. demostrates how to leverage the built-in neural network components through GraphStorm APIs to implement models like RGCN, RGAT and HGT.

Besides built-in model components, following GraphStorm APIs users can develop customized components for their specific requirements. `Notebook 4: Use GraphStorm APIs for Customizing Model Components <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_4_Customized_Models.ipynb>`_ provides an example of a customized `RGAT` encoder and how to integrate it into the GraphStorm model architecture.

Once they're familiar with customizing components, users can refer to `Notebook 5: Use GraphStorm APIs for a Customized Model to Perform Graph-level Prediction <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_5_GP_solution.ipynb>`_ to learn how to perform graph-level prediction in GraphStorm.

These examples utilize GraphStorm APIs, such as ``graphstorm``, ``graphstorm.dataloading.GSgnnDataset``, ``graphstorm.dataloading.GSgnnNodeDataLoader``, ``graphstorm.trainer.GSgnnNodePredictionTrainer``, and ``graphstorm.inference.GSgnnNodePredictionInferrer``, to form the training and infernece pipeline. In terms of the GNN models, users can refer to the `demo_model.py <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/demo_models.py>`_ file in which all models are created by using GraphStorm APIs.

These notebooks can run with the GraphStrom Standalone mode, i.e., on a single CPU or GPU of a single machine. To fully leverage GraphStorm's distributed model training and inference capability, users can follow the guidelines shown in `Notebook 6: Running Custom Model with GraphStorm CLIs <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_6_API2CLI.ipynb>`_ and choose a proper distributed environment to run the custom models for their enterprise-level graph datasets. Users can refer to the `demo_run_train.py <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/demo_run_train.py>`_ and `demo_run_infer.py <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/demo_run_infer.py>`_ as examples of custom models to be used by GraphStorm CLIs.

.. toctree::
  :maxdepth: 2
  :titlesonly:

  Notebook_0_Data_Prepare
  Notebook_1_NC_Pipeline
  Notebook_2_LP_Pipeline
  Notebook_3_Model_Variants
  Notebook_4_Customized_Models
  Notebook_5_GP_solution
  Notebook_6_API2CLI