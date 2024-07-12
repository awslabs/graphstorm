.. _programming-examples:

API Programming Examples
=========================

Besides the Command Line Interfaces (CLIs), GraphStorm releases a set of Application Programming Interfaces (APIs) for users to build their own Graph Machine Learning (GML) models and pipelines. To help users to better use these APIs, we provide a set of Jupyter notebooks as examples. Users can download these notebooks and related python files, and then run them locally in the GraphStorm standalone mode.

These notebooks all use the same ACM data as discussed in the :ref:`User Your Own Data Tutorial<use-own-data>`. Users create the required dataset and explore details of it by running the **Notebook 0: Data Prepare** notebook.

The **Notebook 1** .. _Notebook_1: Notebook_1_NC_Pipeline.ipynb provides an example that demonstrate how to reproduce the node classification training and inference pipeline that is identical as the :ref:`Step 3: Launch training and inference scripts on your own graphs<launch_training_oyog>` does. While the **Notebook 2** provides another example that use the same dataset, but conduct pipelines for link prediction training and inference. 

These two examples utilize GraphStorm APIs, such as ``graphstorm``, ``graphstorm.dataloading.GSgnnDataset``, ``graphstorm.dataloading.GSgnnNodeDataLoader``, ``graphstorm.trainer.GSgnnNodePredictionTrainer``, and ``graphstorm.inference.GSgnnNodePredictionInferrer``, to form the training and infernece pipeline. In terms of the GNN models, users can refer to the `demo_model.py <https://github.com/awslabs/graphstorm/blob/main/docs/source/notebooks/demo_models.py>`_ file in which all models are created by using GraphStorm APIs.

More notebooks will be released.

.. toctree::
  :maxdepth: 2
  :titlesonly:

  Notebook_0_Data_Prepare
  Notebook_1_NC_Pipeline
  Notebook_2_LP_Pipeline
  Notebook_3_Model_Variants
