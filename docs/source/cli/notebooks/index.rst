.. _cli-examples:

GraphStorm CLI Examples
========================

GraphStorm provides Command Line Interfaces (CLIs) for graph data
preparation and GML model training and inference. To help users to
quickly onboard GraphStorm CLIs, we provide a set of Jupyter
notebooks as examples.

.. note:: All runnable Jupyter notebooks can be downloaded from the `GraphStorm Github repository <https://github.com/awslabs/graphstorm/tree/main/docs/source/cli/notebooks>`_.

These notebooks all use the same ACM data as discussed in
the :ref:`User Your Own Data Tutorial<use-own-data>`.
Users can follow the `Notebook 0: Data Preparation <https://github.com/awslabs/graphstorm/blob/main/docs/source/api/notebooks/Notebook_0_Data_Prepare.ipynb>`_
to explore the details of ACM data preparation.

The `CLI Notebook: Use GraphStorm CLI for Multi-task Learning <https://github.com/awslabs/graphstorm/blob/main/docs/source/cli/notebooks/Notebook_CLI_MT.ipynb>`_ provides
an example that demonstrates how to run multi-task GNN model
training and inference with GraphStorm CLIs. The training
tasks include link prediction and node feature reconstruction.

.. toctree::
  :maxdepth: 2
  :titlesonly:

  Notebook_CLI_MT

