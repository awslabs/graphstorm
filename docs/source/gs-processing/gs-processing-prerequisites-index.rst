================================================
GraphStorm Processing Prerequisites
================================================

GraphStorm Distributed Data Processing (GSProcessing) allows you to process
and prepare massive graph data for training with GraphStorm. GSProcessing takes
care of generating unique ids for nodes, using them to encode edge structure files,
process individual features and prepare the data to be passed into the distributed
partitioning and training pipeline of GraphStorm.

We use PySpark to achieve horizontal parallelism, allowing us to scale to graphs with billions of nodes and edges.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  gs-processing-getting-started.rst

.. toctree::
  :maxdepth: 1
  :titlesonly:

  usage/example.rst

.. toctree::
  :maxdepth: 1
  :titlesonly:

  usage/distributed-processing-setup.rst