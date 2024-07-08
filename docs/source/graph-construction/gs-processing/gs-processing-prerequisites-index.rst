===================================
GraphStorm Processing Prerequisites
===================================

GraphStorm Distributed Data Processing (GSProcessing) allows you to process
and prepare massive graph data for training with GraphStorm. GSProcessing takes
care of generating unique ids for nodes, using them to encode edge structure files,
process individual features and prepare the data to be passed into the distributed
partitioning and training pipeline of GraphStorm.

We use PySpark to achieve horizontal parallelism, allowing us to scale to graphs with billions of nodes and edges.

The following sections outline essential prerequisites and provide a detailed guide to using GSProcessing
effectively. The first section provides an introduction to GSProcessing, how to install it locally and a quick example of its input configuration.
The second section provides a comprehensive example, showcasing the full workflow of GSProcessing using a toy graph that we process locally.
Finally, the third section demonstrates how to set up GSProcessing for distributed processing,
enabling scalable and efficient processing using AWS resources.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  gs-processing-getting-started.rst
  usage/distributed-processing-setup.rst
  usage/example.rst