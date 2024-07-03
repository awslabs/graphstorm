================================================
GraphStorm Processing Prerequisites
================================================

GraphStorm Distributed Data Processing (GSProcessing) allows you to process
and prepare massive graph data for training with GraphStorm. GSProcessing takes
care of generating unique ids for nodes, using them to encode edge structure files,
process individual features and prepare the data to be passed into the distributed
partitioning and training pipeline of GraphStorm.

We use PySpark to achieve horizontal parallelism, allowing us to scale to graphs with billions of nodes and edges.

The following documents outline essential prerequisites and provide a detailed guide to using GSProcessing
effectively. The first document initiates GSProcessing, offering a step-by-step walkthrough to get you started.
The second document provides a comprehensive example, showcasing the full workflow of GSProcessing.
Finally, the third document demonstrates how to set up and configure distributed resources,
enabling scalable and efficient processing using AWS resources.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  gs-processing-getting-started.rst
  usage/example.rst
  usage/distributed-processing-setup.rst