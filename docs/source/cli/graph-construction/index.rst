.. _graph_construction:

==============================
GraphStorm Graph Construction
==============================

Graphstorm offers CLIs that support construct graphs on either a single machine or distributed clusters for different use cases.

By using GraphStorm :ref:`single machine graph construction CLIs <single-machine-gconstruction>`, users can handle most common academic graphs or small graphs sampled from enterprise data, typically with millions of nodes and up to one billion edges. It's recommended to use machines with large CPU memory. A general guideline: 1TB of memory for graphs with one billion edges.

Many production-level enterprise graphs contain billions of nodes and edges, with features having hundreds or thousands of dimensions. GraphStorm :ref:`distributed graph construction CLIs <distributed-gconstruction>` help users manage these complex graphs. This is particularly useful for building automatic graph data processing pipelines in production environments. GraphStorm :ref:`distributed graph construction CLIs <distributed-gconstruction>` could be applied on multiple Amazon infrastructures, including `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_,
`EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_, and
`EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_.

.. toctree::
   :maxdepth: 2
   :glob:

   single-machine-gconstruct.rst
   distributed/index.rst