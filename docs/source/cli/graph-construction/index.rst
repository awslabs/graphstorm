.. _graph_construction:

==============================
GraphStorm Graph Construction
==============================

Graphstorm offers CLIs that support construct graphs on either a single machine or distributed clusters for different use cases.

By using GraphStorm :ref:`single machine graph construction CLIs <single-machine-gconstruction>`, users can handle most of common academic graphs or relatively small graphs sampled from enterprise data, which are normally have million-level number of nodes and around one billions (or less) of edges. It is suggested to use machines with large CPU memory volumn. A rule of thumb: 1TB for graphs with one billions of edges. 

Many enterprise-level graphs used in production contain billions of nodes and hundreds of billions of edges, and features with hundreds or thousands dimensions. GraphStorm :ref:`distributed graph construction CLIs <distributed-gconstruction>` can help users to tackle these graphs. This will be useful when users need to build automatic graph data processing pipeline in their production environments. GraphStorm :ref:`distributed graph construction CLIs <distributed-gconstruction>` could be applied to multiple Amazon infrastructures, including `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_,
`EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_, and
`EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_.

.. toctree::
   :maxdepth: 2
   :glob:

   single-machine-gconstruct.rst
   distributed/index.rst