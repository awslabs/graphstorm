.. _graph_construction:

==============================
GraphStorm Graph Construction
==============================

In order to use GraphStorm's graph construction pipeline on a single machine or a distributed environment, users should prepare their input raw data accroding to GraphStorm's specifications. Users can find more details of these specifications in the :ref:`Input Raw Data Explanations <input_raw_data>` section.

Once the raw data is ready, by using GraphStorm :ref:`single machine graph construction CLIs <single-machine-gconstruction>`, users can handle most common academic graphs or small graphs sampled from enterprise data, typically with millions of nodes and up to one billion edges. It's recommended to use machines with large CPU memory. A general guideline: 1TB of memory for graphs with one billion edges.

Many production-level enterprise graphs contain billions of nodes and edges, with features having hundreds or thousands of dimensions. GraphStorm :ref:`distributed graph construction CLIs <distributed-gconstruction>` help users manage these complex graphs. This is particularly useful for building automatic graph data processing pipelines in production environments. GraphStorm :ref:`distributed graph construction CLIs <distributed-gconstruction>` could be applied on multiple Amazon infrastructures, including `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_,
`EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_, and
`EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_.

.. toctree::
   :maxdepth: 2
   :glob:

   raw_data
   single-machine-gconstruct
   distributed/index