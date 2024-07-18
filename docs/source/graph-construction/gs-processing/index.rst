==============================
Distributed Graph Construction
==============================

Beyond single-machine graph construction, distributed graph construction offers enhanced scalability
and efficiency. This process involves two main steps: GraphStorm Distributed Data Processing (GSProcessing)
and GraphStorm Distributed Data Partitioning (GPartition).

The following sections provide guidance on doing distributed graph construction.
The first section details the execution environment setup for GSProcessing.
The second section offers examples on drafting a configuration file for a GSProcessing job.
The third section explains how to deploy your GSProcessing job with AWS infrastructure.
The forth section includes how to do partition based on the previous GSProcessing result.
The final section shows an example to quick start GSProcessing and GPartition.

.. toctree::
   :maxdepth: 1
   :glob:

   prerequisites/index.rst
   input-configuration.rst
   aws-infra/index.rst
   gpartition/index.rst
   example.rst