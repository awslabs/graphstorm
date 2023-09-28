.. graphstorm-processing documentation master file, created by
   sphinx-quickstart on Tue Aug  1 02:04:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphStorm Distributed Data Processing documentation!
=================================================

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    Example <usage/example>
    Distributed processing setup <usage/distributed-processing-setup>
    Running on Amazon Sagemaker <usage/amazon-sagemaker>
    Developer Guide <developer/developer-guide>
    Input configuration <developer/input-configuration>


GraphStorm Distributed Data Processing allows you to process and prepare massive graph data
for training with GraphStorm. GraphStorm Processing takes care of generating
unique ids for nodes, using them to encode edge structure files, process
individual features and prepare the data to be passed into the
distributed partitioning and training pipeline of GraphStorm.

We use PySpark to achieve
horizontal parallelism, allowing us to scale to graphs with billions of nodes
and edges.

.. _installation-ref:

Installation
------------

The project uses Python 3.9. We recommend using `PyEnv <https://github.com/pyenv/pyenv>`_
to have isolated Python installations.

With PyEnv installed you can create and activate a Python 3.9 environment using

.. code-block:: bash

    pyenv install 3.9
    pyenv local 3.9


With a recent version of ``pip`` installed (we recommend ``pip>=21.3``), you can simply run ``pip install .``
from the root directory of the project (``graphstorm/graphstorm-processing``),
which should install the library into your environment and pull in all dependencies.

Install Java 8, 11, or 17
~~~~~~~~~~~~~~~~~~~~~~~~~

Spark has a runtime dependency on the JVM to run, so you'll need to ensure
Java is installed and available on your system.

On MacOS you can install Java using ``brew``:

.. code-block:: bash

    brew install openjdk@11

On Linux it will depend on your distribution's package
manager. For Ubuntu you can use:

.. code-block:: bash

    sudo apt install openjdk-11-jdk

On Amazon Linux 2 you can use:

.. code-block:: bash

    sudo yum install java-11-amazon-corretto-headless
    sudo yum install java-11-amazon-corretto-devel

To check if Java is installed you can use.

.. code-block:: bash

    java -version


Example
-------

See the provided :doc:`usage/example` for an example of how to start with tabular
data and convert them into a graph representation before partitioning and
training with GraphStorm.

Usage
-----

To use the library to process your data, you will need to have your data
in a tabular format, and a corresponding JSON configuration file that describes the
data. The input data can be in CSV (with header(s)) or Parquet format.

The configuration file can be in GraphStorm's GConstruct format,
with the caveat that the file paths need to be relative to the
location of the config file. See :doc:`/usage/example` for more details.

After installing the library, executing a processing job locally can be done using:

.. code-block:: bash

    gs-processing \
        --config-filename gconstruct-config.json \
        --input-prefix /path/to/input/data \
        --output-prefix /path/to/output/data

Once the processing engine has processed the data, we want to ensure
they match the requirements of the DGL distributed partitioning
pipeline, so we need to run an additional script that will
make sure the produced data matches the assumptions of DGL [#f1]_.

.. note::

    Ensure you pass the output path of the previous step as the input path here.

.. code-block:: bash

    gs-repartition \
        --input-prefix /path/to/output/data

Once this script completes, the data are ready to be fed into DGL's distributed
partitioning pipeline.
See `this guide <https://github.com/awslabs/graphstorm/blob/main/sagemaker/README.md#launch-graph-partitioning-task>`_
for more details on how to use GraphStorm distributed partitioning on SageMaker.

See :doc:`/usage/example` for a detailed walkthrough of using GSProcessing to
wrangle data into a format that's ready to be consumed by the GraphStorm/DGL
partitioning pipeline.


Using with Amazon SageMaker
---------------------------

To run distributed jobs on Amazon SageMaker we will have to build a Docker image
and push it to the Amazon Elastic Container Registry, which we cover in
:doc:`usage/distributed-processing-setup` and run a SageMaker Processing
job which we describe in :doc:`/usage/amazon-sagemaker`.


Developer guide
---------------

To get started with developing the package refer to :doc:`/developer/developer-guide`.


.. rubric:: Footnotes

.. [#f1] DGL expects that every file produced for a single node/edge type
    has matching row counts, which is something that Spark cannot guarantee.
    We use the re-partitioning script to fix this where needed in the produced
    output.