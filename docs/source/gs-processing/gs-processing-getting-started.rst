.. _gs-processing:

GraphStorm Processing Getting Started
=====================================


GraphStorm Distributed Data Processing (GSProcessing) allows you to process and prepare massive graph data
for training with GraphStorm. GSProcessing takes care of generating
unique ids for nodes, using them to encode edge structure files, process
individual features and prepare the data to be passed into the
distributed partitioning and training pipeline of GraphStorm.

We use PySpark to achieve
horizontal parallelism, allowing us to scale to graphs with billions of nodes
and edges.

.. _gsp-installation-ref:

Installation
------------

The project needs Python 3.9 and Java 8 or 11 installed. Below we provide brief
guides for each requirement.

Install Python 3.9
^^^^^^^^^^^^^^^^^^

The project uses Python 3.9. We recommend using `PyEnv <https://github.com/pyenv/pyenv>`_
to have isolated Python installations.

With PyEnv installed you can create and activate a Python 3.9 environment using

.. code-block:: bash

    pyenv install 3.9
    pyenv local 3.9

Install GSProcessing from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With a recent version of ``pip`` installed (we recommend ``pip>=21.3``), you can simply run ``pip install .``
from the root directory of the project (``graphstorm/graphstorm-processing``),
which should install the library into your environment and pull in all dependencies:

.. code-block:: bash

    # Ensure Python is at least 3.9
    python -V
    cd graphstorm/graphstorm-processing
    pip install .

Install GSProcessing using poetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also create a local virtual environment using `poetry <https://python-poetry.org/docs/>`_.
With Python 3.9 and ``poetry`` installed you can run:

.. code-block:: bash

    cd graphstorm/graphstorm-processing
    # This will create a virtual env under graphstorm-processing/.venv
    poetry install
    # This will activate the .venv
    poetry shell


Install Java 8 or 11
^^^^^^^^^^^^^^^^^^^^

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

Running locally
---------------

For data that fit into the memory of one machine, you can run jobs locally instead of a
cluster.

To use the library to process your data, you will need to have your data
in a tabular format, and a corresponding JSON configuration file that describes the
data. **The input data need to be in CSV (with header(s)) or Parquet format.**

The configuration file can be in GraphStorm's GConstruct format,
**with the caveat that the file paths need to be relative to the
location of the config file.** Also note that you'll need to convert
all your input data to CSV or Parquet files.

See :ref:`gsp-relative-paths` for more details.

After installing the library, executing a processing job locally can be done using:

.. code-block:: bash

    gs-processing \
        --config-filename gconstruct-config.json \
        --input-prefix /path/to/input/data \
        --output-prefix /path/to/output/data \
        --do-repartition True


Once this script completes, the data are ready to be fed into DGL's distributed
partitioning pipeline.
See `this guide <https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html>`_
for more details on how to use GraphStorm distributed partitioning and training on SageMaker.

See :doc:`usage/example` for a detailed walkthrough of using GSProcessing to
wrangle data into a format that's ready to be consumed by the GraphStorm
distributed training pipeline.


Running on AWS resources
------------------------

GSProcessing supports Amazon SageMaker, EMR on EC2, and EMR Serverless as execution environments.
To run distributed jobs on AWS resources we will have to build a Docker image
and push it to the Amazon Elastic Container Registry, which we cover in
:doc:`usage/distributed-processing-setup`. We can then run either a SageMaker Processing
job which we describe in :doc:`usage/amazon-sagemaker`, an EMR on EC2 job which
we describe in :doc:`usage/emr`, or an EMR Serverless
job that is covered in :doc:`usage/emr-serverless`.


Input configuration
-------------------

GSProcessing supports both the GConstruct JSON configuration format,
as well as its own GSProcessing config. You can learn about the
GSProcessing JSON configuration in :doc:`developer/input-configuration`.


Developer guide
---------------

To get started with developing the package refer to :doc:`developer/developer-guide`.


.. rubric:: Footnotes

.. [#f1] DGL expects that every file produced for a single node/edge type
    has matching row counts, which is something that Spark cannot guarantee.
    We use the re-partitioning script to fix this where needed in the produced
    output. See :doc:`usage/row-count-alignment` for details.