.. _gs-processing:

GSProcessing Getting Started
=====================================

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

See the provided :ref:`example<distributed_construction_example>` for an example of how to start with tabular
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

See :ref:`example<distributed_construction_example>` for a detailed walkthrough of using GSProcessing to
wrangle data into a format that's ready to be consumed by the GraphStorm
distributed training pipeline.


Running on AWS resources
------------------------

GSProcessing supports Amazon SageMaker, EMR on EC2, and EMR Serverless as execution environments.
To run distributed jobs on AWS resources we will have to build a Docker image
and push it to the Amazon Elastic Container Registry, which we cover in
:ref:`distributed processing setup<gsprocessing_distributed_setup>`. We can then run either a SageMaker Processing
job which we describe in :ref:`running GSProcessing on SageMaker<gsprocessing_sagemaker>`, an EMR on EC2 job which
we describe in :ref:`running GSProcessing on EMR EC2<gsprocessing_emr_ec2>`, or an EMR Serverless
job that is covered in :ref:`running GSProcessing on EMR Serverless<gsprocessing_emr_serverless>`.


Input configuration
-------------------

GSProcessing supports both the GConstruct JSON configuration format,
as well as its own GSProcessing config. You can learn about the
GSProcessing JSON configuration in :ref:`GSProcessing Input Configuration<gsprocessing_input_configuration>`.

Re-applying feature transformations to new data
-----------------------------------------------

Often you will process your data at training time and run inference at later
dates. If your data changes in the meantime. e.g. new values appear in a
categorical feature, you'll need to ensure no new values appear in the transformed
data, as the trained model relies on pre-existing values only.

To achieve that, GSProcessing creates an additional file in the output,
named ``precomputed_transformations.json``. To ensure the same transformations
are applied to new data, you can copy this file to the top-level path of your
new input data, and GSProcessing will pick up any transformations there to ensure
the produced data match the ones that were used to train your model.

Currently, we only support re-applying transformations for categorical features.


Developer guide
---------------

To get started with developing the package refer to :ref:`developer guide<gsprocessing_developer_guide>`.


.. rubric:: Footnotes

.. [#f1] DGL expects that every file produced for a single node/edge type
    has matching row counts, which is something that Spark cannot guarantee.
    We use the re-partitioning script to fix this where needed in the produced
    output. See :ref:`row count alignment<row_count_alignment>` for details.