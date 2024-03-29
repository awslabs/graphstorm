GraphStorm Processing Example
=============================

To demonstrate how to use the library locally we will
use the same example data as we use in our
unit tests, which you can find in the project's repository,
under ``graphstorm/graphstorm-processing/tests/resources/small_heterogeneous_graph``.

Install example dependencies
----------------------------

To run the local example you will need to install the GSProcessing
library to your Python environment, and you'll need to clone the
GraphStorm repository to get access to the data.

Follow the :ref:`gsp-installation-ref` guide to install the GSProcessing library.

You can clone the repository using

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

You can then navigate to the ``graphstorm-processing/`` directory
that contains the relevant data:

.. code-block:: bash

    cd ./graphstorm/graphstorm-processing/


Expected file inputs and configuration
--------------------------------------

GSProcessing expects the input files to be in a specific format that will allow
us to perform the processing and prepare the data for partitioning and training.

The data files are expected to be:

* Tabular data files. We support CSV-with-header format, or in Parquet format.
  The files can be split (multiple parts), or a single file.
* Available on a local file system or on S3.
* One prefix per edge and node type. For example, for a particular edge
  type, all node identifiers (source, destination), features, and labels should
  exist as columns in one or more files under a common prefix (local or on S3).

Apart from the data, GSProcessing also requires a configuration file that describes the
data and the transformations we will need to apply to the features and any encoding needed for
labels.
We support both the `GConstruct configuration format <https://graphstorm.readthedocs.io/en/latest/configuration/configuration-gconstruction.html#configuration-json-explanations>`_
, and the library's own GSProcessing format, described in :doc:`/gs-processing/developer/input-configuration`.

.. note::
    We expect end users to only provide a GConstruct configuration file,
    and only use the configuration format of GSProcessing as an intermediate
    layer to decouple the two projects.

    Developers who are looking to use GSProcessing
    as their backend processing engine can either use the GSProcessing configuration
    format directly, or translate their own configuration format to GSProcessing,
    as we do with GConstruct.

For a detailed description of all the entries of the GSProcessing configuration file see
:doc:`/gs-processing/developer/input-configuration`.

.. _gsp-relative-paths:

Relative file paths required
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The one difference with single-instance GConstruct files,
is that we require that the file paths listed in the configuration file are
`relative to the location of the configuration file.` Specifically:

* All file paths listed **must not** start with ``/``.
* Assuming the configuration file is under ``$PATH``, and a filepath is listed as ``${FILEPATH}``
  in the configuration file, the corresponding file is expected to exist at ``${PATH}/${FILEPATH}``.

For example:

.. code-block:: bash

    > pwd
    /home/path/to/data/ # This is the current working directory (cwd)
    > ls
    gconstruct-config.json edge_data # These are the files under the cwd
    > ls edge_data/ # These are the files under the edge_data directory
    movie-included_in-genre.csv

The contents of the ``gconstruct-config.json`` can be:

.. code-block:: python

    {
        "edges" : [
            {
                # Note that the file is a relative path
                "files": ["edge_data/movie-included_in-genre.csv"],
                "format": {
                    "name": "csv",
                    "separator" : ","
                }
                # [...] Other edge config values
            }
        ]
    }

Given the above we can run a job with local input data as:

.. code-block:: bash

    > gs-processing --input-data /home/path/to/data \
        --config-filename gconstruct-config.json --do-repartition True

The benefit with using relative paths is that we can move the same files
to any location, including S3, and run the same job without making changes to the config
file:

.. code-block:: bash

    # Move all files to new directory
    > mv /home/path/to/data /home/new-path/to/data
    # After moving all the files we can still use the same config
    > gs-processing --input-data /home/new-path/to/data \
        --config-filename gconstruct-config.json

    # Upload data to S3
    > aws s3 sync /home/new-path/to/data s3://my-bucket/data/
    # We can still use the same config, just change the prefix to an S3 path
    > python run_distributed_processing.py --input-data s3://my-bucket/data \
        --config-filename gconstruct-config.json

Node files are optional (but recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GSProcessing does not require node files to be provided for
every node type. Any node types that appears as source or destination in one of the edges,
its unique node identifiers will be determined by the edge files.

However, this is an expensive operation, so if you know your node ID
space from the start we recommend providing node input files for each
node type. You can also have a mix of some node types being provided
and others inferred by the edges.

Example data and configuration
------------------------------

For this example we use a small heterogeneous graph inspired by the Movielens dataset.
You can see the configuration file under
``graphstorm/graphstorm-processing/tests/resources/small_heterogeneous_graph/gconstruct-config.json``

We have 4 node types, ``movie``, ``genre``, ``director``, and ``user``. The graph has 3
edge types, ``movie:included_in:genre``, ``user:rated:movie``, and ``director:directed:movie``.

We include one ``no-op`` feature, ``age``, that we directly pass to the output without any transformation,
and one label, ``gender``, that we transform to prepare the data for a node classification task.


Run a GSProcessing job locally
------------------------------

While GSProcessing is designed to run on distributed clusters,
we can also run small jobs in a local environment, using a local Spark instance.

To do so, we will be using the ``gs-processing`` entry point,
to process the data and create the output on our local storage.

We will provide an input and output prefix for our data, passing
local paths to the script.

Assuming our working directory is ``graphstorm/graphstorm-processing/``
we can use the following command to run the processing job locally:

.. code-block:: bash

    gs-processing --config-filename gconstruct-config.json \
        --input-prefix ./tests/resources/small_heterogeneous_graph \
        --output-prefix /tmp/gsprocessing-example/ \
        --do-repartition True

About re-partitioning
~~~~~~~~~~~~~~~~~~~~~

To finalize processing and to wrangle the data into the structure that
DGL distributed partitioning expects, we need an additional step that
guarantees the data conform to the expectations of DGL, after the
Spark job is done.

We have the option to run this additional step on the Spark leader
as shown above by setting `--do-repartition` to `"True"`.
If our data are too large for the memory of our Spark leader
we can run the step as a separate job:

.. code-block:: bash

    gs-repartition --input-prefix /tmp/gsprocessing-example/

For more details on the re-partitioning step see
:doc:`row-count-alignment`.

.. _gsp-examining-output:

Examining the job output
------------------------

Once the processing and re-partitioning jobs are done,
we can examine the outputs they created. The output will be
compatible with the `Chunked Graph Format of DistDGL <https://docs.dgl.ai/guide/distributed-preprocessing.html#chunked-graph-format>`_
and can be used downstream to create a partitioned graph.

.. code-block:: bash

    $ cd /tmp/gsprocessing-example
    $ ls

    edges/  launch_arguments.json  metadata.json  node_data/
    raw_id_mappings/  perf_counters.json  updated_row_counts_metadata.json

We have a few JSON files and the data directories containing
the graph structure, features, and labels. In more detail:

* ``launch_arguments.json``: Contains the arguments that were used
  to launch the processing job, allowing you to check the parameters after the
  job finishes.
* ``updated_row_counts_metadata.json``:
  This file is meant to be used as the input configuration for the
  distributed partitioning pipeline. ``gs-repartition`` produces
  this file using the original ``metadata.json`` file as input.
* ``metadata.json``: Created by ``gs-processing`` and used as input
  for ``gs-repartition``, can be removed the ``gs-repartition`` step.
* ``perf_counters.json``: A JSON file that contains runtime measurements
  for the various components of GSProcessing. Can be used to profile the
  application and discover bottlenecks.

The directories created contain:

* ``edges``: Contains the edge structures, one sub-directory per edge
  type. Each edge file will contain two columns, the source and destination
  `numerical` node id, named ``src_int_id`` and ``dist_int_id`` respectively.
* ``node_data``: Contains the features for the nodes, one sub-directory
  per node type. Each file will contain one column named after the original
  feature name that contains the value of the feature (could be a scalar or a vector).
* ``raw_id_mappings``: Contains mappings from the original node ids to the
  ones created by the processing job. This mapping would allow you to trace
  back predictions to the original nodes/edges. The files will have two columns,
  ``orig`` that contains the original string ID of the node, and ``new``
  that contains the numerical id that the string id was mapped to. This
  can be used downstream to retrieve the original node ids after training.
  You will use the S3 path these mappings are created under in your call to
  `GraphStorm inference <https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html#launch-inference>`_.

If the graph had included edge features they would appear
in an ``edge_data`` directory.

.. note::

    It's important to note that files for edges and edge data will have the
    same order and row counts per file, as expected by DistDGL. Similarly,
    all node feature files will have the same order and row counts, where
    the first row corresponds to the feature value for node id 0, the second
    for node id 1 etc.


At this point you can use the DGL distributed partitioning pipeline
to partition your data, as described in the
`DGL documentation <https://docs.dgl.ai/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline>`_
.

To simplify the process of partitioning and training, without the need
to manage your own infrastructure, we recommend using GraphStorm's
`SageMaker wrappers <https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html>`_
that do all the hard work for you and allow
you to focus on model development. In particular you can follow the GraphStorm documentation to run
`distributed partitioning on SageMaker <https://github.com/awslabs/graphstorm/tree/main/sagemaker#launch-graph-partitioning-task>`_.


To run GSProcessing jobs on Amazon SageMaker we'll need to follow
:doc:`/gs-processing/usage/distributed-processing-setup` to set up our environment
and :doc:`/gs-processing/usage/amazon-sagemaker` to execute the job.


.. rubric:: Footnotes


.. [#f1] Note that this is just a hint to the Spark engine, and it's
    not guaranteed that the number of output partitions will always match
    the requested value.