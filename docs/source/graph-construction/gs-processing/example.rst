.. _distributed_construction_example:

A GraphStorm Distributed Graph Construction Example
===================================================

GraphStorm's distributed graph construction is involved with multiple steps.
To help users better understand these steps, we provide an example of distributed graph construction,
which can run locally in one instance.

To demonstrate how to use distributed graph construction locally we will
use the same example data as we use in our
unit tests, which you can find in the project's repository,
under ``graphstorm/graphstorm-processing/tests/resources/small_heterogeneous_graph``.

Install dependencies
--------------------

To run the local example you will need to install the GSProcessing and GraphStorm
library to your Python environment, and you'll need to clone the
GraphStorm repository to get access to the data, and DGL tool for GSPartition.

Follow the :ref:`gsp-installation-ref` guide to install the GSProcessing library.

To run GSPartition job, you can install the dependencies as following:

.. code-block:: bash

    pip install graphstorm
    pip install pydantic
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip install dgl==1.1.3 -f https://data.dgl.ai/wheels-internal/repo.html
    git clone https://github.com/awslabs/graphstorm.git
    cd graphstorm
    git clone --branch v1.1.3 https://github.com/dmlc/dgl.git

You can then navigate to the ``graphstorm-processing/`` directory
that contains the relevant data:

.. code-block:: bash

    cd ./graphstorm-processing/


Expected file inputs and configuration
--------------------------------------

The example will include GSProcessing as the first step and GSPartition as the second step.

GSProcessing expects the input files to be in a specific format that will allow
us to perform the processing and prepare the data for partitioning and training.
GSPartition then takes the output of GSProcessing to produce graph data in DistDGLGraph format for training or inference..

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
, and the library's own GSProcessing format, described in :ref:`GSProcessing Input Configuration<gsprocessing_input_configuration>`.

.. note::
    We expect end users to only provide a GConstruct configuration file,
    and only use the configuration format of GSProcessing as an intermediate
    layer to decouple the two projects.

    Developers who are looking to use GSProcessing
    as their backend processing engine can either use the GSProcessing configuration
    format directly, or translate their own configuration format to GSProcessing,
    as we do with GConstruct.

For a detailed description of all the entries of the GSProcessing configuration file see
:ref:`GSProcessing Input Configuration<gsprocessing_input_configuration>`.

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

    gs-processing --config-filename gsprocessing-config.json \
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
:ref:`row count alignment<row_count_alignment>`.

.. _gsp-examining-output:

Examining the job output of GSProcessing
------------------------------------------

Once the processing and re-partitioning jobs are done,
we can examine the outputs they created. The output will be
compatible with the `Chunked Graph Format of DistDGL <https://docs.dgl.ai/guide/distributed-preprocessing.html#chunked-graph-format>`_
and can be used downstream to create a partitioned graph.

.. code-block:: bash

    $ cd /tmp/gsprocessing-example
    $ ls -l

    edges/
    gsprocessing-config_with_transformations.json
    launch_arguments.json
    metadata.json
    node_data/
    perf_counters.json
    precomputed_transformations.json
    raw_id_mappings/
    updated_row_counts_metadata.json

We have a few JSON files and the data directories containing
the graph structure, features, and labels. In more detail:

* ``gsprocessing-config_with_transformations.json``: This is the input configuration
  we used, modified to include representations of any supported transformations
  we applied. This file can be used to re-apply the transformations on new data.
* ``launch_arguments.json``: Contains the arguments that were used
  to launch the processing job, allowing you to check the parameters after the
  job finishes.
* ``metadata.json``: Created by ``gs-processing`` and used as input
  for ``gs-repartition``, can be removed the ``gs-repartition`` step.
* ``perf_counters.json``: A JSON file that contains runtime measurements
  for the various components of GSProcessing. Can be used to profile the
  application and discover bottlenecks.
* ``precomputed_transformations.json``: A JSON file that contains representations
  of supported transformations. To re-use these transformations on another dataset,
  place this file in the top level of another set of raw data, at the same level
  as the input GSProcessing/GConstruct configuration JSON file.
  GSProcessing will use the transformation values listed here
  instead of creating new ones, ensuring that models trained with the original
  data can still be used in the newly transformed data. Currently only
  categorical transformations can be re-applied.
* ``updated_row_counts_metadata.json``:
  This file is meant to be used as the input configuration for the
  distributed partitioning pipeline. ``gs-repartition`` produces
  this file using the original ``metadata.json`` file as input.

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


Run a GSPartition job locally
------------------------------
While :ref:`GSPartition<gspartition_index>` is designed to run on a multi-machine cluster,
you can run GSPartition job locally for the example. Once you have completed the installation
and the GSProcessing example described in the previous section, you can proceed to run the GSPartition step.

Assuming your working directory is ``graphstorm``,
you can use the following command to run the partition job locally:

.. code:: bash

    echo 127.0.0.1 > ip_list.txt
    python3 -m graphstorm.gpartition.dist_partition_graph \
        --input-path /tmp/gsprocessing-example/ \
        --metadata-filename updated_row_counts_metadata.json \
        --output-path /tmp/gspartition-example/ \
        --num-parts 2 \
        --dgl-tool-path ./dgl/tools \
        --partition-algorithm random \
        --ip-config ip_list.txt 

The command above will first do graph partitioning to determine the ownership for each partition and save the results.
Then it will do data dispatching to physically assign the partitions to graph data and dispatch them to each machine.
Finally it will generate the graph data ready for training/inference.

Examining the job output of GSPartition
---------------------------------------

Once the partition job is done, you can examine the outputs.

.. code-block:: bash

    $ cd /tmp/gspartition-example
    $ ls -ltR

    dist_graph/
        metadata.json
        |- part0/
            edge_feat.dgl
            graph.dgl
            node_feat.dgl
            orig_eids.dgl
            orig_nids.dgl
    partition_assignment/
        director.txt
        genre.txt
        movie.txt
        partition_meta.json
        user.txt

The ``dist_graph`` folder contains partitioned graph ready for training and inference.

* ``part0``: As we only specify 1 partition in the previous command, we have one part folder here.
There are five files for the partition
    * ``edge_feat.dgl``: The edge features for part 0 stored in binary format.
    * ``graph.dgl``: The graph structure data for part 0 stored in binary format.
    * ``node_feat.dgl``: The node features data for part 0 stored in binary format.
    * ``orig_eids.dgl``: The mapping for edges between raw edge IDs and the partitioned graph edge IDs.
    * ``orig_nids.dgl``: The mapping for nodes between raw node IDs and the partitioned graph node IDs.

* ``metadata.json``: This file contains metadata about the distributed DGL graph.

The ``partition_assignment`` directory contains different partition results for different node types,
which can reused for the `dgl dispatch pipeline <https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline>`_

.. rubric:: Footnotes


.. [#f1] Note that this is just a hint to the Spark engine, and it's
    not guaranteed that the number of output partitions will always match
    the requested value.