.. _input_raw_data:

Input Raw Data Specification
=============================

In order to use GraphStorm's graph construction pipeline on a single machine or a distributed environment, users should prepare their input raw data accroding to GraphStorm's specifications explained below.

Data tables
------------
The main part of GraphStorm input raw data is composed of two sets of tables. One for nodes and one for edges. These data tables could be in one of three file formats: ``csv`` files, ``parquet`` files, or ``HDF5`` files. All of the three file formats store data in tables that contain headers, i.e., a list of column names, and values belonging to each column.

Node tables
............
GraphStorm requires each node type to have its own table(s). It is suggested to have one folder for one node type to store table(s).

In the table for one node type, there **must** be one column that stores the IDs of nodes. The IDs could be non-integers, such as strings. GraphStorm will treat non-integer IDs as strings and convert them into interger IDs. 

If certain type of nodes has features, the features could be stored in multiple columns, each of which stores one type of features. These features could be numerical, categorial, or textual data. Similarly, training labels associated with certain type of nodes could be stored in multiple columns, each of which store one type of labels. 

Edge tables
............
GraphStorm requires each edge type to have its own table(s). It is suggested to have one folder for one edge type to store tables(s).

In the table for one edge type, there **must** be two columns. One column stores the IDs of source node type of the edge type, while another column stores the IDs of destination node type of the edge type. The source and destination node type should have their corresponding node tables. Same as node features and labels, edge features and labels could be stored in multiple columns.

.. note:: 
    
    * If the number of rows is too large, it is suggested to split and store the data into mutliple table files that have the identical schema. Doing so could speed up the data reading process during graph construction if use multiple processing.
    * It is suggested to use **parquet** file format for its popularity and compressed file sizes. The **HDF5** format is only suggested for data with large volume of high dimension features.
    * Users can also store columns in multiple sets of table files, for example, puting "node IDs" and "feature_1" in the set of "table1_1.parquet" file and  "table1_2.parquet" file, and put "feature_2" in another set of "table2_1.h5" file and "table2_2.h5" file with the same row order.

.. warning:: 
    
    If users split both rows and columns into mutliple sets of table files, they need to make sure that after files are sorted according to the file names, the order of the rows of each column will still keep the same.
    
    Suppose the columns are split into two file sets. One set includes a list of files, i.e., ``table_1.h5, table_2.h5, ..., table_9.h5, table_10.h5, table_11.h5``, and another set also includes a list of files, i.e., ``table_001.h5, table_002.h5, ..., table_009.h5, table_010.h5, table_011.h5``. The order of rows in the two set of files is the same when using the original order of files in the two lists. However, after being sorted by Linux OS, we will get ``table_1.h5, table_10.h5, table_11.h5, table_2.h5, ..., table_9.h5`` for the first list, and get ``table_001.h5, table_002.h5, ..., table_009.h5, table_010.h5, table_011.h5`` for the second list. The order of files is different, which will cause mismatch between node IDs and node features.

    Therefore, it is **strongly** suggested to use the ``_000*`` file name template, like ``table_001, table_002, ..., table_009, table_010, table_011, ..., table_100, table_101, ...``.

.. _customized-split-labels:

Label split files (Optional)
-----------------------
In some cases, users may want to control which nodes or edges should be used for training, validation, or testing. To achieve this goal, users can set the customized label split information in three JSON files or parquet files.

For node split files, users just need to list the node IDs used for training in one file, node IDs used for validation in one file, and node IDs used for testing in another file. If use JSON files, put one node ID in one line like :ref:`this example <node-split-json>` below. If use parquet files, place these node IDs in one column and assign a column name to it.

Foe edge split files, users need to provide both source node IDs and destination node IDs in the split files. If use JSON files, put one edge as a JSON list with two elements, i.e., ``["source node ID", "destination node ID"]``, in one line. If use parquet files, place the source node IDs and destination node IDs into two columns, and assign column names to them  like :ref:`this example <edge-split-parquet>` below.

If there is no validation or testing set, users do not need to create the corresponding file(s).

.. _simple-input-raw-data-example:

A simple raw data example
--------------------------
To better help users to prepare the input raw data artifacts, this section provides a simple example.

This simple raw data has three types of nodes, ``paper``, ``subject``, ``author``, and two types of edges, ``paper, has, subject`` and ``paper, written-by, author``.

``paper`` node tables
.......................
The ``paper`` table (``paper_nodes.parquet``) includes three columns, i.e., `nid` for node IDs, `aff` is a feature column with categorial values, `class` is a classification label column with 3 classes, and ``abs`` is a feature column with textual values.

=====  =======  ======= ===============
nid     aff      class   abs
=====  =======  ======= ===============
n1_1    NE       0       chips are
n1_2    MT       1       electricity
n1_3    UL       1       prime numbers
n1_4    TT       2       Questions are
=====  =======  ======= ===============


``subject`` node table
.......................
The ``subject`` table (``subject_nodes.parquet``) includes one column only, i.e., `domain`, functioning as node IDs.

+--------+
| domain |   
+========+
| eee    |
+--------+
| mth    |
+--------+
| llm    |
+--------+

.. _multi-set-table-examle:

``author`` node table
.......................
The ``author`` table (``author_nodes.parquet``) includes two columns, i.e., `n_id` for node IDs, and `hdx` as a feature column with numerical values.

=====  =======
n_id    hdx
=====  =======
60      0.75  
70      25.34 
80      1.34  
=====  =======

The ``author`` nodes also have a 2048 dimension embeddings pre-computed on a textual feature stored as an **HDF5** file (``author_node_embeddings.h5``) as shown below.

+----------------------------------------------------------------+
|                             embedding                          |
+================================================================+
| 0.2964, 0.0779, 1.2763, 2.8971, ..., -0.2564, 0.9060, -0.8740  |
+----------------------------------------------------------------+
| 1.6941, -1.6765, 0.1862, -0.4449, ..., 0.6474, 0.2358, -0.5952 |
+----------------------------------------------------------------+
| -0.8417, 2.5096, -0.0393, -0.8208, ..., 0.9894, 2.3389, 0.9778 |
+----------------------------------------------------------------+

.. note:: The order of rows in the ``author_node_embeddings.h5`` file **MUST** be same as those in the ``author_nodes.parquet`` file, i.e., the first value row contains the embeddings for the ``author`` node with ``n_id`` as ``60``, and the second value row is for ``author`` node with ``n_id`` as ``70``, and so on.

``paper, has, subject`` edge table
......................................
The ``paper, has, subject`` edge table (``paper_has_subject_edges.parquet``) includes three columns, i.e., ``nid`` as the source node IDs, ``domain`` as the destination IDs, and ``cnt`` as the label field for a regression task.

=====  =======  =======
nid    domain    cnt
=====  =======  =======
n1_1    eee       100
n1_2    eee       1
n1_3    mth       39
n1_4    llm       4700
=====  =======  =======

``paper, written-by, author`` edge table
......................................
The ``paper, written-by, author`` edge table (``paper_written-by_author_edges.parquet``) includes two columns, i.e., ``nid`` as the source node IDs, ``n_id`` as the destination IDs.

=====  =======
nid     n_id 
=====  =======
n1_1    60   
n1_2    60   
n1_3    70   
n1_4    70   
=====  =======

.. _node-split-json:

Node split JSON files
......................
This example sets customized node split files on the ``paper`` nodes for a node classification task in the JSON format. There are two nodes in the training set, one node for validation, and one node for testing.

**train.json** contents

.. code:: json

    n1_2
    n1_3

**val.json** contents

.. code:: json

    n1_4

**test.json** contents

.. code:: json

    n1_1

.. _edge-split-parquet:

Edge split parquet files
.........................

This example sets customized edge split files on the ``paper, has, subject`` edges for an edge regression task in the parquet format. There are three edges in the training set, one edge for validation, and no edge for testing.

**train_edges.parquet** contents

=====  =======
nid    domain 
=====  =======
n1_1    eee   
n1_2    eee   
n1_4    llm   
=====  =======

**val_edges.parquet** contents

=====  =======
nid    domain 
=====  =======
n1_3    mth   
=====  =======
