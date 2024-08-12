.. _input_raw_data:

Input Raw Data Explanations
=============================

In order to use GraphStorm's graph construction pipeline both on a single machine and in a distributed environment, users should prepare their input raw data accroding to GraphStorm's requirements explained below.

Data tables
------------
The main part of GraphStorm input raw data is composed of two sets of tables. One for nodes and one for edges. These data tables could be in one of three file formats: **csv** files, **parquet** files, or **HDF5** files. All of the three file formats store data in tables that contain headers, i.e., a list of column names, and values belonging to each column.

.. note:: If the number of rows is too large, it is suggested to split and store the data into mutliple tables that have the identical schema. Doing so could speed up the data reading process during graph construction if use multiple processing.

Node tables
............
GraphStorm requires each node type to have its own table(s). It is suggested to have one folder for one node type to store table(s).

In the table for one node type, there **must** be one column that stores the IDs of nodes. The IDs could be non-integers, such as strings or floats. GraphStorm will treat non-integer IDs as strings and convert them into interger IDs. 

If this type of nodes have features, they could be stored in multiple columns each of which store one type of features. These features could be numerical, categorial, or textual data. Similarly, labels associated with this type of nodes could be stored in multiple columns each of which store one type of labels. 

Edge tables
............
GraphStorm requires each edge type to have it own table(s). It is suggested to have one folder for one edge type to store tables(s).

In the table for one edge type, there **must** be two columns. One column stores the IDs of source node type of the edge type, while another column stores the IDs of destination node type of the edge type. The source and destination node type should have their corresponding node tables. Same as node features and labels, edge features and labels could be stored in multiple columns.

Label files (Optional)
-----------------------
In some cases, users may want to control which nodes or edges should be used for training, validation, or testing. To achieve this goal, users can set the customized label split information in three JSON files or parquet files.

For node split files, users just need to list the node IDs used for training in one file, node IDs used for validation in one file, and node IDs used for testing in another file. If use JSON files, put one node ID in one line. If use parquet files, place these node IDs in one column and assign a column name to it.

Foe edge split files, users need to provide both source node IDs and destination node IDs in the split files. If use JSON files, put one edge as a JSON list with two elements, i.e., ``["source node ID", "destination node ID"]``, in one line. If use parquet files, place the source node IDs and destination node IDs into two columns, and assign column names to them.

If there is no validation or testing set, users do not need to create the corresponding file(s).

A simple raw data example
--------------------------
To better help users to prepare the input raw data artifacts, this section provides a very simple raw data example.

This simple raw data has three types of nodes, ``ntype1``, ``ntype2``, ``ntype3``, and two types of edges, ``ntype1, etype1, ntype2`` and ``ntype1, etype1, ntype3``.

``ntype1`` node table
.......................
=====  =======  =======
nid    region    class
=====  =======  =======
n1_1    NE       0
n1_2    NS       1
n1_3    NS       1
n1_4    NS       2
=====  =======  =======

``ntype1`` table includes three columns, i.e., `nid` for node IDs, `region` is a feature column with categorial values, and `class` is a classification label column with 3 classes.

``ntype2`` node table
.......................
+--------+
| domain |   
+========+
| gml    |
+--------+
| eby    |
+--------+
| amz    |
+--------+

``ntype2`` table includes one column only, i.e., `domain`, functioning as node IDs.

``ntype3`` node table
.......................
=====  =======
n_id    sals
=====  =======
60      0.75    
70      0.234    
80      1.34    
=====  =======

``ntype3`` table includes two columns, i.e., `n_id` for node IDs, and `sals` is a feature column with numerical values.
