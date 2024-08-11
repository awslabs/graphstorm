.. _input_raw_data:

GraphStorm Input Raw Data Artifact Explanations
================================================

In order to use GraphStorm's graph construction pipeline both on a single machine and in a distributed environment, users should prepare their input raw data accroding to GraphStorm's requirements explained below.

Data Tables
------------
The main part of GraphStorm input raw data is composed of two sets of tables. One for nodes and one for edges. These data tables could be in one of three file formats: **csv** files, **parquet** files, or **HDF5** files. All of the three file formats store data in tables that contain headers, i.e., a list of column names, and values belonging to each column.

.. note:: If the number of rows is too large, it is suggested to split and store the data into mutliple tables that have the identical schema. Doing so could speed up the data reading process during graph construction when using multiple processing.

Node tables
............
GraphStorm requires each node type has its own table(s). It is suggested to have one dedicated folder for each node type to store table(s).

Edge tables
............

Label files (Optional)
-----------------------

