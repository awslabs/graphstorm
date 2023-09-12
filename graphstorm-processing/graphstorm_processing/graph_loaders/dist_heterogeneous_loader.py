"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
import logging
import numbers
import os
from collections import Counter, defaultdict
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from pyspark import RDD
from pyspark.sql import Row, SparkSession, DataFrame, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    ArrayType,
    ByteType,
)
from numpy.random import default_rng

from graphstorm_processing.constants import (
    MIN_VALUE,
    MAX_VALUE,
    VALUE_COUNTS,
    COLUMN_NAME,
    SPECIAL_CHARACTERS,
)
from ..config.config_parser import EdgeConfig, NodeConfig, StructureConfig
from ..config.label_config_base import LabelConfig, EdgeLabelConfig
from ..config.feature_config_base import FeatureConfig
from ..data_transformations.dist_feature_transformer import DistFeatureTransformer
from ..data_transformations.dist_label_loader import DistLabelLoader, SplitRates
from ..data_transformations import s3_utils, spark_utils
from .heterogeneous_graphloader import HeterogeneousGraphLoader

# TODO: Remove the pylint disable once we add the rest of the code
from . import schema_utils  # pylint: disable=no-name-in-module
from .row_count_utils import ParquetRowCounter  # pylint: disable=no-name-in-module

FORMAT_NAME = "parquet"
DELIMITER = "" if FORMAT_NAME == "parquet" else ","
NODE_MAPPING_STR = "node_str_id"
NODE_MAPPING_INT = "node_int_id"


class DistHeterogeneousGraphLoader(HeterogeneousGraphLoader):
    """
    A graph loader designed to run distributed processing of a heterogeneous graph.

    Parameters
    ----------
    spark : SparkSession
        The SparkSession that we use to perform the processing
    local_input_path : str
        Local path to input configuration data
    local_metadata_path : str
        Local path to where the output metadata files will be created.
    data_configs : Dict[str, Sequence[StructureConfig]]
        Dictionary of node and edge configurations objects.
    input_prefix : str
        The prefix to the input data. Can be an S3 URI or an **absolute** local path.
    output_prefix : str
        The prefix to where the output data will be created. Can be an S3 URI
        or an **absolute** local path.
    num_output_files : Optional[int], optional
        The number of files (partitions) to create for the output, if None we
        let Spark decide.
    enable_assertions : bool, optional
        When true enables sanity checks for the output created.
        However these are costly to compute, so we disable them by default.
    """

    def __init__(
        self,
        spark: SparkSession,
        local_input_path: str,
        local_output_path: str,
        data_configs: Dict[str, Sequence[StructureConfig]],
        input_prefix: str,
        output_prefix: str,
        num_output_files: Optional[int] = None,
        add_reverse_edges=True,
        enable_assertions=False,
        graph_name: Optional[str] = None,
    ):
        super().__init__(local_input_path, local_output_path, data_configs)

        # TODO: Pass as an argument?
        if input_prefix.startswith("s3://"):
            self.filesystem_type = "s3"
        else:
            assert os.path.isabs(input_prefix), "We expect an absolute path"
            self.filesystem_type = "local"

        self.spark = spark  # type: SparkSession
        self.add_reverse_edges = add_reverse_edges
        # Remove trailing slash in s3 paths
        if self.filesystem_type == "s3":
            self.input_prefix = s3_utils.s3_remove_trailing(input_prefix)
            self.output_prefix = s3_utils.s3_remove_trailing(output_prefix)
        else:
            # TODO: Any checks for local paths?
            self.input_prefix = input_prefix
            self.output_prefix = output_prefix
        self.num_output_files = (
            num_output_files
            if num_output_files and num_output_files > 0
            else int(spark.sparkContext.defaultParallelism)
        )
        # Mapping from node type to filepath, each file is a node-str to node-int-id mapping
        self.node_mapping_paths = {}  # type: Dict[str, Sequence[str]]
        # Mapping from label name to value counts
        self.label_properties: Dict[str, Counter] = defaultdict(Counter)
        self.timers = defaultdict(float)  # type: Dict
        self.enable_assertions = enable_assertions
        # Column names that are valid in CSV can be invalid in Parquet
        # we collect an column name substitutions we had to make
        # here and later output as a JSON file.
        self.column_substitutions = {}  # type: Dict[str, str]
        self.graph_info = {}  # type: Dict[str, Any]
        self.graph_name = graph_name

    def process_and_write_graph_data(
        self, data_configs: Mapping[str, Sequence[StructureConfig]]
    ) -> None:
        """Process and encode all graph data.

        Extracts and encodes graph structure before writing to storage, then applies pre-processing
        steps to node/edge features and labels, and saves transformed output to partitioned files,
        one per node/edge type and feature.

        As processing happens, each step returns values that we use to update
        a common metadata dict that we eventually write to disk.

        Parameters
        ----------
        data_configs : Mapping[str, Sequence[StructureConfig]]
            Dictionary of configuration for nodes and edges
        """
        # TODO: See if it's better to return some data structure
        # for the followup steps instead of just have side-effects
        logging.info("Start Distributed Graph Processing ...")
        process_start_time = perf_counter()

        if not self._at_least_one_label_exists(data_configs):
            self._insert_link_prediction_labels(data_configs["edges"])

        metadata_dict = self._initialize_metadata_dict(data_configs)

        edge_configs = data_configs["edges"]  # type: Sequence[EdgeConfig]

        if "nodes" in data_configs:
            node_configs = data_configs["nodes"]  # type: Sequence[NodeConfig]
            missing_node_types = self._get_missing_node_types(edge_configs, node_configs)
            if len(missing_node_types) > 0:
                logging.info(
                    "At least one node type missing (%s), creating node mapping from edges...",
                    missing_node_types,
                )
                # In this case some node types only exist in the edge configs
                id_map_start_time = perf_counter()
                self.create_node_id_maps_from_edges(edge_configs, missing_node_types)
                self.timers["create_node_id_maps_from_edges"] = perf_counter() - id_map_start_time

            node_start_time = perf_counter()
            metadata_dict["node_data"] = self.process_node_data(node_configs)
            self.timers["process_node_data"] = perf_counter() - node_start_time
        else:
            # In this case no node files exist so we create all node mappings from the edge files
            id_map_start_time = perf_counter()
            missing_node_types = self._get_missing_node_types(edge_configs, [])
            self.create_node_id_maps_from_edges(edge_configs, missing_node_types)
            self.timers["create_node_id_maps_from_edges"] = perf_counter() - id_map_start_time
            metadata_dict["node_data"] = {}

        edges_start_time = perf_counter()
        edge_data_dict, edges_dict = self.process_edge_data(edge_configs)
        self.timers["process_edge_data"] = perf_counter() - edges_start_time
        metadata_dict["edge_data"] = edge_data_dict
        metadata_dict["edges"] = edges_dict
        # We use the data location as the graph name, can also take from user?
        # TODO: Fix this, take from config?
        metadata_dict["graph_name"] = (
            self.graph_name if self.graph_name else self.input_prefix.split("/")[-1]
        )

        # Ensure output dict has the correct order of keys
        for edge_type in metadata_dict["edge_type"]:
            metadata_dict["edges"][edge_type] = metadata_dict["edges"].pop(edge_type)
            if edge_type in metadata_dict["edge_data"]:
                metadata_dict["edge_data"][edge_type] = metadata_dict["edge_data"].pop(edge_type)
        for node_type in metadata_dict["node_type"]:
            if node_type in metadata_dict["node_data"]:
                metadata_dict["node_data"][node_type] = metadata_dict["node_data"].pop(node_type)

        metadata_dict = self._add_node_mappings_to_metadata(metadata_dict)
        row_counts_start_time = perf_counter()
        metadata_dict = self._add_row_counts_to_metadata(metadata_dict)
        self.timers["_add_row_counts_to_metadata"] = perf_counter() - row_counts_start_time

        metadata_dict["graph_info"] = self._finalize_graphinfo_dict(metadata_dict)

        # The metadata dict is written to disk as a JSON file,
        # SageMaker takes care of uploading it to S3
        with open(os.path.join(self.output_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4)

        if len(self.column_substitutions) > 0:
            with open(
                os.path.join(self.output_path, "column_substitutions.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(self.column_substitutions, f, indent=4)

        self.timers["process_and_write_graph_data"] = perf_counter() - process_start_time

        with open(os.path.join(self.output_path, "perf_counters.json"), "w", encoding="utf-8") as f:
            sorted_timers = dict(sorted(self.timers.items(), key=lambda x: x[1], reverse=True))
            json.dump(sorted_timers, f, indent=4)

        logging.info("Finished Distributed Graph Processing ...")

    @staticmethod
    def _at_least_one_label_exists(data_configs: Mapping[str, Sequence[StructureConfig]]) -> bool:
        """
        Checks if at least one of the entries in the edges/nodes contains a label.

        Returns
        -------
        True if at least one entry in the `edges` or `nodes` top-level keys contains
        a label entry, False otherwise.
        """
        edge_configs = data_configs["edges"]  # type: Sequence[StructureConfig]

        for edge_config in edge_configs:
            if edge_config.label_configs:
                return True

        node_configs = data_configs["nodes"]  # type: Sequence[StructureConfig]

        for node_config in node_configs:
            if node_config.label_configs:
                return True

        return False

    @staticmethod
    def _insert_link_prediction_labels(edge_configs: Sequence[StructureConfig]) -> None:
        """
        Inserts a link prediction label entry into the `edges` top-level keys.
        Modifies the data_configs object in-place.

        Parameters
        ----------
        edge_configs
            A sequence of edge structure configurations
        """
        for edge_config in edge_configs:
            config_dict = {
                "column": "",
                "type": "link_prediction",
                "split_rate": {"train": 0.9, "val": 0.1, "test": 0.0},
            }
            edge_config.set_labels([EdgeLabelConfig(config_dict)])

    def _initialize_metadata_dict(
        self, data_configs: Mapping[str, Sequence[StructureConfig]]
    ) -> Dict:
        """Initializes the metadata dict that will be created as output.

        This dict is required downstream by the graph partitioning task,
        see https://docs.dgl.ai/guide/distributed-preprocessing.html#specification
        for details.

        Parameters
        ----------
        data_configs : Mapping[str, Sequence[StructureConfig]]
            A mapping that needs to include an "edges" key,
            and maps to a sequence of `StructureConfig` objects
            describing all the edge types in the graph.

        Returns
        -------
        Dict
            A dict initialized with necessary keys and values that
            will later be further populated during graph processing.
        """
        metadata_dict = {}
        edge_configs = data_configs["edges"]  # type: Sequence[EdgeConfig]

        node_type_set = set()

        edge_types = []
        for edge_config in edge_configs:
            src_type = edge_config.src_ntype
            dst_type = edge_config.dst_ntype
            rel_type = edge_config.get_relation_name()

            node_type_set.update([src_type, dst_type])

            # Add original and reverse edge types
            edge_types.append(f"{src_type}:{rel_type}:{dst_type}")
            if self.add_reverse_edges:
                edge_types.append(f"{dst_type}:rev-{rel_type}:{src_type}")

        metadata_dict["edge_type"] = edge_types
        metadata_dict["node_type"] = sorted(node_type_set)

        return metadata_dict

    def _finalize_graphinfo_dict(self, metadata_dict: Dict) -> Dict:
        self.graph_info["graph_type"] = "heterogeneous"

        self.graph_info["num_nodes"] = sum(metadata_dict["num_nodes_per_type"])
        self.graph_info["num_edges"] = sum(metadata_dict["num_edges_per_type"])

        self.graph_info["num_ntype"] = len(metadata_dict["node_type"])
        self.graph_info["num_rels"] = len(metadata_dict["edge_type"])

        self.graph_info["num_nodes_ntype"] = dict(
            zip(metadata_dict["node_type"], metadata_dict["num_nodes_per_type"])
        )
        self.graph_info["num_edges_etype"] = dict(
            zip(metadata_dict["edge_type"], metadata_dict["num_edges_per_type"])
        )

        if "ntype_label" not in self.graph_info:
            self.graph_info["ntype_label"] = []
        if "ntype_label_property" not in self.graph_info:
            self.graph_info["ntype_label_property"] = []
        if "etype_label" not in self.graph_info:
            self.graph_info["etype_label"] = []
        if "etype_label_property" not in self.graph_info:
            self.graph_info["etype_label_property"] = []

        self.graph_info["label_properties"] = self.label_properties

        return self.graph_info

    def _replace_special_chars_in_cols(self, input_df: DataFrame) -> DataFrame:
        """Replace characters from column names that are not allowed in Parquet column names.

        As a side-effect will store all the column name substitutions in
        `self.column_substitutions`, which are later saved to storage.

        Parameters
        ----------
        input_df : DataFrame
            Spark DataFrame

        Returns
        -------
        DataFrame
            Spark DataFrame with special characters replaced in column names
        """
        problematic_chars = {" ", ",", ";", "{", "}", "(", ")", "=", "\n", "\t"}
        for column in input_df.columns:
            if len(set(column).intersection(problematic_chars)) > 0:
                new_column = column
                for pchar in problematic_chars:
                    new_column = new_column.replace(pchar, "_")

                input_df, new_column = spark_utils.safe_rename_column(input_df, column, new_column)
                self.column_substitutions[column] = new_column

        return input_df

    def _write_df(
        self,
        input_df: DataFrame,
        full_output_path: str,
        out_format: str = "parquet",
        num_files: Optional[int] = None,
    ) -> Sequence[str]:
        """Write a DataFrame to S3 or local storage, in the requested format (csv or parquet).

        If self.num_output_files is set will always coalesce files to that number before writing.


        Parameters
        ----------
        input_df : DataFrame
            Spark DataFrame to be written.
            Could be single column vector or multiple columns.
        full_output_path : str
            S3 URI or local path, under which the files will be written.
            We append the format name (csv/parquet) to the path before writing.
        out_format : str, optional
            "csv" or "parquet", by default "parquet"
        num_files : Optional[int], optional
             Number of partitions the file will be written with.
            If None or -1 we let Spark choose the appropriate number. Note that choosing
            a small number of partitions can have an adverse effect on performance.
            Overrides value of self.num_output_files.

        Returns
        -------
        Sequence[str]
            A (sorted) list of S3 URIs or local file paths that
            were written by Spark under the output_path provided.

        Raises
        ------
        NotImplementedError
            If an output format other than "csv" or "parquet" is requested.
        """
        if self.filesystem_type == "s3":
            output_bucket = full_output_path.split("/")[2]
            output_prefix = full_output_path.split("/", 3)[3]
        else:
            output_bucket = ""
            output_prefix = full_output_path

        if num_files and num_files != -1:
            input_df = input_df.coalesce(num_files)
        elif self.num_output_files:
            input_df = input_df.coalesce(self.num_output_files)

        if out_format == "parquet":
            # Write to parquet
            input_df = self._replace_special_chars_in_cols(input_df)
            input_df.write.mode("overwrite").parquet(os.path.join(full_output_path, "parquet"))
            prefix_with_format = os.path.join(output_prefix, "parquet")
        elif out_format == "csv":
            # TODO: Upstream perhaps it's a good idea for all transformers to guarantee
            # same output type, e.g. list of floats
            # Write to CSV
            def csv_row(data: Row):
                row_dict = data.asDict()
                row_vals = list(row_dict.values())
                if len(row_vals) > 1:
                    # In case there's more than one column
                    for d in row_vals:
                        # TODO: Maybe replace with inner separator
                        assert d != ",", "CSV column val contains comma"
                    return ",".join(str(d) for d in row_vals)
                else:
                    # Single column, but could be a multi-valued vector
                    return f"{row_vals[0]}"

            input_rdd = input_df.rdd.map(csv_row)  # type: RDD
            input_rdd.saveAsTextFile(os.path.join(full_output_path, "csv"))
            prefix_with_format = os.path.join(output_prefix, "csv")
        else:
            raise NotImplementedError(f"Unsupported output format: {out_format}")

        # Partition metadata requires full (relative) paths to files,
        # cannot just return the common prefix.
        # So we first get the full paths, then the common prefix,
        # then strip the common prefix from the full paths,
        # to leave paths relative to where the metadata will be written.
        if self.filesystem_type == "s3":
            object_key_list = s3_utils.list_s3_objects(output_bucket, prefix_with_format)
        else:
            object_key_list = [
                os.path.join(prefix_with_format, f) for f in os.listdir(prefix_with_format)
            ]

        assert (
            object_key_list
        ), f"No files found written under: {output_bucket}/{prefix_with_format}"

        # Only include data files and strip the common output path prefix from the key
        filtered_key_list = []
        if self.filesystem_type == "s3":
            # Get the S3 key prefix without the bucket
            common_prefix = self.output_prefix.split("/", maxsplit=3)[3]
        else:
            common_prefix = self.output_prefix

        for key in object_key_list:
            if key.endswith(".csv") or key.endswith(".parquet"):
                chars_to_skip = len(common_prefix)
                key_without_prefix = key[chars_to_skip:].lstrip("/")
                filtered_key_list.append(key_without_prefix)

        logging.info(
            "Wrote %d files to %s, (%d requested)",
            len(filtered_key_list),
            full_output_path,
            self.num_output_files,
        )
        return filtered_key_list

    def _add_node_mappings_to_metadata(self, metadata_dict: Dict) -> Dict:
        """
        Adds node mappings to the metadata dict that is eventually written to disk.
        """
        metadata_dict["node_id_mappings"] = {}
        for node_type in metadata_dict["node_type"]:
            node_mapping_metadata_dict = {
                "format": {"name": "parquet", "delimiter": ""},
                "data": self.node_mapping_paths[node_type],
            }
            metadata_dict["node_id_mappings"][node_type] = node_mapping_metadata_dict

        return metadata_dict

    def _add_row_counts_to_metadata(self, metadata_dict: Dict) -> Dict:
        """
        Add the number of rows per file for edge and node files generated to the metadata dict.
        Modifies the provided metadata_dict in-place and returns it.
        """
        logging.info("Adding row counts to metadata...")

        counter = ParquetRowCounter(metadata_dict, self.output_prefix, self.filesystem_type)

        return counter.add_row_counts_to_metadata(metadata_dict)

    @staticmethod
    def _get_missing_node_types(
        edge_configs: Sequence[EdgeConfig], node_configs: Sequence[NodeConfig]
    ) -> Set[str]:
        """Gets node types missing from the existing node configurations.

        Given the config objects for all nodes and edges, returns the
        node types that exist in the edges but are missing from the
        node configs.

        Parameters
        ----------
        edge_configs : Sequence[EdgeConfig]
            A sequence of all edge configuration objects.
        node_configs : Sequence[NodeConfig]
            A sequence of all node configuration objects.

        Returns
        -------
        Set[str]
            The set of node type names that appear in the edge configs but
            are missing from the node configs.
        """
        node_types = {node_config.ntype for node_config in node_configs}
        missing_node_types = set()
        for edge_config in edge_configs:
            if edge_config.src_ntype not in node_types:
                missing_node_types.add(edge_config.src_ntype)
            if edge_config.dst_ntype not in node_types:
                missing_node_types.add(edge_config.dst_ntype)

        return missing_node_types

    def create_node_id_maps_from_edges(
        self, edge_configs: Sequence[EdgeConfig], missing_node_types: Optional[Set[str]] = None
    ) -> None:
        """Creates node id mappings from edges.

        In the case where at least one node type files does exist,
        we need to extract node ids and create node mappings from the edge files.

        Given a list of edge configuration objects, this function will extract
        all the referenced node string ids for each node type and persist them
        so they can be used downstream during edge structure
        creation. The secondary argument missing_node_types lists node
        types that do not have their own node files, so
        we need to create node id mapping for them from the edge files.

        The function modifies the state of the DistHeterogeneousGraphLoader object by
        populating the node_mapping_paths member field.

        Parameters
        ----------
        edge_configs : Sequence[EdgeConfig]
            A list of EdgeConfig objects that contain all edge types in the graph.
        missing_node_types : Optional[Set[str]], optional
            An optional set of node types that do not have corresponding node files,
            by default None. We create mappings from the edges for those missing node
            types.
        """
        # TODO: If it is possible to have multiple edge files for a single node type,
        # we should use self.node_mapping_paths here.
        if missing_node_types is None:
            missing_node_types = set()
        node_type_to_mapping_df = {}  # type: Dict[str, DataFrame]
        for edge_config in edge_configs:
            src_type = edge_config.src_ntype
            src_col = edge_config.src_col
            dst_type = edge_config.dst_ntype
            dst_col = edge_config.dst_col
            rel_type = edge_config.rel_type

            # TODO: Read only column ids when we have that
            edges_df = self._read_edge_df(edge_config)
            edges_df.cache()

            for ntype, ncol in [(src_type, src_col), (dst_type, dst_col)]:
                # We only create mapping for node types that don't have a corresponding node file
                if ntype in missing_node_types:
                    if ntype in node_type_to_mapping_df:
                        mapping_df = node_type_to_mapping_df[ntype]
                        logging.info(
                            "Re-encountered node type '%s' in edge file for "
                            "'%s:%s:%s', "
                            "re-creating mapping for it using '%s' as join column...",
                            ntype,
                            src_type,
                            rel_type,
                            dst_type,
                            ncol,
                        )
                        node_type_to_mapping_df[ntype] = self._extend_mapping_from_edges(
                            mapping_df, edges_df, ncol
                        )
                    else:
                        node_type_to_mapping_df[ntype] = self._create_initial_mapping_from_edges(
                            edges_df, ncol
                        )
        # Write back each mapping and populate self.node_mapping_paths
        for ntype, mapping_df in node_type_to_mapping_df.items():
            assert ntype in missing_node_types
            self._write_nodeid_mapping_and_update_state(mapping_df, ntype)

    def _create_initial_mapping_from_edges(self, edges_df: DataFrame, node_col: str) -> DataFrame:
        assert isinstance(node_col, str), f"{node_col=} not of str type, got {type(node_col)=}"
        distinct_nodes_df = edges_df.select(node_col).distinct()
        # TODO: Instead of using the below that triggers a computation,
        # skip zipWithIndex and only trigger at write time?
        node_mapping_df = self.create_node_id_map_from_nodes_df(distinct_nodes_df, node_col)

        return node_mapping_df

    def _extend_mapping_from_edges(
        self, mapping_df: DataFrame, edges_df: DataFrame, node_col: str
    ) -> DataFrame:
        """Extends existing node mapping from incoming edge structure.

        Given an existing node ud mapping df, incoming edges_df and a node column,
        will extend the mapping to include any new node ids from the
        incoming edges DataFrame.

        Parameters
        ----------
        mapping_df : DataFrame
            An existing mapping from node string ids to numerical ids.
        edges_df : DataFrame
            An incoming edges structure that includes the node type.
        node_col : str
            The name of the column that corresponds to the node type
            in the incoming edges DataFrame.

        Returns
        -------
        DataFrame
            Returns a new mapping DataFrame, ordered by the updated
            integer node ids.
        """
        join_col = NODE_MAPPING_STR
        index_col = NODE_MAPPING_INT
        incoming_node_ids, join_col = spark_utils.safe_rename_column(
            edges_df.select(node_col).distinct().dropna(), node_col, join_col
        )
        if self.enable_assertions:
            null_counts = (
                edges_df.select(node_col)
                .select(
                    F.count(
                        F.when(
                            F.col(node_col).contains("None")
                            | F.col(node_col).contains("NULL")
                            | (F.col(node_col) == "")
                            | F.col(node_col).isNull()
                            | F.isnan(node_col),
                            node_col,
                        )
                    ).alias(node_col)
                )
                .collect()[0][0]
            )
            if null_counts > 0:
                raise ValueError(
                    f"Found {null_counts} null values in node column {node_col}"
                    " while extracting node ids from incoming edges structure "
                    f"with cols: {edges_df.columns}"
                )

        node_df_with_ids = incoming_node_ids.join(mapping_df, on=join_col, how="fullouter")

        # If we have new node ids we should re-index
        # TODO: The counts trigger computations, should we avoid this?
        if node_df_with_ids.count() > mapping_df.count():
            logging.info(
                "Node mapping count mismatch, for node_col: %s, re-assigning int ids...", node_col
            )
            node_rdd_with_ids = (
                node_df_with_ids.select(join_col)
                .rdd.zipWithIndex()
                .map(lambda rdd_row: (list(rdd_row[0]) + [rdd_row[1]]))
            )

            map_schema = StructType(
                [
                    StructField(join_col, StringType(), True),
                    StructField(index_col, IntegerType(), True),
                ]
            )

            node_df_with_ids = self.spark.createDataFrame(node_rdd_with_ids, map_schema)

        # TODO: Do not perform ordering here but only once at end?
        # Would it actually make a difference or is this optimized away?
        return node_df_with_ids.orderBy(NODE_MAPPING_INT)

    def create_node_id_map_from_nodes_df(self, node_df: DataFrame, node_col: str) -> DataFrame:
        """Given a nodes DF will attach sequential node ids, order the DF by them,
        and return a new DF with int id attached.

        The mapping DF will have two columns, `node_str_id`
        containing the original id, and `node_int_id`
        that has the corresponding index.

        Parameters
        ----------
        node_df : DataFrame
            DataFrame that contains all the data for a node type
        node_col : str
            The column in the DataFrame that corresponds to the string ID of the nodes

        Returns
        -------
        DataFrame
            The input nodes DataFrame, with additional integer ids attached for each row, ordered
            by the integer ids.
        """
        original_schema = node_df.schema

        assert node_col in node_df.columns, f"{node_col=} not in {node_df.columns}"

        # This will trigger a computation
        node_rdd_with_ids = node_df.rdd.zipWithIndex()

        node_id_col = f"{node_col}-int_id"
        new_schema = original_schema.add(StructField(node_id_col, IntegerType(), False))

        node_rdd_with_ids = node_rdd_with_ids.map(
            lambda rdd_row: (list(rdd_row[0]) + [rdd_row[1]])  # type: ignore
        )

        node_df_with_ids = self.spark.createDataFrame(node_rdd_with_ids, new_schema)  # type: ignore
        # Order by the index and rename columns before returning
        # TODO: Check if cols exist before renaming? Create a safe_rename function
        assert isinstance(node_col, str)
        assert {node_col, node_id_col}.issubset(
            set(node_df_with_ids.columns)
        ), f"{node_col=} or {node_id_col=} not in {node_df_with_ids.columns=}"
        node_df_with_ids = node_df_with_ids.withColumnRenamed(
            node_col, NODE_MAPPING_STR
        ).withColumnRenamed(node_id_col, NODE_MAPPING_INT)

        return node_df_with_ids.orderBy(NODE_MAPPING_INT)

    def _write_nodeid_mapping_and_update_state(
        self, node_df_with_ids: DataFrame, node_type: str
    ) -> None:
        """
        Given a nodes mapping DF write it to the output path under the correct
        node_type sub-directory.

        Also modifies the loader's state to add the mapping path to
        the node_mapping_paths member variable.
        """
        mapping_output_path = f"{self.output_prefix}/node_id_mappings/{node_type}"

        # TODO: For node-file-exists path: Test to see if it's better to keep these in memory
        # until needed instead of writing out now i.e. we can maintain a dict of DFs instead
        # of filepaths and only materialize when the edge mapping happens.
        path_list = self._write_df(
            node_df_with_ids.select([NODE_MAPPING_STR, NODE_MAPPING_INT]),
            mapping_output_path,
            out_format="parquet",
        )
        self.node_mapping_paths[node_type] = path_list

    def process_node_data(self, node_configs: Sequence[NodeConfig]) -> Dict:
        """Given a list of node config objects will perform the processing steps for each feature,
        write the output to S3 and return the corresponding dict entry for the node_data key of
        the output metadata.json.

        As an additional side-effect the function populates the values of
        `self.node_chunk_counts` for each edge type.

        Parameters
        ----------
        node_configs : Sequence[NodeConfig]
            A list of `NodeConfig` objects that contain the information for each node type
            to be processed.

        Returns
        -------
        Dict
            A dict entry for the node_data key of the output metadata.json.
        """
        node_data_dict = {}  # type: Dict[str, Dict]
        self.graph_info["nfeat_size"] = {}
        self.graph_info["ntype_label"] = []
        self.graph_info["ntype_label_property"] = []
        for node_config in node_configs:
            logging.info("node_config: %s", node_config)
            files = node_config.files
            file_paths = [f"{self.input_prefix}/{f}" for f in files]

            node_type = node_config.ntype
            node_col = node_config.node_col

            read_nodefile_start = perf_counter()
            # TODO: Maybe we use same enforced type for Parquet and CSV
            # to ensure consistent behavior downstream?
            if node_config.format == "csv":
                separator = node_config.separator
                node_file_schema = schema_utils.parse_node_file_schema(node_config)
                nodes_df_untyped = self.spark.read.csv(path=file_paths, sep=separator, header=True)
                nodes_df_untyped = nodes_df_untyped.select(node_file_schema.fieldNames())
                # Select only the columns referenced in the config
                # and cast each column to the correct type
                # TODO: It's possible that columns have different types from the
                # expected type in the config. We need to handle conversions when needed.
                nodes_df = nodes_df_untyped.select(
                    *(
                        F.col(col_name).cast(col_field.dataType).alias(col_name)
                        for col_name, col_field in zip(
                            node_file_schema.fieldNames(), node_file_schema.fields
                        )
                    )
                )
            else:
                nodes_df = self.spark.read.parquet(*file_paths)

            # TODO: Assert columns from config exist in the nodes df
            self.timers["node_read_file"] += perf_counter() - read_nodefile_start

            node_id_map_start_time = perf_counter()
            if node_type in self.node_mapping_paths:
                logging.warning(
                    "Encountered node type '%s' that already has "
                    "a mapping, skipping map creation",
                    node_type,
                )
                # Use the existing mapping to convert the ids from str-to-int
                mapping_df = self.spark.read.parquet(
                    *[
                        f"{self.output_prefix}/{map_path}"
                        for map_path in self.node_mapping_paths[node_type]
                    ]
                )
                node_df_with_ids = mapping_df.join(
                    nodes_df, mapping_df[NODE_MAPPING_STR] == nodes_df[node_col], "left"
                )
                if self.enable_assertions:
                    nodes_df_count = nodes_df.count()
                    mapping_df_count = mapping_df.count()
                    assert nodes_df_count == mapping_df_count, (
                        f"Nodes df count ({nodes_df_count}) does not match "
                        f"mapping df count ({mapping_df_count})"
                    )
                nodes_df = node_df_with_ids.withColumnRenamed(node_col, NODE_MAPPING_STR).orderBy(
                    NODE_MAPPING_INT
                )
            else:
                nodes_df = self.create_node_id_map_from_nodes_df(nodes_df, node_col)
                self._write_nodeid_mapping_and_update_state(nodes_df, node_type)

            nodes_df.cache()
            self.timers["node_map_creation"] += perf_counter() - node_id_map_start_time

            node_type_metadata_dicts = {}
            if node_config.feature_configs is not None:
                process_node_features_start = perf_counter()
                node_type_feature_metadata, ntype_feat_sizes = self._process_node_features(
                    node_config.feature_configs, nodes_df, node_type
                )
                self.graph_info["nfeat_size"].update({node_type: ntype_feat_sizes})
                node_type_metadata_dicts.update(node_type_feature_metadata)
                self.timers["_process_node_features"] += (
                    perf_counter() - process_node_features_start
                )
            if node_config.label_configs is not None:
                process_node_labels_start = perf_counter()
                node_type_label_metadata = self._process_node_labels(
                    node_config.label_configs, nodes_df, node_type
                )
                node_type_metadata_dicts.update(node_type_label_metadata)
                self.graph_info["ntype_label"].append(node_type)
                self.graph_info["ntype_label_property"].append(node_config.label_configs[0].cols[0])
                self.timers["_process_node_labels"] += perf_counter() - process_node_labels_start

            if node_type_metadata_dicts:
                if node_type in node_data_dict:
                    node_data_dict[node_type].update(node_type_metadata_dicts)
                else:
                    node_data_dict[node_type] = node_type_metadata_dicts
            nodes_df.unpersist()

        logging.info("Finished processing node features")
        return node_data_dict

    def _process_node_features(
        self, feature_configs: Sequence[FeatureConfig], nodes_df: DataFrame, node_type: str
    ) -> Tuple[Dict, Dict]:
        """Transform node features and write to storage.

        Given a list of feature config objects for a particular node type,
        will perform the processing steps for each feature,
        write the output to storage and return the corresponding dict
        entry for the node type to build up metadata.json.

        Parameters
        ----------
        feature_configs : Sequence[FeatureConfig]
            A list of feature configuration objects
            describing the features for the node type.
        nodes_df : DataFrame
            A DataFrame containing the features for the given node type.
        node_type : str
            The node type name.

        Returns
        -------
        Tuple[Dict, Dict]
            A tuple with two dicts, both dicts have names as their keys.
            The first dict has the output metadata for the feature
            as  values, and the second has the lengths of the
            vector representations of the features as values.
        """
        node_type_feature_metadata = {}
        ntype_feat_sizes = {}  # type: Dict[str, int]
        for feat_conf in feature_configs:
            logging.info(
                "Processing feat_name: '%s' feat_cols: %s", feat_conf.feat_name, feat_conf.cols
            )

            transformer = DistFeatureTransformer(feat_conf)

            transformed_feature_df = transformer.apply_transformation(nodes_df)

            # TODO: Remove hack with [feat_conf.feat_name]
            for feat_name, feat_col in zip([feat_conf.feat_name], feat_conf.cols):
                node_transformation_start = perf_counter()
                single_feature_df = transformed_feature_df.select(feat_col).withColumnRenamed(
                    feat_col, feat_name
                )

                feature_output_path = os.path.join(
                    self.output_prefix, f"node_data/{node_type}-{feat_name}"
                )

                logging.info(
                    "Writing output for feat_name: '%s' to %s", feat_name, feature_output_path
                )
                path_list = self._write_df(
                    single_feature_df, feature_output_path, out_format="parquet"
                )

                node_feature_metadata_dict = {
                    "format": {"name": FORMAT_NAME, "delimiter": DELIMITER},
                    "data": path_list,
                }
                node_type_feature_metadata[feat_name] = node_feature_metadata_dict

                self.timers[f"{transformer.get_transformation_name()}-{node_type}-{feat_name}"] = (
                    perf_counter() - node_transformation_start
                )

                feat_val = single_feature_df.take(1)[0].asDict()[feat_name]

                if isinstance(feat_val, (int, float)):
                    nfeat_size = 1
                else:
                    nfeat_size = len(feat_val)

                ntype_feat_sizes.update({feat_name: nfeat_size})

        return node_type_feature_metadata, ntype_feat_sizes

    def _process_node_labels(
        self, label_configs: Sequence[LabelConfig], nodes_df: DataFrame, node_type: str
    ) -> Dict:
        """
        Given a list of label config objects will perform the processing steps for each label,
        write the output to S3 and return the corresponding dict entry for the labels of node_type
        for metadata.json
        """
        node_type_label_metadata = {}
        for label_conf in label_configs:
            self.graph_info["task_type"] = (
                "node_class" if label_conf.task_type == "classification" else "node_regression"
            )
            self.graph_info["is_multilabel"] = label_conf.multilabel
            node_label_loader = DistLabelLoader(label_conf, self.spark)
            transformed_label = node_label_loader.process_label(nodes_df)
            self.graph_info["label_map"] = node_label_loader.label_map

            label_output_path = (
                f"{self.output_prefix}/node_data/{node_type}-label-{label_conf.cols[0]}"
            )

            path_list = self._write_df(transformed_label, label_output_path)

            label_metadata_dict = {
                "format": {"name": FORMAT_NAME, "delimiter": DELIMITER},
                "data": path_list,
            }
            node_type_label_metadata[label_conf.cols[0]] = label_metadata_dict

            self._update_label_properties(node_type, nodes_df, label_conf)

            split_masks_output_prefix = f"{self.output_prefix}/node_data/{node_type}"
            logging.info("Creating train/test/val split for node type %s...", node_type)
            if label_conf.split_rate:
                split_rates = SplitRates(
                    train_rate=label_conf.split_rate["train"],
                    val_rate=label_conf.split_rate["val"],
                    test_rate=label_conf.split_rate["test"],
                )
            else:
                split_rates = None
            label_split_dicts = self._create_split_files_from_rates(
                nodes_df, label_conf.cols[0], split_rates, split_masks_output_prefix
            )
            node_type_label_metadata.update(label_split_dicts)

        return node_type_label_metadata

    def write_edge_structure(
        self, edge_df: DataFrame, edge_config: EdgeConfig
    ) -> Tuple[DataFrame, Sequence[str], Sequence[str]]:
        """Write edge structure to storage

        Given a DataFrame of edges and corresponding edge configuration will
        use the node str-to-int-id mapping to convert the node str ids to
        ints and write the edge structure as a file with two
        columns: "src_int_id","dst_int_id".

        If `self.add_reverse_edges` is True, it it will also write the
        reverse edge type with the src and dst columns reversed.

        Parameters
        ----------
        edge_df : DataFrame
            A DataFrame of edges, with columns for the source and destination node
            types.
        edge_config : EdgeConfig
            The edge configuration object.

        Returns
        -------
        Tuple[DataFrame, Sequence[str], Sequence[str]]
            A tuple containing the original edge dataframe with ids converted to int,
            followed by two lists of the files that were written by Spark, as S3 URIs.
            The first list contains the original edge files, the second is the reversed
            edge files, will be empty if `self.add_reverse_edges` is False.
        """
        src_col = edge_config.src_col
        src_ntype = edge_config.src_ntype
        dst_col = edge_config.dst_col
        dst_ntype = edge_config.dst_ntype
        edge_type = (
            f"{edge_config.src_ntype}:{edge_config.get_relation_name()}:{edge_config.dst_ntype}"
        )
        rev_edge_type = (
            f"{edge_config.dst_ntype}:rev-{edge_config.get_relation_name()}:{edge_config.src_ntype}"
        )

        src_node_id_mapping = (
            self.spark.read.parquet(
                *[
                    os.path.join(self.output_prefix, src_path)
                    for src_path in self.node_mapping_paths[src_ntype]
                ]
            )
            .withColumnRenamed(NODE_MAPPING_INT, "src_int_id")
            .withColumnRenamed(NODE_MAPPING_STR, "src_str_id")
        )
        # Join incoming edge df with mapping df to transform source str-ids to int ids
        edge_df_with_int_src = src_node_id_mapping.join(
            edge_df,
            src_node_id_mapping["src_str_id"] == edge_df[src_col],
            how="inner",
        )

        if self.enable_assertions:
            incoming_edge_count = edge_df.count()
            intermediate_edge_count = edge_df_with_int_src.count()
            if incoming_edge_count != intermediate_edge_count:
                logging.fatal(
                    "Incoming and outgoing edge counts do not match for "
                    "%s when joining %s with src_str_id! "
                    "%d in != %d out",
                    edge_type,
                    src_col,
                    incoming_edge_count,
                    intermediate_edge_count,
                )

        dst_node_id_mapping = (
            self.spark.read.parquet(
                *[
                    f"{self.output_prefix}/{dst_path}"
                    for dst_path in self.node_mapping_paths[dst_ntype]
                ]
            )
            .withColumnRenamed(NODE_MAPPING_INT, "dst_int_id")
            .withColumnRenamed(NODE_MAPPING_STR, "dst_str_id")
        )
        # Join the newly created src-int-id edge df with mapping
        # df to transform destination str-ids to int ids
        edge_df_with_int_ids = dst_node_id_mapping.join(
            edge_df_with_int_src,
            dst_node_id_mapping["dst_str_id"] == edge_df_with_int_src[dst_col],
            how="inner",
        )

        # After the two joins, the result is the incoming edge
        # df now has the src and dst node ids as int ids
        # TODO: We need to repartition to ensure same file count for
        # all downstream DataFrames, but it causes a full shuffle.
        # Can it be avoided?
        edge_df_with_int_ids = edge_df_with_int_ids.repartition(self.num_output_files)
        edge_df_with_int_ids_and_all_features = edge_df_with_int_ids
        edge_df_with_only_int_ids = edge_df_with_int_ids.select(["src_int_id", "dst_int_id"])

        edge_structure_path = os.path.join(self.output_prefix, f"edges/{edge_type}")
        logging.info("Writing edge structure for edge type %s...", edge_type)
        path_list = self._write_df(edge_df_with_only_int_ids, edge_structure_path)

        if self.add_reverse_edges:
            reversed_edges = edge_df_with_only_int_ids.select("dst_int_id", "src_int_id")
            reversed_edge_structure_path = os.path.join(
                self.output_prefix, f"edges/{rev_edge_type}"
            )
            logging.info("Writing edge structure for reverse edge type %s...", rev_edge_type)
            reverse_path_list = self._write_df(reversed_edges, reversed_edge_structure_path)
        else:
            reverse_path_list = []

        # Verify counts
        if self.enable_assertions:
            outgoing_edge_count = edge_df_with_only_int_ids.count()
            if incoming_edge_count != outgoing_edge_count:
                logging.fatal(
                    "Incoming and outgoing edge counts do not match for '%s'! %d in != %d out",
                    edge_type,
                    incoming_edge_count,
                    outgoing_edge_count,
                )
        return edge_df_with_int_ids_and_all_features, path_list, reverse_path_list

    def _read_edge_df(self, edge_config: EdgeConfig) -> DataFrame:
        # TODO: Allow reading only id-cols for mapping building to allow faster reads
        files = edge_config.files
        separator = edge_config.separator
        file_paths = [os.path.join(self.input_prefix, f) for f in files]

        if edge_config.format == "csv":
            edge_file_schema = schema_utils.parse_edge_file_schema(edge_config)
            edges_df_untyped = self.spark.read.csv(path=file_paths, sep=separator, header=True)
            edges_df_untyped = edges_df_untyped.select(edge_file_schema.fieldNames())
            # After reading the CSV without types, we cast each column to its detected type
            edges_df = edges_df_untyped.select(
                *(
                    F.col(col_name).cast(col_field.dataType).alias(col_name)
                    for col_name, col_field in zip(
                        edge_file_schema.fieldNames(), edge_file_schema.fields
                    )
                )
            )
        else:
            # TODO: Need tests for Parquet input
            edges_df = self.spark.read.parquet(*file_paths)

        return edges_df

    def process_edge_data(self, edge_configs: Sequence[EdgeConfig]) -> Tuple[Dict, Dict]:
        """Given a list of edge config objects will extract and write the edge structure
        data to S3 or local storage, perform the processing steps for each feature,
        and return a tuple with two dict entries for the metadata.json file.
        The first element is aimed for the "edge_data" key that describes the edge features,
        and the second being the "edges" key that describes the edge structures.



        Parameters
        ----------
        edge_configs : Sequence[EdgeConfig]
            A list of `EdgeConfig` objects describing all the edge types in the graph.

        Returns
        -------
        Tuple[Dict, Dict]
            A tuple of two dicts, the first containing the values for the "edge_data"
            key and the second the values of the "edges" key in the output metadata.json.
        """
        # iterates over entries of the edge section in the export config
        edge_data_dict = {}
        edges_dict = {}
        logging.info("Processing edge data...")
        self.graph_info["etype_label"] = []
        self.graph_info["etype_label_property"] = []
        for edge_config in edge_configs:
            read_edges_start = perf_counter()
            edges_df = self._read_edge_df(edge_config)
            # TODO: Assert columns from config exist in the edge df

            # This will throw an "already cached" warning if
            # we created mappings from the edges alone
            edges_df.cache()
            edge_type = (
                f"{edge_config.src_ntype}:{edge_config.get_relation_name()}:{edge_config.dst_ntype}"
            )
            reverse_edge_type = (
                f"{edge_config.dst_ntype}"
                f":rev-{edge_config.get_relation_name()}"
                f":{edge_config.src_ntype}"
            )
            logging.info("Processing edge type '%s'...", edge_type)

            # The following performs the str-to-node-id mapping conversion
            # on the edge files and writes them to S3, along with their reverse.
            edges_df, edge_structure_path_list, reverse_edge_path_list = self.write_edge_structure(
                edges_df, edge_config
            )
            self.timers["read_edges_and_write_structure"] += perf_counter() - read_edges_start

            edges_metadata_dict = {
                "format": {"name": FORMAT_NAME, "delimiter": DELIMITER},
                "data": edge_structure_path_list,
            }
            edges_dict[edge_type] = edges_metadata_dict

            if self.add_reverse_edges:
                reverse_edges_metadata_dict = {
                    "format": {"name": FORMAT_NAME, "delimiter": DELIMITER},
                    "data": reverse_edge_path_list,
                }

                edges_dict[reverse_edge_type] = reverse_edges_metadata_dict

            # Without features or labels
            if edge_config.feature_configs is None and edge_config.label_configs is None:
                logging.info("No features or labels for edge type: %s", edge_type)
            # With features or labels
            else:
                relation_col = edge_config.rel_col
                edge_type_metadata_dicts = {}

                if edge_config.feature_configs is not None:
                    edge_feature_start = perf_counter()
                    edge_feature_metadata_dicts, etype_feat_sizes = self._process_edge_features(
                        edge_config.feature_configs, edges_df, edge_type
                    )
                    self.graph_info["efeat_size"].update({edge_type: etype_feat_sizes})
                    edge_type_metadata_dicts.update(edge_feature_metadata_dicts)
                    self.timers["_process_edge_features"] += perf_counter() - edge_feature_start
                if edge_config.label_configs is not None:
                    if relation_col is None or relation_col == "":
                        edge_label_start = perf_counter()
                        # All edges have the same relation type
                        label_metadata_dicts = self._process_edge_labels(
                            edge_config.label_configs, edges_df, edge_type, edge_config.rel_type
                        )
                        edge_type_metadata_dicts.update(label_metadata_dicts)
                        self.graph_info["etype_label"].append(edge_type)
                        self.graph_info["etype_label"].append(reverse_edge_type)
                        if edge_config.label_configs[0].task_type != "link_prediction":
                            self.graph_info["etype_label_property"].append(
                                edge_config.label_configs[0].cols[0]
                            )

                        if self.add_reverse_edges:
                            # For reverse edges only the label metadata
                            # (labels + split masks) are relevant.
                            edge_data_dict[reverse_edge_type] = label_metadata_dicts
                        self.timers["_process_edge_labels"] += perf_counter() - edge_label_start
                    else:
                        # Different edges can have different relation types
                        # TODO: For each rel_type we need get the label and output it separately.
                        # We'll create one file output per rel_type, with the labels for each type
                        raise NotImplementedError(
                            "Currently we do not support loading edge "
                            "labels with multiple edge relation types"
                        )
                if edge_type_metadata_dicts:
                    edge_data_dict[edge_type] = edge_type_metadata_dicts
                edges_df.unpersist()

        logging.info("Finished processing edge features")
        return edge_data_dict, edges_dict

    def _process_edge_features(
        self, feature_configs: Sequence[FeatureConfig], edges_df: DataFrame, edge_type: str
    ) -> Tuple[Dict, Dict]:
        """Process edge features.

        Parameters
        ----------
        feature_configs : Sequence[FeatureConfig]
            A list of feature configurations for the current edge type.
        edges_df : DataFrame
            The edges DataFrame
        edge_type : str
            The name of the edge type

        Returns
        -------
        Tuple[Dict, Dict]
            A tuple with two dicts: the first are the metadata dicts
            for the edge features, the second a dict of feature sizes
            for each feature, which tells us the column size of the
            encoded features.
        """
        edge_feature_metadata_dicts = {}
        etype_feat_sizes = {}  # type: Dict[str, int]
        for feat_conf in feature_configs:
            logging.info(
                "Processing edge feature(s) '%s' for edge type '%s'...",
                feat_conf.feat_name,
                edge_type,
            )
            transformer = DistFeatureTransformer(feat_conf)

            transformed_feature_df = transformer.apply_transformation(edges_df)

            for feat_name, feat_col in zip(feat_conf.feat_name, feat_conf.cols):
                edge_feature_start = perf_counter()
                single_feature_df = transformed_feature_df.select(feat_col).withColumnRenamed(
                    feat_col, feat_name
                )
                feature_output_path = os.path.join(
                    self.output_prefix, f"edge_data/{edge_type}-{feat_name}"
                )

                path_list = self._write_df(
                    single_feature_df, feature_output_path, out_format=FORMAT_NAME
                )

                edge_feature_metadata_dict = {
                    "format": {"name": FORMAT_NAME, "delimiter": DELIMITER},
                    "data": path_list,
                }
                edge_feature_metadata_dicts[feat_name] = edge_feature_metadata_dict

                self.timers[f"{transformer.get_transformation_name()}-{edge_type}-{feat_name}"] = (
                    perf_counter() - edge_feature_start
                )

                feat_val = single_feature_df.take(1)[0].asDict()[feat_name]

                if isinstance(feat_val, numbers.Number):
                    efeat_size = 1
                else:
                    efeat_size = len(feat_val)

                etype_feat_sizes.update({feat_name: efeat_size})

        return edge_feature_metadata_dicts, etype_feat_sizes

    def _process_edge_labels(
        self,
        label_configs: Sequence[LabelConfig],
        edges_df: DataFrame,
        edge_type: str,
        rel_type_prefix: str,
    ) -> Dict:
        """
        Process edge labels and create train/test/val splits.
        Transforms classification labels to single float values and multi-label ones
        to a multi-hot float-vector representation.
        For regression, float values are passed through as is.

        Parameters
        ----------
        label_configs : Sequence[LabelConfig]
            The labels to be processed.
        edges_df : DataFrame
            The edges dataframe containing the labels as columns.
        edge_type : str
            The edge type, in <src_type:relation:dst_type> format.
        rel_type_prefix : str
            The prefix of the relation type.
            This is used to create the output path for the labels.
            For example, if the relation type is "author" then the output path for the labels
            will be "<output_prefix>/edge_data/<edge_type>-label-author".

        Returns
        -------
        The entries {<label-name> : {'format' : {'name' : ..., 'delimiter' : ...}},  ...}
        for the metadata dictionary to be added to the current edge type.
        Labels include "label" in their S3 path to tell them apart from "regular"
        edge features.
        """
        label_metadata_dicts = {}
        for label_conf in label_configs:
            if label_conf.task_type != "link_prediction":
                edge_label_loader = DistLabelLoader(label_conf, self.spark)
                self.graph_info["task_type"] = (
                    "edge_class" if label_conf.task_type == "classification" else "edge_regression"
                )
                if label_conf.task_type == "classification":
                    self.graph_info["is_multilabel"] = "True" if label_conf.multilabel else "False"
                    self.graph_info["label_map"] = edge_label_loader.label_map

                logging.info(
                    "Processing edge label(s) '%s' for edge type '%s'...",
                    label_conf.cols[0],
                    edge_type,
                )
                transformed_label = edge_label_loader.process_label(edges_df)

                label_output_path = os.path.join(
                    self.output_prefix, f"edge_data/{edge_type}-label-{rel_type_prefix}"
                )

                path_list = self._write_df(transformed_label, label_output_path)

                label_metadata_dict = {
                    "format": {"name": FORMAT_NAME, "delimiter": DELIMITER},
                    "data": path_list,
                }
                label_metadata_dicts[label_conf.cols[0]] = label_metadata_dict

                self._update_label_properties(edge_type, edges_df, label_conf)
            else:
                self.graph_info["task_type"] = "link_predict"
                logging.info(
                    "Skipping processing label for '%s' because task is link prediction",
                    rel_type_prefix,
                )

            split_masks_output_prefix = os.path.join(self.output_prefix, f"edge_data/{edge_type}")
            logging.info("Creating train/test/val split for edge type %s...", edge_type)
            if label_conf.split_rate:
                split_rates = SplitRates(
                    train_rate=label_conf.split_rate["train"],
                    val_rate=label_conf.split_rate["val"],
                    test_rate=label_conf.split_rate["test"],
                )
            else:
                split_rates = None
            label_split_dicts = self._create_split_files_from_rates(
                edges_df, label_conf.cols[0], split_rates, split_masks_output_prefix
            )
            label_metadata_dicts.update(label_split_dicts)
            # TODO: Support custom_split_filenames

        return label_metadata_dicts

    def _update_label_properties(
        self, node_or_edge_type: str, original_labels: DataFrame, label_config: LabelConfig
    ) -> None:
        """Extracts and stores statistics about labels.

        For a given node or edge type, label configuration and the original
        (non-transformed) label DataFrame, extract some statistics
        and store them in `self.label_properties` to be written to
        storage later. The statistics can be used by downstream tasks.

        Parameters
        ----------
        node_or_edge_type : str
            The node or edge type name for which the label information is being updated.
        original_labels : DataFrame
            The original (non-transformed) label DataFrame.
        label_config : LabelConfig
            The label configuration object.

        Raises
        ------
        RuntimeError
            In case an invalid task type name is specified in the label config.
        """
        label_col = label_config.cols[0]
        if not node_or_edge_type in self.label_properties:
            self.label_properties[node_or_edge_type] = Counter()
        # TODO: Something wrong with the assignment here? Investigate
        self.label_properties[node_or_edge_type][COLUMN_NAME] = label_col

        if label_config.task_type == "regression":
            label_min = original_labels.agg(F.min(label_col).alias("min")).collect()[0]["min"]
            label_max = original_labels.agg(F.max(label_col).alias("max")).collect()[0]["max"]

            current_min = self.label_properties[node_or_edge_type].get(MIN_VALUE, float("inf"))
            current_max = self.label_properties[node_or_edge_type].get(MAX_VALUE, float("-inf"))
            self.label_properties[node_or_edge_type][MIN_VALUE] = min(current_min, label_min)
            self.label_properties[node_or_edge_type][MAX_VALUE] = max(current_max, label_max)
        elif label_config.task_type == "classification":
            # Collect counts per label
            if label_config.multilabel:
                assert label_config.separator
                # Spark's split function uses a regexp so we
                # need to escape special chars to be used as separators
                if label_config.separator in SPECIAL_CHARACTERS:
                    separator = f"\\{label_config.separator}"
                else:
                    separator = label_config.separator
                label_counts_df = (
                    original_labels.select(label_col)
                    .withColumn(label_col, F.explode(F.split(F.col(label_col), separator)))
                    .groupBy(label_col)
                    .count()
                )
            else:  # Single-label classification
                label_counts_df = original_labels.groupBy(label_col).count()

            # TODO: Verify correctness here
            label_counts_dict: Counter = Counter()
            for label_count_row in label_counts_df.collect():
                row_dict = label_count_row.asDict()
                # Label value to count
                label_counts_dict[row_dict[label_col]] = row_dict["count"]

            self.label_properties[node_or_edge_type][VALUE_COUNTS] = (
                self.label_properties.get(VALUE_COUNTS, Counter()) + label_counts_dict
            )
        else:
            raise RuntimeError(f"Invalid task type: {label_config.task_type}")

    def _create_split_files_from_rates(
        self,
        input_df: DataFrame,
        label_column: str,
        split_rates: Optional[SplitRates],
        output_path: str,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Given an input dataframe and a list of split rates creates the
        split masks and writes them to S3 and returns the corresponding
        metadata.json dict elements.

        Parameters
        ----------
        input_df: DataFrame
            Input dataframe for which we will create split masks.
        label_column: str
            The name of the label column. If provided, the values in the column
            need to be not null for the data point to be included in one of the masks.
            If an empty string, all rows in the input_df are included in one of train/val/test sets.
        split_rates: Optional[SplitRates]
            A SplitRates object indicating the train/val/test split rates.
            If None, a default split rate of 0.9:0.05:0.05 is used.
        output_path: str
            The output path under which we write the masks.
        seed: int
            An optional random seed for reproducibility.

        Returns
        -------
        The metadata dict elements for the train/test/val masks, to be added to the caller's
        edge/node type metadata.
        """
        # If the user did not provide a split rate we use a default
        split_metadata = {}
        if split_rates is None:
            split_rates = SplitRates(train_rate=0.8, val_rate=0.1, test_rate=0.1)
        else:
            if sum(split_rates.tolist()) != 1.0:
                raise RuntimeError(f"Provided split rates  do not sum to 1: {split_rates}")

        split_list = split_rates.tolist()
        logging.info(
            "Creating split files for label column '%s' with split rates: %s",
            label_column,
            split_list,
        )

        rng = default_rng(seed=seed)

        # We use multinomial sampling to create a one-hot
        # vector indicating train/test/val membership
        def multinomial_sample(label_col: str) -> Sequence[int]:
            if label_col in {"", "None", "NaN", None}:
                return [0, 0, 0]
            return rng.multinomial(1, split_list).tolist()

        group_col_name = "sample_boolean_mask"  # TODO: Ensure uniqueness of column?

        # TODO: Use PandasUDF and check if it is faster than UDF
        split_group = F.udf(multinomial_sample, ArrayType(IntegerType()))
        # Convert label col to string and apply UDF
        # to create one-hot vector indicating train/test/val membership
        input_col = F.col(label_column).astype("string") if label_column else F.lit("dummy")
        int_group_df = input_df.select(split_group(input_col).alias(group_col_name))

        def create_metadata_entry(path_list):
            return {"format": {"name": FORMAT_NAME, "delimiter": DELIMITER}, "data": path_list}

        def write_mask(kind: str, mask_df: DataFrame) -> Sequence[str]:
            out_path_list = self._write_df(
                mask_df.select(F.col(f"{kind}_mask").cast(ByteType()).alias(f"{kind}_mask")),
                f"{output_path}-{kind}-mask",
            )
            return out_path_list

        train_mask_df = int_group_df.withColumn("train_mask", F.col(group_col_name)[0])
        out_path_list = write_mask("train", train_mask_df)
        split_metadata["train_mask"] = create_metadata_entry(out_path_list)

        val_mask_df = int_group_df.withColumn("val_mask", F.col(group_col_name)[1])
        out_path_list = write_mask("val", val_mask_df)
        split_metadata["val_mask"] = create_metadata_entry(out_path_list)

        test_mask_df = int_group_df.withColumn("test_mask", F.col(group_col_name)[2])
        out_path_list = write_mask("test", test_mask_df)
        split_metadata["test_mask"] = create_metadata_entry(out_path_list)

        return split_metadata

    def load(self):
        self.process_and_write_graph_data(self._data_configs)
