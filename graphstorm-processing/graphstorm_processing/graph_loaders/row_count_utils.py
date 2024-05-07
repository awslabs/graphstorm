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

This module is used to determine row counts for Parquet files.
"""

import logging
import os
from typing import Dict, List, Sequence

from joblib import Parallel, delayed
import pyarrow.parquet as pq
from pyarrow import fs

from ..data_transformations import s3_utils  # pylint: disable=relative-beyond-top-level


class ParquetRowCounter:
    """We use this class to determine row counts for Parquet files

    The row counts are needed to ensure our output is consistent
    with the assumptions of the DGL partitioning pipeline.

    Parameters
    ----------
    metadata_dict : dict
        The original metadata dict, which we will modify in-place.
    output_prefix : str
        The prefix where the output data are expected.
    filesystem_type : str
        The filesystem type. Can be 'local' or 's3'.
    """

    def __init__(self, metadata_dict: dict, output_prefix: str, filesystem_type: str):
        self.output_prefix = output_prefix
        self.filesystem_type = filesystem_type
        self.metadata_dict = metadata_dict
        if self.filesystem_type == "s3":
            output_bucket, _ = s3_utils.extract_bucket_and_key(output_prefix)
            bucket_region = s3_utils.get_bucket_region(output_bucket)
            # Increase default retries because we are likely to run into
            # throttling errors
            self.pyarrow_fs = fs.S3FileSystem(
                region=bucket_region,
                retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=10),
            )
        else:
            self.pyarrow_fs = fs.LocalFileSystem()

    def add_row_counts_to_metadata(self, metadata_dict: dict) -> dict:
        """Add the number of rows per file for edge and node files generated to the metadata dict.
        Modifies the provided metadata_dict in-place and returns it.

        Parameters
        ----------
        metadata_dict : dict
            The input metadata dict, which contains all the Parquet files that we have created
            during processing.

        Returns
        -------
        dict
            The modified data dict, now including a 'row_counts' key for each data file entry,
            and two new top-level keys, 'num_nodes_per_type' and 'num_edges_per_type'.
        """
        all_edges_counts = self._add_counts_for_graph_structure(
            top_level_key="edges", edge_or_node_type_key="edge_type"
        )
        self._add_counts_for_features(top_level_key="edge_data", edge_or_node_type_key="edge_type")

        all_node_mapping_counts = self._add_counts_for_graph_structure(
            top_level_key="raw_id_mappings", edge_or_node_type_key="node_type"
        )
        self._add_counts_for_features(top_level_key="node_data", edge_or_node_type_key="node_type")

        metadata_dict["num_nodes_per_type"] = [
            sum(type_rows) for type_rows in all_node_mapping_counts
        ]
        metadata_dict["num_edges_per_type"] = [sum(type_rows) for type_rows in all_edges_counts]

        return metadata_dict

    def get_row_count_for_parquet_file(self, relative_parquet_file_path: str) -> int:
        """Returns the number of rows for the parquet file in the provided
        relative file path or S3 key.

        Parameters
        ----------
        relative_parquet_file_path : str
            A file path relative to the `output_prefix` the object was initialized with.

        Returns
        -------
        int
            The number of rows in the Parquet file.
        """
        if self.filesystem_type == "s3":
            file_bucket, file_key = s3_utils.extract_bucket_and_key(
                self.output_prefix, relative_parquet_file_path
            )
            # TODO: Sometimes we get:
            # OSError AWS Error UNKNOWN (HTTP status 503) during
            # HeadObject operation: No response body.
            # Can try catching that error and retrying with some backoff mechanism
            nrows = pq.read_metadata(
                f"{file_bucket}/{file_key}", filesystem=self.pyarrow_fs
            ).num_rows
        else:
            file_path = os.path.join(self.output_prefix, relative_parquet_file_path)
            nrows = pq.read_metadata(file_path, filesystem=self.pyarrow_fs).num_rows

        return nrows

    def get_row_counts_for_parquet_files(self, parquet_file_paths: Sequence[str]) -> Sequence[int]:
        """Returns a list of the number of rows in each parquet file in the passed-in
        list of relative S3 keys or filepaths.

        Parameters
        ----------
        parquet_file_paths : Sequence[str]
            A list of file path relative to the `output_prefix` the
            object was initialized with.

        Returns
        -------
        Sequence[int]
            A list of file counts that corresponds to the order of the file paths
            passed in.
        """
        # TODO: Despite parallel call this can still be slow for thousands of files.
        # See if we can skip or at least do fully async
        cpu_count = os.cpu_count() if os.cpu_count() else 4
        # Assertion to indicate to mypy that cpu_count is not None
        assert cpu_count
        row_counts_per_file = Parallel(n_jobs=min(16, cpu_count), backend="threading")(
            delayed(self.get_row_count_for_parquet_file)(parquet_path)
            for parquet_path in parquet_file_paths
        )

        return row_counts_per_file

    def _add_counts_for_graph_structure(
        self, top_level_key: str, edge_or_node_type_key: str
    ) -> List[Sequence[int]]:
        """Returns a nested list of counts for each structure of the graph,
        either for edges structure or node mappings. Modifies `self.metadata_dict` in place.

        Parameters
        ----------
        top_level_key : str
            The top level key that refers to the structure we'll be getting
            counts for, can be "edges" to get counts for edges structure,
            or "raw_id_mappings" to get counts for node mappings.
        edge_or_node_type_key : str
            The secondary key we use to iterate over structure types,
            can be 'edge_type' or 'node_type'.

        Returns
        -------
        List[Sequence[int]]
            A nested list of counts, the outer list is per type, and each
            inner list is a row count.
        """
        # We use the order of types in edge_type and node_type to create the counts
        assert top_level_key in {"edges", "raw_id_mappings"}, (
            "top_level_key needs to be one of 'edges', 'raw_id_mappings' " f"got {top_level_key}"
        )
        assert edge_or_node_type_key in {"edge_type", "node_type"}, (
            "edge_or_node_type_key needs to be one of 'edge_type', 'node_type' "
            f"got {edge_or_node_type_key}"
        )
        all_entries_counts = []  # type: List[Sequence[int]]
        for type_value in self.metadata_dict[edge_or_node_type_key]:
            logging.info("Getting counts for %s, %s", top_level_key, type_value)
            relative_file_list = self.metadata_dict[top_level_key][type_value]["data"]
            type_row_counts = self.get_row_counts_for_parquet_files(relative_file_list)
            self.metadata_dict[top_level_key][type_value]["row_counts"] = type_row_counts
            all_entries_counts.append(type_row_counts)

        return all_entries_counts

    def _add_counts_for_features(self, top_level_key: str, edge_or_node_type_key: str) -> None:
        """Returns a nested list of counts for each feature, either for edges features
        or node features. Modifies `self.metadata_dict` in place.

        Parameters
        ----------
        top_level_key : str
            The top level key that refers to the features we'll be getting
            counts for, can be "edge_data" to get counts for edge features,
            or "node_data" to get counts for node features.
        edge_or_node_type_key : str
            The secondary key we use to iterate over structure types,
            can be 'edge_type' or 'node_type'.
        """
        # We use the order of types in edge_type and node_type to create the counts
        assert top_level_key in {"edge_data", "node_data"}, (
            "top_level_key needs to be one of 'edge_data', 'node_data' " f"got {top_level_key}"
        )
        assert edge_or_node_type_key in {"edge_type", "node_type"}, (
            "edge_or_node_type_key needs to be one of 'edge_type', 'node_type' "
            f"got {edge_or_node_type_key}"
        )
        all_feature_counts = []  # type: List[List[Sequence[int]]]
        for type_name in self.metadata_dict[edge_or_node_type_key]:
            # We don't list features for reverse edges
            type_is_edge = edge_or_node_type_key == "edge_type"
            if type_is_edge and type_name.split(":")[1].startswith("rev"):
                continue
            features_per_type_counts = []  # type: List[Sequence[int]]
            # Only retrieve counts for types that have features
            if type_name not in self.metadata_dict[top_level_key].keys():
                logging.info(
                    "Skipping type %s, as it's not part of metadata_dict for %s: %s",
                    type_name,
                    top_level_key,
                    self.metadata_dict[top_level_key].keys(),
                )
                continue
            for feature_name, feature_data_dict in self.metadata_dict[top_level_key][
                type_name
            ].items():
                relative_file_list = feature_data_dict["data"]  # type: Sequence[str]
                logging.info(
                    "Getting counts for %s, type: %s, feature: %s",
                    top_level_key,
                    type_name,
                    feature_name,
                )
                feature_row_counts = self.get_row_counts_for_parquet_files(relative_file_list)
                logging.debug(
                    "Row counts for %s, type: %s, feature: %s, %s",
                    top_level_key,
                    type_name,
                    feature_name,
                    feature_row_counts,
                )
                feature_data_dict["row_counts"] = feature_row_counts
                features_per_type_counts.append(feature_row_counts)
            all_feature_counts.append(features_per_type_counts)

    @staticmethod
    def verify_features_and_graph_structure_match(
        data_meta: Dict[str, Dict], structure_meta: Dict[str, Dict]
    ) -> bool:
        """Verifies that the row counts of structure and feature files match.

        Distributed processing assumes that for a given edge or node type,
        the number of files for its structure and all feature files must match,
        and the row counts for each corresponding file must also match.

        This function checks if that is the case given the passed in metadata,
        and returns false if they don't.

        Parameters
        ----------
        data_meta : Dict[str, Dict]
            A metadata entry for feature files.
        structure_meta : Dict[str, Dict]
            A metadata entry for graph structure files.

        Returns
        -------
        bool
            True if the row counts match, False otherwise.

        Raises
        ------
        ValueError
            If the feature metadata contains a type that does not exist
            in the structure metadata.
        KeyError
            If 'row_counts' does not exist as a key in the feature metadata.
        """
        all_match = True
        for type_name, type_data_dict in data_meta.items():
            if type_name not in structure_meta.keys():
                raise ValueError(
                    f"type_name {type_name} not found in "
                    f"structure_meta of type {type_name}: {structure_meta.keys()}"
                )
            structure_counts = structure_meta[type_name]["row_counts"]
            for feature_name, feature_dict in type_data_dict.items():
                try:
                    feature_counts = feature_dict["row_counts"]
                except KeyError:
                    raise KeyError(
                        "key 'row_counts' not found in keys of "
                        f"feature_dict of feature {feature_name}: {feature_dict.keys()}"
                    )
                if len(feature_counts) != len(structure_counts):
                    all_match = False
                    logging.debug(
                        "Mismatch in number of feature files (%d) "
                        "and structure files (%d) for type '%s'",
                        len(feature_counts),
                        len(structure_counts),
                        type_name,
                    )
                if sum(feature_counts) != sum(structure_counts):
                    all_match = False
                    logging.debug("Mismatch in total row counts for type '%s'", type_name)
                    logging.debug(
                        "Total row counts for feature '%s': {%d}\n"
                        "do not match total row counts of structure files "
                        "for corresponding type %s: %d",
                        feature_name,
                        sum(feature_counts),
                        type_name,
                        sum(structure_counts),
                    )
                    continue
                if feature_counts != structure_counts:
                    all_match = False
                    logging.debug("Mismatch in row counts for type '%s'", type_name)
                    logging.debug(
                        "Row counts for feature '%s':\n%s\n"
                        "do not match row counts of structure files for corresponding type %s:\n%s",
                        feature_name,
                        feature_counts,
                        type_name,
                        structure_counts,
                    )
        return all_match

    @staticmethod
    def verify_all_features_match(data_meta: Dict[str, Dict]) -> bool:
        """Verifies that the row counts for all features in a particular data entry match.

        Parameters
        ----------
        data_meta : Dict[str, Dict]
            A metadata entry for feature files.

        Returns
        -------
        bool
            True if the row counts match, False otherwise.
        """
        all_match = True
        for type_name, type_data_dict in data_meta.items():
            previous_feature_counts = None
            previous_feature_name = None
            for feature_name, feature_dict in type_data_dict.items():
                feature_counts = feature_dict["row_counts"]
                if not previous_feature_counts:
                    previous_feature_counts = feature_counts
                    previous_feature_name = feature_name
                if len(feature_counts) != len(previous_feature_counts):
                    all_match = False
                    logging.debug(
                        (
                            "Mismatch in number of feature file counts for type '%s': "
                            "num files for feature %s (%d) does not match num files "
                            "for feature %s (%d)"
                        ),
                        type_name,
                        feature_name,
                        len(feature_counts),
                        previous_feature_name,
                        previous_feature_counts,
                    )
                if feature_counts != previous_feature_counts:
                    all_match = False
                    logging.debug("Mismatch in row counts for type '%s'", type_name)
                    logging.debug(
                        (
                            "Row counts for feature '%s':\n%s\n"
                            "do not match row counts for feature '%s':\n%s"
                        ),
                        feature_name,
                        feature_counts,
                        previous_feature_name,
                        previous_feature_counts,
                    )

        return all_match


def verify_metadata_match(graph_meta: Dict[str, Dict]) -> bool:
    """Verifies that the row counts for all edges and nodes match
    for the provided graph metadata.

    Parameters
    ----------
    graph_meta : Dict[str, Any]
        A graph metadata dict with Dict entries for "node_data", "edge_data",
        and "edges".

    Returns
    -------
    bool
        True if all row counts match for each type, False otherwise.
    """
    logging.info("Verifying features and structure row counts match...")
    all_edge_counts_match = ParquetRowCounter.verify_features_and_graph_structure_match(
        graph_meta["edge_data"], graph_meta["edges"]
    )
    all_node_data_counts_match = ParquetRowCounter.verify_all_features_match(
        graph_meta["node_data"]
    )
    all_edge_data_counts_match = ParquetRowCounter.verify_all_features_match(
        graph_meta["edge_data"]
    )

    all_match = True
    if (
        not all_edge_counts_match
        or not all_node_data_counts_match
        or not all_edge_data_counts_match
    ):
        all_match = False
        # TODO: Should we create a file as indication
        # downstream that repartitioning is necessary?
        logging.info(
            "Some edge/node row counts do not match, "
            "will need to re-partition before creating distributed graph."
        )

    return all_match
