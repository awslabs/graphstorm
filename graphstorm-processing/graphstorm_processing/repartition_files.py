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

This scripts repartitions files produced by `gs-processing`
to ensure they conform to the expectations of the DGL distributed
partitioning pipeline. It is used as an entry point script with the
`gs-repartition` command.

The script reads the generated metadata.json output of the
distributed processing pipeline, and re-partitions the
non-conforming files so that all corresponding part-* features of
the same edge/node type have the same number of rows per corresponding part-file.
The output is written to storage and a new metadata JSON file is generated.
"""

import argparse
import json
import logging
import os
import pprint
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pyarrow
from pyarrow import parquet as pq
from pyarrow import fs
import pyarrow.dataset as ds
from joblib import Parallel, delayed


from graphstorm_processing.data_transformations import s3_utils
from graphstorm_processing.graph_loaders.row_count_utils import ParquetRowCounter
from graphstorm_processing.constants import FilesystemType

NUM_WRITE_THREADS = 16


class ParquetRepartitioner:
    """Performs re-partitioning of Parquet files.

    Parameters
    ----------
    input_prefix : str
        Prefix for the input files.
    filesystem_type : FilesystemType
        The type of the filesystem being used. Should be S3 or LOCAL.
    region : Optional[str]
        Region to be used for S3 interactions, by default None.
    verify_outputs : bool, optional
        Whether to verify the correctness of the outputs created by the script, by default True.
    streaming_repartitioning: bool, optional
        When True will perform file streaming re-partitioning, holding at most 2 files
        worth of data in memory. When False (default), will load an entire feature/structure
        in memory and perform the re-partitioning using thread-parallelism.
    """

    def __init__(
        self,
        input_prefix: str,
        filesystem_type: FilesystemType,
        region: Optional[str] = None,
        verify_outputs: bool = True,
        streaming_repartitioning=False,
    ):
        # Pyarrow expects paths of the form "bucket/path/to/file", so we strip the s3:// prefix
        self.input_prefix = input_prefix[5:] if input_prefix.startswith("s3://") else input_prefix
        self.filesystem_type = filesystem_type
        if self.filesystem_type == FilesystemType.S3:
            self.bucket = self.input_prefix.split("/")[1]
            self.pyarrow_fs = fs.S3FileSystem(
                region=region, retry_strategy=fs.AwsDefaultS3RetryStrategy(max_attempts=10)
            )
        else:
            self.pyarrow_fs = fs.LocalFileSystem()
        self.verify_outputs = verify_outputs
        self.streaming_repartitioning = streaming_repartitioning
        self.verbosity = 1 if logging.getLogger().getEffectiveLevel() <= logging.INFO else 0

    def read_dataset_from_relative_path(self, relative_path: str) -> ds.Dataset:
        """
        Read a parquet dataset from storage, prepending a common prefix to the relative path.

        `relative_path` needs to not start with '/' or 's3://'.
        """
        if relative_path.startswith("s3://") or relative_path.startswith("/"):
            raise RuntimeError(
                f"Expected relative path, got path in the form {relative_path}. "
                "relative_path needs to not start with '/' or 's3://'."
            )
        if Path(relative_path).suffix == ".parquet":
            logging.debug("Deriving dataset location from filepath %s", relative_path)
            dataset_relative_path = str(Path(relative_path).parent)
        else:
            dataset_relative_path = relative_path
        dataset_location = os.path.join(self.input_prefix, dataset_relative_path)

        return ds.dataset(dataset_location, filesystem=self.pyarrow_fs, exclude_invalid_files=False)

    def read_parquet_from_relative_path(self, relative_path: str) -> pyarrow.Table:
        """
        Read a parquet file from S3, prepending the `self.input_prefix` to the object path.
        """
        file_path = os.path.join(self.input_prefix, relative_path)
        return pq.read_table(file_path, filesystem=self.pyarrow_fs, memory_map=True)

    def write_parquet_to_relative_path(
        self, relative_path: str, table: pyarrow.Table, desired_count: Optional[int] = None
    ) -> None:
        """
        Write a parquet file to storage, prepending `self.input_prefix` to the object path.
        Optionally checks if the file written has the desired number of rows.
        """
        # TODO: Might be easier to update the output file list every time
        # this is called to ensure consistency?
        file_path = os.path.join(self.input_prefix, relative_path)
        if self.filesystem_type == FilesystemType.LOCAL:
            os.makedirs(Path(file_path).parent, exist_ok=True)
        pq.write_table(table, file_path, filesystem=self.pyarrow_fs, compression="snappy")
        if self.verify_outputs:
            expected_rows = desired_count if desired_count else table.num_rows
            actual_rows = pq.read_metadata(file_path, filesystem=self.pyarrow_fs).num_rows
            assert expected_rows == actual_rows, f"Expected {expected_rows} rows, got {actual_rows}"

    @staticmethod
    def create_new_relative_path_from_existing(
        original_relative_path: str, repartitioned_file_index: int, suffix: Optional[str] = None
    ) -> str:
        """Changes the index of the filename ``part-<index>`` in `original_relative_path`
        to the one given in `repartitioned_file_index`.

        Given a path of the form 'path/to/parquet/part-00001-filename.snappy.parquet', changes the
        numerical part of the filename to match the provided `repartitioned_file_index`,
        and changes the path prefix to 'path/to/parquet-repartitioned/', or
        'path/to/parquet-repartitioned-{suffix}/' if `suffix` is provided.


        Parameters
        ----------
        original_relative_path : str
            Filepath of the form 'path/to/parquet/part-00001-filename.snappy.parquet'.
        repartitioned_file_index : int
            The new index to assign to the file
        suffix : str, optional
            Suffix to add to the returned path, by default None

        Returns
        -------
        str
            The `original_relative_path` with the part index modified, and `parquet/`
            modified to `parquet-repartitioned` or `parquet-repartitioned-{suffix}/`}

        Raises
        ------
        RuntimeError
            If the filename does not conform to the ``r"^part-[0-9]{5}"`` regex,
            which is the expected Spark filename output.

        Examples
        --------
            >>> create_new_relative_path_from_existing(
                "path/to/parquet/part-00001-filename.snappy.parquet", 3)
            "path/to/parquet-repartitioned-{uuid}/part-00003-filename.snappy.parquet"
            >>> create_new_relative_path_from_existing(
                "path/to/parquet/part-00001-filename.snappy.parquet", 3, "my-suffix")
            "path/to/parquet-repartitioned-my-suffix/part-00003-filename.snappy.parquet"
        """
        original_relative_path_obj = Path(original_relative_path)
        # We expect files to have a path of the form /path/to/parquet/part-00001.snappy.parquet
        assert original_relative_path_obj.parts[-2] == "parquet"  # TODO: Remove this assumption?

        padded_file_idx = f"part-{str(repartitioned_file_index).zfill(5)}"

        # Ensure the input path conforms to expected format
        if not re.match(r"^part-[0-9]{5}", original_relative_path_obj.parts[-1]):
            raise RuntimeError(
                "Expected file name to be of the regexp form 'part-[0-9]{5}',  got "
                + original_relative_path_obj.parts[-1]
            )

        new_file_name = re.sub(
            r"^part-[0-9]{5}", padded_file_idx, original_relative_path_obj.parts[-1]
        )

        new_sub_path = (
            "parquet-repartitioned" if suffix is None else f"parquet-repartitioned-{suffix}"
        )
        new_relative_path = "/".join(
            [*original_relative_path_obj.parts[:-2], new_sub_path, new_file_name]
        )

        return new_relative_path

    def repartition_parquet_files(self, data_entry_dict: Dict, desired_counts: list[int]) -> Dict:
        """
        Re-partitions the Parquet files in `data_entry_dict` so that their row count
        matches the one provided in `desired_counts`. We assume that the number of files between the
        input and output will remain the same.

        The output is written to storage and the `data_entry_dict` dictionary file is
        modified in-place and returned.

        Parameters
        ----------
        data_entry_dict : Dict
            A data format dictionary formatted as:
            {
                "format": {
                    "name": "parquet"
                },
                "data": [
                    "relative/path/to/file1.parquet",
                    "relative/path/to/file2.parquet",
                    ...
                ] # n files
                "row_counts": [
                    10,
                    12,
                    ...
                ] # n row counts
            }
        desired_counts : Collection[int]
            A list of desired row counts.

        Returns
        -------
        Dict
            A data format dictionary with the row
            counts updated to match desired_counts.
        """
        if self.streaming_repartitioning:
            return self._repartition_parquet_files_streaming(data_entry_dict, desired_counts)
        else:
            return self._repartition_parquet_files_in_memory(data_entry_dict, desired_counts)

    def _repartition_parquet_files_in_memory(
        self, data_entry_dict: Dict, desired_counts: list[int]
    ) -> Dict:
        """
        In-memory, thread-parallel implementation of Parquet file repartitioning.

        Notes
        -----
        This function assumes the entire dataset described in `data_entry_dict`
        can be held in memory.

        Raises
        ------
        RuntimeError
            In cases where the sum of the desired counts does not match
            the sum of actual file row counts, or the files are not in Parquet format.
        """
        if sum(desired_counts) != sum(data_entry_dict["row_counts"]):
            raise RuntimeError(
                f"Mismatch between total requested rows: {sum(desired_counts)}, "
                f"and rows in the input files: {sum(data_entry_dict['row_counts'])}"
            )

        data_format = data_entry_dict["format"]["name"]
        if data_format != "parquet":
            raise RuntimeError(
                f"We only support Parquet file row count conversion, got format: {data_format}"
            )

        datafile_list = data_entry_dict["data"]
        # Get the root location of the dataset, which is the same for all files
        dataset_location = f"{self.input_prefix}/{Path(datafile_list[0]).parent}"

        # Early exit if the problem is already solved
        if list(desired_counts) == data_entry_dict["row_counts"]:
            logging.info(
                "Desired counts match with actual for dataset under %s, "
                "no need to repartition files",
                dataset_location,
            )
            return data_entry_dict

        dataset = self.read_dataset_from_relative_path(datafile_list[0])
        table = dataset.to_table()
        if table.num_rows != sum(desired_counts):
            raise RuntimeError(
                "Actual data had a different total row count "
                f"({table.num_rows}) from expected ({sum(desired_counts)})"
            )

        logging.debug(
            "Desired counts do not match for dataset under %s, repartitioning files...",
            dataset_location,
        )
        logging.debug("Desired counts: %s", desired_counts)
        logging.debug("Row counts: %s", data_entry_dict["row_counts"])

        # From the dataset we read into memory, we slice a part according to desired_counts and
        # write a new file to S3.
        offsets = accumulate([0] + desired_counts)
        zero_copy_slices = [
            table.slice(offset=offset, length=desired_count)
            for offset, desired_count in zip(offsets, desired_counts)
        ]
        uid_for_entry = uuid.uuid4().hex[:8]
        relative_paths = [
            self.create_new_relative_path_from_existing(datafile_list[0], idx, uid_for_entry)
            for idx in range(len(desired_counts))
        ]
        with Parallel(
            n_jobs=min(NUM_WRITE_THREADS, os.cpu_count() or 16),
            verbose=self.verbosity,
            prefer="threads",
        ) as parallel:
            parallel(
                delayed(self.write_parquet_to_relative_path)(
                    relative_path,
                    slice,
                )
                for slice, relative_path in zip(zero_copy_slices, relative_paths)
            )

        data_entry_dict["data"] = relative_paths
        data_entry_dict["row_counts"] = desired_counts

        return data_entry_dict

    def _repartition_parquet_files_streaming(
        self, data_entry_dict: Dict, desired_counts: Sequence[int]
    ) -> Dict:
        """Repartition parquet files using file streaming.

        This function will maintain at most 2 files worth of data in memory.

        We iterate over the desired counts and compare against the existing file counts.
        If the row counts match, we use the original file and continue to the next iteration.
        If the existing file has fewer rows than the desired we "borrow" rows from the next
        file. The leftover rows are added to a 'remainder' table.
        If the existing file has more rows than the desired, we slice it and add the leftover
        rows to the remainder.

        The output is written to storage and the `data_entry_dict` dictionary file is
        modified in-place and returned.

        Raises
        ------
        RuntimeError
            In case we either run our of files to stream rows from,
            or if the actual rows do not match the desired while slicing.
        """
        assert len(data_entry_dict["data"]) == len(
            desired_counts
        ), "We assume input and output file counts are the same"

        data_format = data_entry_dict["format"]["name"]
        assert (
            data_format == "parquet"
        ), f"We only support parquet file row count conversion, got {data_format}"

        original_file_index = 0
        repartitioned_file_index = 0
        files_list = data_entry_dict["data"]
        remainder_table = None  # pyarrow.Table
        new_data_entries = []

        # TODO: Instead of limiting to two tables in memory, we could monitor memory and load files
        # until we run out of memory and process together to speed up the process.

        # TODO: Zip with original counts, if num rows match, no need to read the file into memory
        uid_for_entry = uuid.uuid4().hex[:8]
        for repartitioned_file_index, desired_count in enumerate(desired_counts):
            logging.debug(
                "At start of iter: repartitioned_file_index: %d, original_file_index: %d",
                repartitioned_file_index,
                original_file_index,
            )
            original_relative_path = files_list[original_file_index]
            # We assume files are originally written as
            # relative/path/to/file/parquet/part-00000.snappy.parquet
            # and rename to
            # relative/path/to/file/parquet-repartitioned/part-00000.snappy.parquet
            new_relative_path = self.create_new_relative_path_from_existing(
                original_relative_path, repartitioned_file_index, uid_for_entry
            )

            remainder_used = False
            if remainder_table:
                remainder_used = True
                logging.debug(
                    "Remainder table rows: %d, desired count: %d",
                    remainder_table.num_rows,
                    desired_count,
                )
                # Handle case where the remainder is larger than the desired count
                if remainder_table.num_rows > desired_count:
                    logging.debug(
                        "Remainder table exceeds desired count, writing to new file: %s",
                        new_relative_path,
                    )
                    sliced_table = remainder_table.slice(length=desired_count)
                    assert (
                        sliced_table.num_rows == desired_count
                    ), f"{sliced_table.num_rows} != {desired_count}"
                    self.write_parquet_to_relative_path(new_relative_path, sliced_table)
                    new_data_entries.append(new_relative_path)
                    remainder_table = remainder_table.slice(offset=desired_count)
                    continue

                # Handle case where the remainder is exactly than the desired count
                if remainder_table.num_rows == desired_count:
                    logging.debug(
                        "Remainder table has exactly desired count rows, "
                        "writing to new file: %s",
                        new_relative_path,
                    )
                    self.write_parquet_to_relative_path(new_relative_path, remainder_table)
                    new_data_entries.append(new_relative_path)
                    remainder_table = None
                    continue

                # Handle case where the remainder is smaller than the desired count
                table = self.read_parquet_from_relative_path(original_relative_path)
                logging.debug(
                    "Appending new file (%d rows) to remainder (%d) rows",
                    table.num_rows,
                    remainder_table.num_rows,
                )
                combined_table = pyarrow.concat_tables([remainder_table, table])
                # We have now used up the remainder.
                remainder_table = None
            else:
                # We avoid overwriting the original variable because
                # pyarrow.concat_tables is zero-copy, necessary?
                # TODO: Have a fast path where we only read the num_rows
                # and check against desired before reading file into memory
                combined_table = self.read_parquet_from_relative_path(original_relative_path)

            current_table_row_count = combined_table.num_rows
            logging.debug(
                "Combined row counts: %d, desired count: %d", current_table_row_count, desired_count
            )

            # If the combined row count is already the same we just write to S3 and move on
            if desired_count == current_table_row_count:
                if remainder_used:
                    logging.debug(
                        "Combined table matches expected count, writing to new file: %s",
                        new_relative_path,
                    )
                    self.write_parquet_to_relative_path(new_relative_path, combined_table)
                    new_data_entries.append(new_relative_path)
                else:
                    logging.debug("Existing table matches expected count, using existing file...")
                    new_data_entries.append(original_relative_path)
                original_file_index += 1
                continue

            # If desired_count != current_table_row_count, we need to re-partition the file
            logging.debug(
                "Re-partitioning file %s to %d rows.", original_relative_path, desired_count
            )
            logging.debug("New file: %s", new_relative_path)

            if desired_count < current_table_row_count:
                # If the desired count is less than the current file row count,
                # we need to truncate the file
                logging.debug(
                    "Combined table (%d) has more rows than "
                    "the desired count (%d), truncating...",
                    current_table_row_count,
                    desired_count,
                )
                sliced_table = combined_table.slice(length=desired_count)
                assert (
                    sliced_table.num_rows == desired_count
                ), f"{sliced_table.num_rows} != {desired_count}"
                # And we maintain the remainder in memory to append to the next file
                remainder_table = combined_table.slice(offset=desired_count)
                # Write the new file with the correct row count to S3
                self.write_parquet_to_relative_path(new_relative_path, sliced_table)
                new_data_entries.append(new_relative_path)
                # TODO: Delete the file from S3 and simply replace it with the modified file
                original_file_index += 1
            else:
                # If the desired count is greater than the current file row count,
                # we need to "borrow" rows from the next file.
                # We do this by slicing the next file and appending it to the current file
                logging.debug(
                    "Combined table has fewer rows (%d) than"
                    " the desired count %d, borrowing rows from next file...",
                    current_table_row_count,
                    desired_count,
                )
                while desired_count > current_table_row_count:
                    original_file_index += 1
                    if original_file_index >= len(files_list):
                        raise RuntimeError("We have run out of files to borrow from.")
                    next_file_relative_path = files_list[original_file_index]
                    next_table = self.read_parquet_from_relative_path(next_file_relative_path)
                    combined_table = pyarrow.concat_tables([combined_table, next_table])
                    current_table_row_count = combined_table.num_rows
                    logging.debug(
                        "Combined table row count after borrowing from next file: %d",
                        current_table_row_count,
                    )
                sliced_table = combined_table.slice(length=desired_count)
                # Sanity check in case the the combined table does not have enough rows
                if sliced_table.num_rows < desired_count:
                    raise RuntimeError(
                        f"Sliced table rows {sliced_table.num_rows} < desired {desired_count}"
                    )
                remainder_table = combined_table.slice(offset=desired_count)
                self.write_parquet_to_relative_path(new_relative_path, sliced_table)
                new_data_entries.append(new_relative_path)

            if remainder_table:
                logging.debug(
                    "Remainder table at end of iteration %d: %d",
                    repartitioned_file_index,
                    remainder_table.num_rows,
                )

        data_entry_dict["data"] = new_data_entries
        data_entry_dict["row_counts"] = desired_counts

        return data_entry_dict

    def modify_metadata_for_flat_arrays(self, file_list: List[str]) -> None:
        """Fix metadata to match DistPartitioning assumptions.

        Given a Parquet file list will iterate over each file and if the underlying
        table is a single-column-scalar table, will append the value {"shape": (num_rows,)}
        to its metadata and overwrite the file. This indicates to the downstream
        data dispatch step that the file should be read in as a flat array.

        DistPartitioning expects arrays to include a `shape` metadata parameter,
        otherwise treats all data as 2D arrays. See:
        https://github.com/dmlc/dgl/blob/9dc361c6b959e0de7af4565bf649670786ff0f36/tools/distpartitioning/array_readwriter/parquet.py#L21-L26

        Parameters
        ----------
        file_list : List[str]
            A list of relative file paths in storage (S3 or local) whose metadata we want to modify.
        """

        def modify_file(file_relative_path: str) -> None:
            """Modify metadata for single file.

            Parameters
            ----------
            file_relative_path : str
                Relative S3 or local path, we prepend `self.input_prefix`
                to read from storage and write back.
            """
            full_path = os.path.join(self.input_prefix, file_relative_path)
            file_metadata = pq.read_metadata(full_path, filesystem=self.pyarrow_fs)

            if file_metadata.num_columns == 1:
                table_schema = pq.read_schema(full_path, filesystem=self.pyarrow_fs)
                data_types = table_schema.types
                # If the column type is List, the file is a 2D array so we return
                if isinstance(data_types[0], pyarrow.ListType):
                    return
                table = self.read_parquet_from_relative_path(file_relative_path)

                # If the column type is not List, the file is a 1D array so we modify the metadata
                shape_metadata = {b"shape": f"({table.num_rows},)".encode("utf-8")}
                updated_metadata = {**shape_metadata, **(table.schema.metadata or {})}
                self.write_parquet_to_relative_path(
                    file_relative_path, table.replace_schema_metadata(updated_metadata)
                )

        # Execute modify_file in parallel, with up to NUM_WRITE_THREADS threads.
        Parallel(n_jobs=NUM_WRITE_THREADS, prefer="threads", verbose=self.verbosity)(
            delayed(modify_file)(relative_path) for relative_path in file_list
        )


def collect_frequencies_for_data_counts(
    data_meta: Dict[str, Dict[str, Dict]]
) -> Dict[str, Counter]:
    """Gather the frequency of each row count list for each feature type in the provided data dict.

    Parameters
    ----------
    data_meta : Dict[str, Dict[str, Dict]]
        A dictionary describing the data sources for multiple edge/node types.
        See example for expected schema.

    Returns
    -------
    Dict[str, Counter]
        A dict with type name as keys and Counter objects as values.
        Each Counter object has row count tuples as keys and their frequency as values.

    Raises
    ------
    KeyError
        If the 'row_counts' key is missing from one of the data entries.

    Example
    -------
    If the provided data dict is:

    {
        "type_name_1": {
            "feature_1": {
                "row_counts": [1, 2, 3],
                "data": ["path/to/file1", "path/to/file2", "path/to/file3"]
            },
            "feature_2": {
                "row_counts": [1, 2, 3],
                "data": ["path/to/file1", "path/to/file2", "path/to/file3"]
            }
        },
        "type_name_2": {
            "feature_1": {
                "row_counts": [2, 2, 2],
                "data": ["path/to/file1", "path/to/file2", "path/to/file3"]
            }
        }
    }

    Then the returned dict will be:

    {
        "type_name_1": Counter({(1, 2, 3): 2}),
        "type_name_2": Counter({(2, 2, 2): 1})
    }

    Notes
    -----
    The row counts are converted to tuples to
    enable inserting as a key to a Counter object.
    """
    row_counts_frequencies = defaultdict(Counter)  # type: Dict[str, Counter]

    for type_name, type_data_dict in data_meta.items():
        for feature_name, feature_dict in type_data_dict.items():
            try:
                # We convert to tuple to enable inserting as a key to a counter
                feature_counts = tuple(feature_dict["row_counts"])
                row_counts_frequencies[type_name][feature_counts] += 1
            except KeyError:
                raise KeyError(
                    f"key 'row_counts' not found in keys of feature_dict of "
                    f"feature {feature_name}: {feature_dict.keys()}"
                )

    return row_counts_frequencies


def verify_metadata(
    edges_graph_structure_meta: Dict,
    edge_data_meta: Optional[Dict] = None,
    node_data_meta: Optional[Dict] = None,
) -> None:
    """Verify that the produced metadata is correct.

    All edges structure files (of the same type) should have matching row
    counts will all their features.
    All node features should have matching row counts (per type).

    Parameters
    ----------
    edge_structure_meta : Dict
        The metadata describing the graph's edges structure.
    edge_data_meta : Optional[Dict], optional
        The metadata describing the graph's edges data., by default None
    node_data_meta : Optional[Dict], optional
        The metadata describing the graph's node data, by default None

    Raises
    ------
    RuntimeError
        If there is a mismatch in the counts for at least one of the
        edge types or node types.
    """
    edges_match = True
    if edges_graph_structure_meta and edge_data_meta:
        edges_match = ParquetRowCounter.verify_features_and_graph_structure_match(
            edge_data_meta, edges_graph_structure_meta
        )
        if not edges_match:
            logging.error(
                "Edge feature/structure row count metadata did "
                "not match with expected, check produced output: \nedge_structure_metadata: \n%s\n"
                "edge_data_metadata: \n%s\n",
                pprint.pformat(edges_graph_structure_meta),
                pprint.pformat(edge_data_meta),
            )
    nodes_match = True
    if node_data_meta is not None:
        nodes_match = ParquetRowCounter.verify_all_features_match(node_data_meta)
        if not nodes_match:
            logging.error(
                "Node data row count metadata did not match with expected, "
                "check produced output: %s",
                pprint.pformat(node_data_meta),
            )

    if not edges_match or not nodes_match:
        raise RuntimeError("Row count metadata did not match with expected, check produced output")


def parse_args(args):
    """
    Parse arguments for script.
    """
    parser = argparse.ArgumentParser(
        description="Re-partitions the output of the distributed processing pipeline "
        "to ensure all files that belong to the same edge/node type have the same "
        "number of rows per corresponding part-file."
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        required=True,
        help="Prefix path to where the output was generated "
        "from the distributed processing pipeline. "
        "Can be a local path (starting with '/') or S3 prefix (starting with 's3://').",
    )
    parser.add_argument(
        "--streaming-repartitioning",
        type=lambda x: (str(x).lower() in ["true", "1"]),
        default=False,
        help="When True will use low-memory file-streaming repartitioning. "
        "Note that this option is much slower than the in-memory default.",
    )
    parser.add_argument(
        "--input-metadata-file-name",
        default="metadata.json",
        type=str,
        required=False,
        help="Name of the original partitioning pipeline metadata file.",
    )
    parser.add_argument(
        "--updated-metadata-file-name",
        default="updated_row_counts_metadata.json",
        type=str,
        required=False,
        help="Name of the updated (output) partitioning pipeline metadata file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser.parse_args(args)


@dataclass
class RepartitionConfig:
    """
    Repartition configuration object.

    See graphstorm_processing.repartition_files.RepartitionConfig for details.

    Example:

    RepartitionConfig(
        input_prefix="/path/to/input",
        input_metadata_file_name="metadata.json",
        updated_metadata_file_name="updated_row_counts_metadata.json",
        streaming_repartitioning=False,
        log_level="INFO",
    )

    Notes
    -----
        - input_prefix can be a local path (starting with '/') or S3 prefix (starting with 's3://')
        - input_metadata_file_name is the name of the original partitioning pipeline metadata file
        - updated_metadata_file_name is the name of the output partitioning pipeline metadata file
        - streaming_repartitioning is a boolean flag to enable/disable file-streaming repartitioning
        - log_level is the logging level to use (e.g. "DEBUG", "INFO", "WARNING", "ERROR")
        - input_prefix is the only required field
    """

    input_prefix: str
    input_metadata_file_name: str = "metadata.json"
    updated_metadata_file_name: str = "updated_row_counts_metadata.json"
    streaming_repartitioning: bool = False
    log_level: str = "INFO"


def modify_flat_array_metadata(
    metadata_dict: dict,
    repartitioner: ParquetRepartitioner,
):
    """Modifies the Parquet metadata for any flat arrays in the data in `metadata_dict`.

    We need labels and masks to be treated as flat arrays downstream, but
    DistPartitioning treats all arrays that don't include a `shape` metadata parameter,
    as 2D arrays. So we modify the Parquet file metadata here. See:
    https://github.com/dmlc/dgl/blob/9dc361c6b959e0de7af4565bf649670786ff0f36/tools/distpartitioning/array_readwriter/parquet.py#L21-L26

    Parameters
    ----------
    metadata_dict : dict
        A chunked graph metadata dictionary, augmented with the "graph_info" dict
        that contains "task_type", "etype_label", "etype_label_property",
        "ntype_label", "ntype_label_property".
    repartitioner : ParquetRepartitioner
        The partitioner object that performs the metadata modification.
    """
    edge_data_meta: dict = metadata_dict.get("edge_data", {})
    node_data_meta: dict = metadata_dict.get("node_data", {})

    if edge_data_meta:
        task_type = metadata_dict["graph_info"].get("task_type", "link_prediction")  # type: str
        edge_types_with_labels = metadata_dict["graph_info"]["etype_label"]  # type: List[str]
        # If there exist edge types with labels
        if edge_types_with_labels:
            # And at least one edge label
            if len(metadata_dict["graph_info"]["etype_label_property"]) > 0:
                etype_label_property = metadata_dict["graph_info"]["etype_label_property"][
                    0
                ]  #  type: str
                if task_type not in {"link_predict", "link_prediction"}:
                    assert etype_label_property, (
                        "When task is not link prediction, providing an 'etype_label_property' "
                        f"is required. Got task {task_type}, but `metadata_dict['graph_info']"
                        "['etype_label_property']` was empty."
                    )

        for type_idx, (type_name, type_data_dict) in enumerate(edge_data_meta.items()):
            _, relation, _ = type_name.split(":")

            if relation.endswith("-rev"):
                # Reverse edge types do not have their own data,
                # and if needed we re-partition their structure while
                # handling the "regular" edge type.
                logging.info(
                    "Skipping modifying Parquet metadata for reverse edge type %d/%d:, '%s' "
                    "because it happens during handling regular edge type...",
                    type_idx + 1,
                    len(edge_data_meta),
                    type_name,
                )
                continue
            logging.info(
                "Modifying parquet metadata for edge type %d/%d:, '%s' ",
                type_idx + 1,
                len(edge_data_meta),
                type_name,
            )
            if type_name in edge_types_with_labels:
                # TODO: To support multi-task training, we'll have to handle train_mask_a,
                # train_mask_b, etc. and similarly for multiple label columns.

                # If the task is not link_prediction, we need to modify the label file's metadata
                if task_type not in {"link_predict", "link_prediction"}:
                    assert etype_label_property in type_data_dict, (
                        "When task is not link prediction, providing an 'etype_label_property' "
                        f"is required. Got task {task_type}, but {etype_label_property=}"
                        f"was not in {type_data_dict=}"
                    )
                    label_files = type_data_dict[etype_label_property]["data"]  # type: List[str]
                    logging.info(
                        "Modifying Parquet metadata for %d files of label '%s' of edge type '%s'",
                        len(label_files),
                        etype_label_property,
                        type_name,
                    )
                    repartitioner.modify_metadata_for_flat_arrays(label_files)
                for mask in ["train_mask", "test_mask", "val_mask"]:
                    if mask in type_data_dict:
                        edge_mask_files = type_data_dict[mask]["data"]  # type: List[str]
                        logging.info(
                            "Modifying Parquet metadata for %d files of '%s' for edge type '%s'",
                            len(edge_mask_files),
                            mask,
                            type_name,
                        )
                        repartitioner.modify_metadata_for_flat_arrays(edge_mask_files)

    if node_data_meta:
        node_types_with_labels = metadata_dict["graph_info"]["ntype_label"]  # type: List[str]
        if node_types_with_labels:
            ntype_label_property = metadata_dict["graph_info"]["ntype_label_property"][
                0
            ]  #  type: str
        for type_idx, (type_name, type_data_dict) in enumerate(node_data_meta.items()):
            logging.info(
                "Modifying Parquet metadata for node type '%s', %d/%d:",
                type_name,
                type_idx + 1,
                len(node_data_meta),
            )
            if type_name in node_types_with_labels:
                node_label_files = type_data_dict[ntype_label_property]["data"]  # type: List[str]
                logging.info(
                    "Modifying Parquet metadata for %d files of label '%s' of node type '%s'",
                    len(node_label_files),
                    ntype_label_property,
                    type_name,
                )
                repartitioner.modify_metadata_for_flat_arrays(node_label_files)
                for mask in ["train_mask", "test_mask", "val_mask"]:
                    if mask in type_data_dict:
                        node_mask_files = type_data_dict[mask]["data"]  # type: List[str]
                        logging.info(
                            "Modifying Parquet metadata for %d files of '%s' for node type '%s'",
                            len(node_mask_files),
                            mask,
                            type_name,
                        )
                        repartitioner.modify_metadata_for_flat_arrays(node_mask_files)


def _repartition_edge_files(metadata_dict: dict[str, Any], repartitioner: ParquetRepartitioner):
    edge_structure_meta = metadata_dict["edges"]  # type: Dict[str, Dict[str, Dict]]
    # We first collect the most frequent row counts, to minimize the number of
    # repartitions we have to perform.
    edge_data_meta = metadata_dict["edge_data"]
    assert edge_data_meta
    # TODO: Frequencies are important, but so is data size,
    # we would like to avoid downloading large feature files if possible
    edge_row_counts_frequencies = collect_frequencies_for_data_counts(edge_data_meta)

    # Re-partition the edge files based on the most frequent row counts for each edge type
    for type_idx, (type_name, type_data_dict) in enumerate(edge_data_meta.items()):
        src, relation, dst = type_name.split(":")

        if relation.endswith("-rev"):
            # Reverse edge types do not have their own data,
            # and if needed we re-partition their structure while
            # handling the "regular" edge type.
            logging.info(
                "Skipping repartitioning for reverse edge type %d/%d:, '%s' "
                "because it happens during handling regular edge type...",
                type_idx + 1,
                len(edge_data_meta),
                type_name,
            )
            continue

        reverse_edge_type_name = f"{dst}:{relation}-rev:{src}"
        most_frequent_counts = list(edge_row_counts_frequencies[type_name].most_common(1)[0][0])

        structure_counts = edge_structure_meta[type_name]["row_counts"]

        # Repartition edge structure files if the row counts don't match the most frequent
        if structure_counts != most_frequent_counts:
            logging.info(
                "Repartitioning %d structure files for edge type %d/%d: '%s'",
                len(edge_structure_meta[type_name]["data"]),
                type_idx + 1,
                len(edge_data_meta),
                type_name,
            )

            edge_structure_meta[type_name] = repartitioner.repartition_parquet_files(
                edge_structure_meta[type_name], most_frequent_counts
            )
        else:
            logging.info(
                "Structure files for edge type %d/%d: '%s' match, skipping repartitioning",
                type_idx + 1,
                len(edge_data_meta),
                type_name,
            )

        # If the reverse structure counts don't match we'll need to re-partition
        # them because the feature files are shared between regular and reverse
        if reverse_edge_type_name in edge_structure_meta:
            if edge_structure_meta[reverse_edge_type_name]["row_counts"] != most_frequent_counts:
                logging.info(
                    "Repartitioning %d structure files for reverse edge type '%s'",
                    len(edge_structure_meta[reverse_edge_type_name]["data"]),
                    reverse_edge_type_name,
                )
                edge_structure_meta[reverse_edge_type_name] = (
                    repartitioner.repartition_parquet_files(
                        edge_structure_meta[reverse_edge_type_name], most_frequent_counts
                    )
                )

        # Repartition edge feature files if the row counts don't match the most frequent
        for feature_idx, (feature_name, feature_dict) in enumerate(type_data_dict.items()):
            if feature_dict["row_counts"] != most_frequent_counts:
                logging.info(
                    "Repartitioning %d feature files for edge type '%s', feature %d/%d '%s'",
                    len(feature_dict["data"]),
                    type_name,
                    feature_idx + 1,
                    len(type_data_dict),
                    feature_name,
                )
                feature_dict = repartitioner.repartition_parquet_files(
                    feature_dict, most_frequent_counts
                )
                if (
                    reverse_edge_type_name in edge_data_meta
                    and feature_name in edge_data_meta[reverse_edge_type_name]
                ):
                    logging.info(
                        "Assigning re-partitioned feature files for '%s' to reverse "
                        "edge type '%s'",
                        feature_name,
                        reverse_edge_type_name,
                    )
                    edge_data_meta[reverse_edge_type_name][feature_name] = feature_dict
                else:
                    logging.debug(
                        "Did not find reverse of edge type '%s', %s in %s",
                        type_name,
                        reverse_edge_type_name,
                        edge_data_meta.keys(),
                    )
            else:
                logging.info(
                    (
                        "Feature '%s' (%d/%d) of edge type '%s', already has correct row counts, "
                        "skipping repartitioning."
                    ),
                    feature_name,
                    feature_idx + 1,
                    len(type_data_dict),
                    type_name,
                )


def _repartition_node_files(metadata_dict: dict[str, Any], repartitioner: ParquetRepartitioner):
    node_data_meta = metadata_dict["node_data"]
    assert node_data_meta
    node_row_counts_frequencies = collect_frequencies_for_data_counts(node_data_meta)

    logging.info("Repartitioning node feature files")
    for type_idx, (type_name, type_data_dict) in enumerate(node_data_meta.items()):
        logging.info(
            "Repartitioning feature files for node type '%s', (%d/%d)",
            type_name,
            type_idx + 1,
            len(node_data_meta),
        )
        most_frequent_counts = list(node_row_counts_frequencies[type_name].most_common(1)[0][0])

        for feature_idx, (feature_name, feature_dict) in enumerate(type_data_dict.items()):
            if feature_dict["row_counts"] != most_frequent_counts:
                logging.info(
                    "Repartitioning %d feature files for node type '%s', feature '%s' (%d/%d)",
                    len(feature_dict["data"]),
                    type_name,
                    feature_name,
                    feature_idx + 1,
                    len(type_data_dict),
                )
                repartitioner.repartition_parquet_files(feature_dict, most_frequent_counts)
            else:
                logging.info(
                    (
                        "Feature '%s' (%d/%d) of node type '%s', already has correct row counts, "
                        "skipping repartitioning."
                    ),
                    feature_name,
                    feature_idx + 1,
                    len(type_data_dict),
                    type_name,
                )


def repartition_files(metadata_dict: Dict[str, Any], repartitioner: ParquetRepartitioner):
    """Applies repartition of node and edge data files and modifies flat array Parquet metadata.

    The `metadata_dict` argument is modified in-place.

    Returns
    -------
    dict
        The provided `metadata_dict`, modified in-place.
    """
    edge_data_exist = "edge_data" in metadata_dict.keys() and metadata_dict["edge_data"]
    node_data_exist = "node_data" in metadata_dict.keys() and metadata_dict["node_data"]

    if not edge_data_exist and not node_data_exist:
        logging.info("No edge or node data found in metadata, skipping repartitioning")
        return metadata_dict

    edge_data_meta: Optional[Dict[str, Dict[str, Dict]]] = None
    node_data_meta: Optional[Dict[str, Dict[str, Dict]]] = None

    # Repartition edge structure and feature files
    if edge_data_exist:
        _repartition_edge_files(metadata_dict, repartitioner)

    # Re-partition the node feature data files
    if node_data_exist:
        _repartition_node_files(metadata_dict, repartitioner)

    # Modify label and mask 1D arrays metadata so they will be read in as flat arrays
    # by dgl/tools/dispatch_data.py
    modify_flat_array_metadata(metadata_dict, repartitioner)

    # Ensure output dict has the correct order of keys, as expected by dispatch_data.py
    if edge_data_exist:
        for edge_type in metadata_dict["edge_type"]:
            metadata_dict["edges"][edge_type] = metadata_dict["edges"].pop(edge_type)
            if edge_type in metadata_dict["edge_data"]:
                metadata_dict["edge_data"][edge_type] = metadata_dict["edge_data"].pop(edge_type)
    if node_data_exist:
        for node_type in metadata_dict["node_type"]:
            if node_type in metadata_dict["node_data"].keys():
                metadata_dict["node_data"][node_type] = metadata_dict["node_data"].pop(node_type)

    edge_structure_meta = metadata_dict["edges"]
    verify_metadata(edge_structure_meta, edge_data_meta, node_data_meta)

    return metadata_dict


def main():
    """
    Re-partitions the output of the distributed processing pipeline
    to ensure all files that belong to the same edge/node type have the same
    number of rows per corresponding part-file.

    This is done by reading the metadata file and partitioning the files
    according to the row counts of the features in the metadata file.

    The output is written under the same prefix as the input, with the
    updated metadata file name.

    Parameters
    ----------
    input_prefix : str
        Prefix path to where the output was generated
        from the distributed processing pipeline.
        Can be a local path or S3 prefix (starting with 's3://').
    streaming_repartitioning: bool
        When True will use low-memory file-streaming repartitioning.
        Note that this option is much slower than the in-memory default.
    input_metadata_file_name : str
        Name of the original partitioning pipeline metadata file.
    updated_metadata_file_name : str
        Name of the updated (output) partitioning pipeline metadata file.
    log_level : str
        Log level.

    Notes
    -----
    This function is meant to be used as a command-line tool.
    """
    repartition_config = RepartitionConfig(**vars(parse_args(sys.argv[1:])))
    logging.basicConfig(level=getattr(logging, repartition_config.log_level.upper(), None))

    if repartition_config.input_prefix.startswith("s3://"):
        filesystem_type = FilesystemType.S3
    else:
        input_prefix = str(Path(repartition_config.input_prefix).resolve(strict=True))
        filesystem_type = FilesystemType.LOCAL

    # Trim trailing '/' from S3 URI
    if filesystem_type == FilesystemType.S3:
        input_prefix = s3_utils.s3_path_remove_trailing(repartition_config.input_prefix)

    logging.info(
        "Re-partitioning files under %s to ensure all files that belong to the same "
        "edge/node type have the same number of rows per part-file.",
        input_prefix,
    )

    t0 = time.time()

    logging.info(
        "Reading metadata from %s/%s", input_prefix, repartition_config.input_metadata_file_name
    )

    # Get the metadata file
    region = None
    if filesystem_type == FilesystemType.S3:
        bucket = input_prefix.split("/")[2]
        s3_key_prefix = input_prefix.split("/", 3)[3]
        region = s3_utils.get_bucket_region(bucket)
        s3_client = boto3.client("s3", region_name=region)
        s3_client.download_file(
            bucket,
            f"{s3_key_prefix}/{repartition_config.input_metadata_file_name}",
            f"/tmp/{repartition_config.input_metadata_file_name}",
        )
        metadata_filepath = os.path.join("/tmp/", repartition_config.input_metadata_file_name)
    else:
        metadata_filepath = os.path.join(input_prefix, repartition_config.input_metadata_file_name)

    metadata_filepath = str(Path(metadata_filepath).resolve(strict=True))

    with open(metadata_filepath, "r", encoding="utf-8") as metafile:
        metadata_dict = json.load(metafile)  # type: Dict[str, Dict]

    repartitioner = ParquetRepartitioner(
        input_prefix,
        filesystem_type,
        region,
        verify_outputs=True,
        streaming_repartitioning=repartition_config.streaming_repartitioning,
    )

    metadata_dict = repartition_files(metadata_dict, repartitioner)

    with tempfile.NamedTemporaryFile(mode="w") as metafile:
        json.dump(metadata_dict, metafile, indent=4)
        metafile.flush()

        # Upload the updated metadata file to S3
        if filesystem_type == FilesystemType.S3:
            s3_client.upload_file(
                metafile.name,
                bucket,
                f"{s3_key_prefix}/{repartition_config.updated_metadata_file_name}",
            )
            logging.info(
                "Uploaded updated metadata file to s3://%s/%s/%s",
                bucket,
                s3_key_prefix,
                repartition_config.updated_metadata_file_name,
            )
        else:
            shutil.copyfile(
                metafile.name,
                os.path.join(input_prefix, repartition_config.updated_metadata_file_name),
            )

    t1 = time.time()
    logging.info("File repartitioning time: %s", t1 - t0)


if __name__ == "__main__":
    main()
