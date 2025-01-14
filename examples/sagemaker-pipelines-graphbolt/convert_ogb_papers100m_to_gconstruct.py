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

Convert papers100M data and prepare for input to GConstruct
"""

import argparse
import gzip
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs
from tqdm_loggable.auto import tqdm

# pylint: disable=logging-fstring-interpolation


def parse_args():
    """Parse conversion arguments."""
    parser = argparse.ArgumentParser(
        description="Convert raw OGB papers-100M data to GConstruct format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing OGB papers-100M data",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Prefix path to the output directory for gconstruct format. Can be local or s3://",
    )
    return parser.parse_args()


def get_filesystem(path):
    """Choose the appropriate filesystem based on the path"""
    return fs.S3FileSystem() if path.startswith("s3://") else fs.LocalFileSystem()


def process_and_upload_chunk(
    data, schema, output_dir, filesystem, entity_type, start, end
):
    """Worker function that writes the input data as a parquet file"""
    table = pa.Table.from_arrays(data, schema=schema)
    ds.write_dataset(
        table,
        base_dir=f"{output_dir}/{entity_type}",
        basename_template=f"{entity_type}-{start:012}-{end:012}-{{i}}.parquet",
        format="parquet",
        schema=schema,
        filesystem=filesystem,
        file_options=ds.ParquetFileFormat().make_write_options(compression="snappy"),
        max_rows_per_file=end - start,
        existing_data_behavior="overwrite_or_ignore",
    )


def process_data(input_dir, output_dir, filesystem):
    """Process papers100M data using threads"""
    # Load data using memory mapping to minimize memory usage
    node_feat = np.load(input_dir / "raw" / "node_feat.npy", mmap_mode="r")
    node_year = np.load(input_dir / "raw" / "node_year.npy", mmap_mode="r")
    edge_index = np.load(input_dir / "raw" / "edge_index.npy", mmap_mode="r")
    labels = np.load(input_dir / "raw" / "node-label.npz", mmap_mode="r")["node_label"]

    num_nodes, num_features = node_feat.shape
    num_edges = edge_index.shape[1]
    logging.info(
        f"Node features shape: {node_feat.shape}, Number of edges: {num_edges:,}"
    )

    # Define schemas for nodes and edges
    node_schema = pa.schema(
        [
            ("nid", pa.int64()),
            ("feat", pa.large_list(pa.float32())),
            ("label", pa.float32()),
            ("year", pa.int16()),
        ]
    )
    edge_schema = pa.schema([("src", pa.int64()), ("dst", pa.int64())])

    # Calculate chunk sizes and number of workers based on available memory
    available_ram = psutil.virtual_memory().available

    # Calculate memory usage per node row
    node_row_bytes = (
        num_features * 4 + 8 + 2
    )  # 4 bytes per float32, 8 bytes for int64 nid, 2 bytes for int16 year
    # Set node chunk size to fit within 1GB or the total number of nodes, whichever is smaller
    node_chunk_size = min((1024**3) // node_row_bytes, num_nodes)

    # Calculate memory usage per edge row
    edge_row_bytes = 16  # 8 bytes for each int64 (src and dst)
    # Set edge chunk size to fit within 1GB or the total number of edges, whichever is smaller
    edge_chunk_size = min((1024**3) // edge_row_bytes, num_edges)

    # Set the number of worker threads
    # Use 2 times the number of CPU cores (or 4 if CPU count can't be determined)
    # But limit based on available RAM, assuming each worker might use up to 2GB
    max_workers = min(16, available_ram // (2 * 1024**3))

    logging.info(
        f"Node chunk size: {node_chunk_size:,} rows, Edge chunk size: {edge_chunk_size:,} rows."
    )
    logging.info(f"Max concurrent workers: {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        num_node_chunks = num_nodes // node_chunk_size
        # Process and upload nodes in chunks
        for idx, start in enumerate(
            tqdm(range(0, num_nodes, node_chunk_size)), start=1
        ):
            end = min(start + node_chunk_size, num_nodes)
            logging.info(
                f"Reading data chunk {idx}/{num_node_chunks} for nodes {start:,}-{end:,}"
            )
            data = [
                pa.array(range(start, end)),
                pa.array(list(node_feat[start:end])),
                pa.array(labels[start:end].squeeze()),
                pa.array(node_year[start:end].astype(np.int16).squeeze()),
            ]
            logging.info(f"Submitting job {idx} for nodes {start:,}-{end:,}")
            futures.append(
                executor.submit(
                    process_and_upload_chunk,
                    data,
                    node_schema,
                    output_dir,
                    filesystem,
                    "nodes",
                    start,
                    end,
                )
            )

        # Process and upload edges in chunks
        num_edge_chunks = num_edges // edge_chunk_size
        for idx, start in enumerate(
            tqdm(range(0, num_edges, edge_chunk_size)), start=1
        ):
            end = min(start + edge_chunk_size, num_edges)
            logging.info(
                f"Reading data chunk {idx}/{num_edge_chunks} for edges {start:,}-{end:,}"
            )
            data = [
                pa.array(edge_index[0, start:end]),
                pa.array(edge_index[1, start:end]),
            ]
            logging.info(f"Submitting job {idx} for edges {start:,}-{end:,}")
            futures.append(
                executor.submit(
                    process_and_upload_chunk,
                    data,
                    edge_schema,
                    output_dir,
                    filesystem,
                    "edges",
                    start,
                    end,
                )
            )

        # Wait for all uploads to complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing and uploading"
        ):
            # This will raise any exceptions that occurred during processing or upload
            future.result()

    # Process split files
    split_files = {}
    for split in ["train", "valid", "test"]:
        with gzip.open(input_dir / "split" / "time" / f"{split}.csv.gz", "rt") as f:
            split_indices = [int(line.strip()) for line in f]
        split_table = pa.table({"nid": split_indices})
        pq.write_table(
            split_table, f"{output_dir}/{split}_idx.parquet", filesystem=filesystem
        )
        split_files[split] = f"{split}_idx.parquet"

    return split_files


def create_config(output_dir, filesystem, split_files):
    """Create the GConstruct configuration file and write to output_dir"""
    config = {
        "version": "gconstruct-v0.1",
        "nodes": [
            {
                "node_id_col": "nid",
                "node_type": "paper",
                "format": {"name": "parquet"},
                "files": ["nodes"],
                "features": [
                    {"feature_col": "feat", "feature_name": "paper_feat"},
                    {
                        "feature_col": "year",
                        "feature_name": "paper_year",
                        "transform": {"name": "max_min_norm"},
                    },
                ],
                "labels": [
                    {
                        "label_col": "label",
                        "task_type": "classification",
                        "custom_split_filenames": {
                            "column": "nid",
                            "train": split_files["train"],
                            "valid": split_files["valid"],
                            "test": split_files["test"],
                        },
                        "label_stats_type": "frequency_cnt",
                    }
                ],
            }
        ],
        "edges": [
            {
                "source_id_col": "src",
                "dest_id_col": "dst",
                "relation": ["paper", "cites", "paper"],
                "format": {"name": "parquet"},
                "files": ["edges"],
            },
            {
                "source_id_col": "dst",
                "dest_id_col": "src",
                "relation": ["paper", "cites-rev", "paper"],
                "format": {"name": "parquet"},
                "files": ["edges"],
            },
        ],
    }

    # Write the configuration to a JSON file
    with filesystem.open_output_stream(
        f"{output_dir}/gconstruct_config_papers100m.json"
    ) as f:
        f.write(json.dumps(config, indent=2).encode("utf-8"))


def main():
    """Runs the conversion from raw data to GConstruct input format"""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    input_path = Path(args.input_dir)
    filesystem = get_filesystem(args.output_prefix)

    # Adjust the output prefix for S3 if necessary
    if args.output_prefix.startswith("s3://"):
        # PyArrow expects 'bucket/key...' for S3
        output_prefix = args.output_prefix[5:]
    else:
        output_prefix = args.output_prefix

    # Remove trailing slash from output prefix
    output_prefix = output_prefix[:-1] if output_prefix.endswith("/") else output_prefix

    # Create output directories
    for path in ["nodes", "edges"]:
        filesystem.create_dir(f"{output_prefix}/{path}", recursive=True)

    # Process the data and get split files information
    split_files = process_data(input_path, output_prefix, filesystem)

    # Create and write the configuration file
    create_config(output_prefix, filesystem, split_files)

    print(
        "Conversion for ogbn-papers100M completed. "
        f"Output files and GConstruct configuration are in {output_prefix}"
    )


if __name__ == "__main__":
    main()
