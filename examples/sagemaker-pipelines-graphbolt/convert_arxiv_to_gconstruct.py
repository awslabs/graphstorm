"""
Copyright Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Download ogbn-arxiv data and prepare for input to GConstruct
"""

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs
from ogb.nodeproppred import NodePropPredDataset


def parse_args() -> argparse.Namespace:
    """Get the output prefix argument for the scrip"""
    parser = argparse.ArgumentParser(
        description="Convert OGB arxiv data to gconstruct format and write to S3"
    )
    parser.add_argument(
        "--output-s3-prefix",
        type=str,
        required=True,
        help="S3 prefix for the output directory for gconstruct format",
    )
    return parser.parse_args()


def get_filesystem(path):
    """Choose the appropriate filesystem based on the path"""
    return fs.S3FileSystem() if path.startswith("s3://") else fs.LocalFileSystem()


def convert_ogbn_arxiv(output_prefix: str):
    """Convert ogbn-arxiv data to GConstruct and output to output_prefix"""
    pyarrow_fs = get_filesystem(output_prefix)

    if output_prefix.startswith("s3://"):
        output_prefix = output_prefix[5:]

    # Load the entire dataset
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()

    # Convert node features and labels
    node_feat = graph["node_feat"]
    table = pa.Table.from_arrays(
        [
            pa.array(range(len(node_feat))),
            pa.array(list(node_feat)),
            pa.array(labels.squeeze()),
            pa.array(graph["node_year"].squeeze()),
        ],
        names=["nid", "feat", "labels", "year"],
    )
    pq.write_table(
        table, f"{output_prefix}/nodes/paper/nodes.parquet", filesystem=pyarrow_fs
    )

    # Convert edge index
    edge_index = graph["edge_index"]
    edge_table = pa.Table.from_arrays(
        [pa.array(edge_index[0]), pa.array(edge_index[1])], names=["src", "dst"]
    )
    pq.write_table(
        edge_table,
        f"{output_prefix}/edges/paper-cites-paper/edges.parquet",
        filesystem=pyarrow_fs,
    )

    # Convert train/val/test splits
    assert split_idx, "Split index must exist for ogbn-arxiv"
    for split in ["train", "valid", "test"]:
        split_indices = split_idx[split]
        split_table = pa.Table.from_arrays([pa.array(split_indices)], names=["nid"])
        pq.write_table(
            split_table,
            f"{output_prefix}/splits/{split}_idx.parquet",
            filesystem=pyarrow_fs,
        )

    config = {
        "version": "gconstruct-v0.1",
        "nodes": [
            {
                "node_id_col": "nid",
                "node_type": "node",
                "format": {"name": "parquet"},
                "files": ["nodes/paper/nodes.parquet"],
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "paper_feat",
                    },
                    {
                        "feature_col": "year",
                        "feature_name": "paper_year",
                        "transform": {"name": "max_min_norm"},
                    },
                ],
                "labels": [
                    {
                        "label_col": "labels",
                        "task_type": "classification",
                        "custom_split_filenames": {
                            "column": "nid",
                            "train": "splits/train_idx.parquet",
                            "valid": "splits/valid_idx.parquet",
                            "test": "splits/test_idx.parquet",
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
                "relation": ["node", "cites", "node"],
                "format": {"name": "parquet"},
                "files": ["edges/paper-cites-paper/edges.parquet"],
            },
            {
                "source_id_col": "dst",
                "dest_id_col": "src",
                "relation": ["node", "cites-rev", "node"],
                "format": {"name": "parquet"},
                "files": ["edges/paper-cites-paper/edges.parquet"],
            },
        ],
    }

    # Write config to output
    with pyarrow_fs.open_output_stream(
        f"{output_prefix}/gconstruct_config_arxiv.json"
    ) as f:
        f.write(json.dumps(config, indent=2).encode("utf-8"))

    print(
        "Conversion for ogbn-arxiv completed. "
        f"Output files and configuration are in {output_prefix}"
    )


if __name__ == "__main__":
    args = parse_args()

    convert_ogbn_arxiv(args.output_s3_prefix)
