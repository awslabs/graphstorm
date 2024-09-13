"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Convert existing partitioned data to GraphBolt format.
"""

import argparse
import importlib.metadata
import logging
import time

from dgl import distributed as dgl_distributed
from packaging import version


def parse_gbconv_args() -> argparse.Namespace:
    """Parses GraphBolt conversion arguments"""
    parser = argparse.ArgumentParser("Convert partitioned DGL graph to GraphBolt format.")
    parser.add_argument("--metadata-filepath", type=str, required=True,
                           help="File path to the partitioned DGL metadata file.")
    parser.add_argument("--logging-level", type=str, default="info",
                           help="The logging level. The possible values: debug, info, warning, \
                                   error. The default value is info.")

    return parser.parse_args()


def main():
    """ Entry point
    """
    dgl_version = importlib.metadata.version('dgl')
    if version.parse(dgl_version) < version.parse("2.1.0"):
        raise ValueError(
                "GraphBolt conversion requires DGL version >= 2.1.0, "
                f"but DGL version was {dgl_version}. "
        )

    gb_conv_args = parse_gbconv_args()
    # set logging level
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, gb_conv_args.logging_level.upper()),
    )

    part_config = gb_conv_args.metadata_filepath

    gb_start = time.time()
    logging.info("Converting partitions to GraphBolt format")

    dgl_distributed.dgl_partition_to_graphbolt(
            part_config,
            store_eids=True,
            graph_formats="coo",
        )

    logging.info("GraphBolt conversion took %f sec.",
                    time.time() - gb_start)

if __name__ == '__main__':
    main()
