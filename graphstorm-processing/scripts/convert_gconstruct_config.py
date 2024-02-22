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

Usage:
    python convert_gconstruct_config.py --input <path-to-input-file> \
        --output <path-to-output-file> --input-type gconstruct

Entry point for graph spec conversion. Allows us to convert a graph
data specification from GConstruct to the format used by GSProcessing.
"""

import argparse
import json

from graphstorm_processing.config.config_conversion import GConstructConfigConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input", help="input GConstruct config file", type=str, required=True)
    parser.add_argument(
        "--output",
        help="output GSProcessing config.",
        type=str,
        default="gsprocessing_converted_config.json",
    )
    parser.add_argument(
        "--input-type",
        help="The type of the configuration file",
        type=str,
        default="gconstruct",
        choices=["gconstruct"],
    )
    args = parser.parse_args()

    # read input json files
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # select mode first
    if args.input_type == "gconstruct":
        gc_converter = GConstructConfigConverter()
        data_convert = gc_converter.convert_to_gsprocessing(data)
    else:
        raise SystemError(f"Unexpected type of input configuration file: {args.input_type}")

    # Serializing json and write output
    with open(args.output, "w", encoding="utf-8") as outfile:
        json.dump(data_convert, outfile, indent=4)
