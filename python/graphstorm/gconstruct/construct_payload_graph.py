"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import logging


def construct_json_payload_graph(args, process_confs):
    """ Construct DGLGraph from json payload.
    """
    output_format = args.output_format
    if len(output_format) != 1 and output_format[0] != "DGL":
        logging.warning("We only support building DGLGraph for ")
    for out_format in output_format:
        assert out_format == "DGL", \
            f'Unknown output format: {format}'

    for node_conf in process_confs["graph"]["nodes"]:
        print(node_conf)

    for edge_conf in process_confs["graph"]["edges"]:
        print(edge_conf)
