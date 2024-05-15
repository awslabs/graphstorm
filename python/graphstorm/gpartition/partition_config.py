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

    Parmetis partition configuration
"""
from dataclasses import dataclass

@dataclass
class ParMETISConfig:
    """
    Dataclass for holding the configuration for a ParMETIS partitioning algorithm.

    Parameters
    ----------
    ip_list: str
        ip list
    input_path: str
        Path to the input graph data
    dgl_tool_path: str
        Path to the dgl tool added in the PYTHONPATH
    metadata_filename: str
        schema file name defined in the parmetis step
    """
    ip_list: str
    input_path: str
    dgl_tool_path: str
    metadata_filename: str
