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

from dataclasses import dataclass
from typing import Optional


@dataclass
class NodeConfig:
    """
    This class can be used for a dataclass to build node config for GSProcessing

    Parameters
    ----------
    node_type: str
        node type
    file_format: str
        data format, "csv" or "parquet"
    files: list of string
        list of files
    column: str
        column name
    separator: Optional[str]
    features: Optional[list[dict]]
        List of feature configuration dicts
    labels: Optional[list[dict]]
        List of label configuration dicts
    """

    node_type: str
    file_format: str
    files: list[str]
    column: str
    separator: Optional[str] = None
    features: Optional[list[dict]] = None
    labels: Optional[list[dict]] = None


@dataclass
class EdgeConfig:
    """
    This class can be used for a dataclass to build edge config for GSProcessing

    Parameters
    ----------
    source_col: str
        column name for source node
    source_type: str
        column type for source node
    dest_col: str
        column name for destination node
    dest_type: str
        column type for destination node
    file_format: str
        data format, "csv" or "parquet"
    files: list of string
        list of files
    relation: dict[str, str]
        Fully qualified relation type. Can include a plain type,
        and a column that identifies the relation sub-type.
    separator: Optional[str]
    features: Optional[list[dict]]
        List of feature configuration dicts
    labels: Optional[list[dict]]
        List of label configuration dicts
    """

    source_col: str
    source_type: str
    dest_col: str
    dest_type: str
    file_format: str
    files: list[str]
    relation: dict
    separator: Optional[str] = None
    features: Optional[list[dict]] = None
    labels: Optional[list[dict]] = None
