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
from typing import Sequence, Optional

from graphstorm_processing.constants import SUPPORTED_FILE_TYPES


@dataclass
class DataStorageConfig:
    """
    Data configuration class, used to store information about a data source.
    """

    format: str
    files: Sequence[str]
    separator: Optional[str] = None

    def __post_init__(self):
        assert self.format in SUPPORTED_FILE_TYPES, f"Unsupported file format {self.format}"
        if self.format == "csv":
            assert self.separator, "CSV format requires separator"
            for file in self.files:
                if file.startswith("/"):
                    raise ValueError(
                        f"File paths need to be relative (not starting with '/'), got : {file}"
                    )
