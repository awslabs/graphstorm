"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass
from typing import Sequence, Optional

from graphstorm_processing.constants import SUPPORTED_FILE_TYPES


@dataclass
class DataConfig:
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
