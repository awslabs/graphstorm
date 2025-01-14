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

import abc
import logging
from typing import Any, Dict, Optional


class LabelConfig(abc.ABC):
    """Basic class for label config"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

        if "column" in config_dict:
            self._label_column = config_dict["column"]
        else:
            self._label_column = ""
        self._task_type: str = config_dict["type"]
        self._separator: Optional[str] = (
            config_dict["separator"] if "separator" in config_dict else None
        )
        self._multilabel = self._separator is not None
        self._custom_split_filenames: Dict[str, list[str]] = {}
        self._split: Dict[str, float] = {}
        if "custom_split_filenames" not in config_dict:
            # Get the custom split rate, or use the default 80/10/10 split
            self._split = config_dict.get(
                "split_rate",
                {"train": 0.8, "val": 0.1, "test": 0.1},
            )
        else:
            if "split_rate" in config_dict:
                logging.warning(
                    "custom_split_filenames and split_rate are set at the same time, "
                    "will do custom data split"
                )
            self._custom_split_filenames = config_dict["custom_split_filenames"]
        if "mask_field_names" in config_dict:
            self._mask_field_names: Optional[list[str]] = config_dict["mask_field_names"]
        else:
            self._mask_field_names = None

    def _sanity_check(self):
        if self._label_column == "":
            assert self._task_type == "link_prediction", (
                "When no label column is specified, the task type must be link_prediction, "
                f"got {self._task_type}"
            )
        # Sanity checks for random and custom splits
        if "custom_split_filenames" not in self._config:
            assert isinstance(self._task_type, str)
            assert isinstance(self._split, dict)
        else:
            assert isinstance(self._custom_split_filenames, dict)

            # Ensure at least one of ["train", "valid", "test"] is in the keys
            assert any(x in self._custom_split_filenames.keys() for x in ["train", "valid", "test"])

            for entry in ["train", "valid", "test", "column"]:
                # All existing entries must be lists of strings
                if entry in self._custom_split_filenames:
                    assert isinstance(self._custom_split_filenames[entry], list)
                    assert all(isinstance(x, str) for x in self._custom_split_filenames[entry])

        assert isinstance(self._separator, str) if self._multilabel else self._separator is None

        if self._mask_field_names:
            assert isinstance(self._mask_field_names, list)
            assert all(isinstance(x, str) for x in self._mask_field_names)
            assert len(self._mask_field_names) == 3

    @property
    def label_column(self) -> str:
        """The name of the column storing the target label property value."""
        return self._label_column

    @property
    def task_type(self) -> str:
        """The task type of this label configuration."""
        return self._task_type

    @property
    def split_rate(self) -> Dict[str, float]:
        """The split rate of this label configuration."""
        return self._split

    @property
    def separator(self) -> Optional[str]:
        """The separator of this label configuration, if the task
        is multilabel classification.
        """
        return self._separator

    @property
    def multilabel(self) -> bool:
        """Whether the task is multilabel classification."""
        return self._multilabel

    @property
    def custom_split_filenames(self) -> Dict[str, Any]:
        """The config for custom split labels."""
        return self._custom_split_filenames

    @property
    def mask_field_names(self) -> Optional[tuple[str, str, str]]:
        """Custom names to assign to masks for multi-task learning."""
        if self._mask_field_names is None:
            return None
        else:
            assert len(self._mask_field_names) == 3
            return (self._mask_field_names[0], self._mask_field_names[1], self._mask_field_names[2])


class EdgeLabelConfig(LabelConfig):
    """Holds the configuration of an edge label.

    Parameters:
    ------------
    config: dict
        Input config
    """

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self._sanity_check()

    def _sanity_check(self):
        super()._sanity_check()
        assert self._task_type in ["classification", "regression", "link_prediction"], (
            "Currently GraphStorm only supports three task types: "
            "classification, regression and link_prediction, but given {}".format(self._task_type)
        )


class NodeLabelConfig(LabelConfig):
    """Holds the configuration of a node label.

    Parameters:
    ------------
    config: dict
        Input config
    """

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self._sanity_check()

    def _sanity_check(self):
        super()._sanity_check()
        assert self._task_type in ["classification", "regression"], (
            "Currently GraphStorm only supports three task types: "
            "classification, regression, but given {}".format(self._task_type)
        )
