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
from typing import Any, Dict, List, Optional, Sequence


class LabelConfig(abc.ABC):
    """Basic class for label config"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

        if "column" in config_dict:
            cols = config_dict["column"]
        else:
            cols = [""]
            assert config_dict["type"] == "link_prediction"
        if isinstance(cols, list) is False:
            cols = [cols]
        self._cols: List[str] = cols
        self._task_type: str = config_dict["type"]
        self._split: Dict[str, float] = config_dict["split_rate"]
        self._separator: str = config_dict["separator"] if "separator" in config_dict else None
        self._multilabel = self._separator is not None

    def _sanity_check(self):
        assert self._cols is not None and self._cols != "", (
            "The header names of the column storing the "
            "target label property value should be provided."
        )
        assert isinstance(self._cols, list)
        assert isinstance(self._task_type, str)
        assert isinstance(self._split, dict)
        assert isinstance(self._separator, str) if self._multilabel else self._separator is None

    @property
    def cols(self) -> Sequence[str]:
        """The columns this label configuration concerns."""
        return self._cols

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
