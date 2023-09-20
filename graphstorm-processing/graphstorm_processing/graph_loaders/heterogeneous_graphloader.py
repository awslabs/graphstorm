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
from .loader_base import GraphLoader


class HeterogeneousGraphLoader(GraphLoader):
    """DGL Heterogeneous Graph Loader

    Parameters
    ----------
    data_path: str
        Path to input.
    local_metadata_path: str
        Local metadata files output path. For SageMaker it is from os.environ['SM_MODEL_DIR'].
    data_config: dict
        Config for loading raw data.
    """

    def __init__(self, data_path, local_metadata_path, data_configs):
        super().__init__(
            data_path=data_path,
            local_metadata_path=local_metadata_path,
            data_configs=data_configs,
        )
