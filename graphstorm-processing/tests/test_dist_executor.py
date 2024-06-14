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

import os
import shutil
import tempfile

import pytest

from graphstorm_processing.distributed_executor import DistributedExecutor, ExecutorConfig
from graphstorm_processing.constants import ExecutionEnv, FilesystemType

_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(autouse=True, name="tempdir")
def tempdir_fixture():
    """Create temp dir for output files"""
    tempdirectory = tempfile.mkdtemp(
        prefix=os.path.join(_ROOT, "resources/test_output/"),
    )
    yield tempdirectory
    shutil.rmtree(tempdirectory)


def test_merge_input_and_transform_dicts(tempdir: str):
    """Test run function with local data"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    executor_configuration = ExecutorConfig(
        local_config_path=input_path,
        local_metadata_output_path=tempdir,
        input_prefix=input_path,
        output_prefix=tempdir,
        num_output_files=-1,
        config_filename="gsprocessing-config.json",
        execution_env=ExecutionEnv.LOCAL,
        filesystem_type=FilesystemType.LOCAL,
        add_reverse_edges=True,
        graph_name="small_heterogeneous_graph",
        do_repartition=True,
    )

    dist_executor = DistributedExecutor(executor_configuration)

    pre_comp_transormations = {
        "node_features": {
            "user": {
                "state": {
                    "transformation_name": "categorical",
                }
            }
        },
        "edge_features": {},
    }

    input_config_with_transforms = dist_executor._merge_config_with_transformations(
        dist_executor.gsp_config_dict,
        pre_comp_transormations,
    )

    # Ensure the "user" node type's "age" feature includes a transformation entry
    for node_input_dict in input_config_with_transforms["graph"]["nodes"]:
        if "user" == node_input_dict["type"]:
            for feature in node_input_dict["features"]:
                if "state" == feature["column"]:
                    transform_for_feature = feature["precomputed_transformation"]
                    assert transform_for_feature["transformation_name"] == "categorical"
