"""
    Copyright 2025 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Utils to package trained model artifacts and upload to S3 bucket
"""

import os
import argparse
from graphstorm.config import BUILTIN_TASK_NODE_CLASSIFICATION
from graphstorm.config import BUILTIN_TASK_NODE_REGRESSION
from graphstorm.config import BUILTIN_TASK_EDGE_CLASSIFICATION
from graphstorm.config import BUILTIN_TASK_EDGE_REGRESSION
from graphstorm.inference import wrap_model_artifacts

SUPPORTED_TASKS = [BUILTIN_TASK_NODE_CLASSIFICATION,
                   BUILTIN_TASK_NODE_REGRESSION,
                   BUILTIN_TASK_EDGE_CLASSIFICATION,
                   BUILTIN_TASK_EDGE_REGRESSION]

def main(args):
    """ Wrap model artifacts and entry point file into a compressed tar file.
    """
    # chose the target entry point file according to task
    entry_base_path = 'python/graphstorm/sagemaker/deploy/'
    if args.gml_task == BUILTIN_TASK_NODE_CLASSIFICATION:
        path_to_entry = os.path.join(args.graphstorm_home_folder, entry_base_path,
                                     'node_classification_entry_point.py')
    elif args.gml_task == BUILTIN_TASK_NODE_REGRESSION:
        path_to_entry = os.path.join(args.graphstorm_home_folder, entry_base_path,
                                     'node_regression_entry_point.py')
    elif args.gml_task == BUILTIN_TASK_EDGE_CLASSIFICATION:
        path_to_entry = os.path.join(args.graphstorm_home_folder, entry_base_path,
                                     'edge_classification_entry_point.py')
    elif args.gml_task == BUILTIN_TASK_EDGE_REGRESSION:
        path_to_entry = os.path.join(args.graphstorm_home_folder, entry_base_path,
                                     'edge_regression_entry_point.py')
    else:
        raise NotImplementedError(f'Not supported GML task {args.gml_task}.')

    output_file = wrap_model_artifacts(args.model_file_path,
                                        args.model_config_yaml_paath,
                                        args.graph_metadata_json_path,
                                        path_to_entry, wrap_name='model',
                                        output_path=args.output_folder)
    print(f'Successfully pack GraphStorm model artifacts into {output_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Package GraphStorm model artifacts.')
    parser.add_argument('--model-file-path', type=str, required=True,
                        help='The file path of trained GraphStorm model. By default, ' + \
                             'GraphStorm saves model parameters into a binary file, named ' + \
                             '\'model.bin\' file.')
    parser.add_argument('--model-config-yaml-path', type=str, required=True,
                        help='The file path of model configuration YAML file. By default, ' + \
                             'GraphStorm generates a new YAML file that stores all model ' + \
                             'configurations during a training.')
    parser.add_argument('--graph-metadata-json-path', type=str, required=True,
                        help='The file path of graph metadata JSON file. By default, ' + \
                             'GraphStorm generates a new JSON file that stores all graph ' + \
                             'metadata during a graph construction.')
    parser.add_argument('--graphstorm-home-folder', type=str, required=True,
                        help='The home folder path of GraphStorm source code.')
    parser.add_argument('--gml-task-type', type=str, choices=SUPPORTED_TASKS, required=True,
                        help='The GML task that a GraphStorm model was trained for. Options ' + \
                             f'include {SUPPORTED_TASKS}.')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='The output folder of the packed model artifacts.')

    args = parser.parse_args()
    main(args)