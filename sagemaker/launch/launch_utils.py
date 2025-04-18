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

    Utility functions for inference
"""

import os
import shutil
import tarfile


def wrap_model_artifacts(path_to_model, path_to_yaml, path_to_json, path_to_entry,
                         output_path, output_tarfile_name='model'):
    """ A utility function to zip model artifacts into a tar package

    According to SageMaker's specification of the `Model Directory Structure
    https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#model-directory-structure`_,
    this function will put the entry point file into a sub-folder,named `code`, and then zip the
    `model.bin`, the `**.yaml`, the `**.json` file, and the `code` sub-folder into a tar package
    with the name specified by the wrap_name argument, and save it to the `output_path`.

    This function will check if given files exist. Non-existance will raise an error.
    
    Parameters
    ----------
    path_to_model: str
        The path of GraphStorm's `model.bin` file.
    path_to_yaml: str
        The path of the YAML file generated by GraphStorm during model training.
    path_to_json: str
        The path of JSON file generated by GraphStorm gconstruct or GSProcessing.
    path_to_entry: str
        The path of the entry point file for a specific task. The file will be put into a
        sub-folder, named 'code'.
    output_path: str
        The folder where the output tar package will be saved. If not provided, will
        raise an error.
    output_tarfile_name: str
        The name of the tar package. Default is `model`.

    """
    # check if files exist
    assert os.path.exists(path_to_model), f'The model file, {path_to_model}, does not exist.'
    assert os.path.isfile(path_to_model), f'The model file, {path_to_model}, should be a file \
                                            path, but got a folder.'
    assert os.path.exists(path_to_yaml), f'The model configuration YAML file, {path_to_yaml}, \
                                           does not exist.'
    assert os.path.isfile(path_to_yaml), f'The model configuration YAML file, {path_to_yaml}, \
                                           should be a file path, but got a folder.'
    assert os.path.exists(path_to_json), f'The graph metadata JSON file, {path_to_json}, does \
                                           not exist.'
    assert os.path.isfile(path_to_json), f'The graph metadata JSON file, {path_to_json}, should \
                                           be a file path, but got a folder.'
    assert os.path.exists(path_to_entry), f'The SageMaker entry point file, {path_to_entry}, does \
                                           not exist.'
    assert os.path.isfile(path_to_entry), f'The SageMaker entry point file, {path_to_entry}, \
                                           should be a file path, but got a folder.'

    # create the output folder if not exist.
    assert os.path.isdir(output_path), 'Output path should be a folder name, but got a ' + \
                                       f'file {output_path}.'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'code'), exist_ok=True)

    # copy the artifacts files to output_path
    shutil.copy(path_to_entry, os.path.join(output_path, 'code', os.path.basename(path_to_entry)))
    if output_path != os.path.dirname(path_to_model):
        shutil.copy(path_to_model, os.path.join(output_path, os.path.basename(path_to_model)))
    if output_path != os.path.dirname(path_to_yaml):
        shutil.copy(path_to_yaml, os.path.join(output_path, os.path.basename(path_to_yaml)))
    if output_path != os.path.dirname(path_to_json):
        shutil.copy(path_to_json, os.path.join(output_path, os.path.basename(path_to_json)))

    output_file = os.path.join(output_path, output_tarfile_name + '.tar.gz')
    with tarfile.open(output_file, 'w:gz') as tar:
        tar.add(output_path, arcname='')

    return output_file
