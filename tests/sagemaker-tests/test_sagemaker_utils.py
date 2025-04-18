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
"""
import os
import tempfile
import tarfile
from launch_utils import wrap_model_artifacts
from numpy.testing import assert_raises


def create_dummy_file(file_path):
    """ Create an empty dummy file for testing
    """
    if os.path.exists(file_path):
        return

    with open(file_path, 'w') as f:
        f.close()

def test_wrap_model_artifacts():
    """
    """
    # test case 1: normal case, everything is given and the tar file created.
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(tmpdirname, exist_ok=True)
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)

        output_path = os.path.join(tmpdirname, 'output_folder')
        os.makedirs(output_path, exist_ok=True)
        output_file = wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model', output_path=output_path)
        assert os.path.exists(output_file)
    
        tar_object = tarfile.open(output_file)
        contents = tar_object.getmembers()
        results = {content.name: content.isfile() for content in contents}

        assert 'code' in results        
        assert 'code/nc_infer_entry.py' in results and \
               results['code/nc_infer_entry.py'] == True
        assert 'model.bin' in results and results['model.bin'] == True
        assert 'test.yaml' in results and results['test.yaml'] == True
        assert 'test.json' in results and results['test.json'] == True
        tar_object.close()

    # test case 2: abnormal cases
    #     2.1: missing one of the four files
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(tmpdirname, exist_ok=True)
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        output_path = os.path.join(tmpdirname, 'output_folder')
        os.makedirs(output_path, exist_ok=True)

        # not create the entry point file
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model', output_path=output_path)

        os.remove(model_path)
        create_dummy_file(entry_path)
        # not create the model.bin file.
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model', output_path=output_path)

        os.remove(yaml_path)
        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        # not create the test.yaml file.
        create_dummy_file(json_path)
        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model', output_path=output_path)

        os.remove(json_path)
        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        # not create the test.json file.
        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model', output_path=output_path)

    #     2.2: missing output folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(tmpdirname, exist_ok=True)
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)

        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model')

    #     2.3: given folders instead of files
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(tmpdirname, exist_ok=True)
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        # create a entry point file as a folder
        os.makedirs(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)

        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model')

        create_dummy_file(entry_path)
        # create a model file as a folder
        os.remove(model_path)
        os.makedirs(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)

        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        # create a yaml file as a folder
        os.remove(yaml_path)
        os.makedirs(yaml_path)
        create_dummy_file(json_path)

        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        # create a json file as a folder
        os.remove(json_path)
        os.makedirs(json_path)

        with assert_raises(AssertionError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                        output_package_name='model')
