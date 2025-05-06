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
import argparse
import tempfile
import tarfile
import pytest
import boto3
import sagemaker as sm
from argparse import ArgumentTypeError
from unittest.mock import patch, Mock
from botocore.exceptions import ClientError
from urllib.parse import urlparse

from launch_utils import (wrap_model_artifacts,
                          check_tarfile_s3_object,
                          parse_s3_url,
                          upload_data_to_s3,
                          check_name_format)


def create_dummy_file(file_path):
    """ Create an empty dummy file for testing
    """
    if os.path.exists(file_path):
        return

    with open(file_path, 'w') as f:
        f.close()

def test_wrap_model_artifacts():
    """ Test the wrapping model artifacts function.
    """
    # test case 1: normal case, everything is given and the tar file created.
    #       1.1: all are given, including output folder
    with tempfile.TemporaryDirectory() as tmpdirname:
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
                                 output_path=output_path, output_tarfile_name='model')
        assert os.path.exists(output_file)
    
        with tarfile.open(output_file) as tar_object:
            contents = tar_object.getmembers()
            results = {content.name: content.isfile() for content in contents}

            assert 'code' in results        
            assert 'code/nc_infer_entry.py' in results and \
                results['code/nc_infer_entry.py'] == True
            assert 'model.bin' in results and results['model.bin'] == True
            assert 'test.yaml' in results and results['test.yaml'] == True
            assert 'test.json' in results and results['test.json'] == True

    #       1.2: all are given, but not create output folder. Should be created by wrap func.
    with tempfile.TemporaryDirectory() as tmpdirname:
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)

        output_path = os.path.join(tmpdirname, 'output_folder')
        output_file = wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')
        assert os.path.exists(output_file)
    
        with tarfile.open(output_file) as tar_object:
            contents = tar_object.getmembers()
            results = {content.name: content.isfile() for content in contents}

            assert 'code' in results        
            assert 'code/nc_infer_entry.py' in results and \
                results['code/nc_infer_entry.py'] == True
            assert 'model.bin' in results and results['model.bin'] == True
            assert 'test.yaml' in results and results['test.yaml'] == True
            assert 'test.json' in results and results['test.json'] == True

    # test case 2: abnormal cases
    #     2.1: missing one of the four files
    with tempfile.TemporaryDirectory() as tmpdirname:
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
        with pytest.raises(AssertionError, match='SageMaker entry point .* not exist'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')

        os.remove(model_path)
        create_dummy_file(entry_path)
        # not create the model.bin file.
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        with pytest.raises(AssertionError, match='model file, .* not exist'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')

        os.remove(yaml_path)
        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        # not create the test.yaml file.
        create_dummy_file(json_path)
        with pytest.raises(AssertionError, match='YAML .* not exist'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')

        os.remove(json_path)
        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        # not create the test.json file.
        with pytest.raises(AssertionError, match='JSON .* not exist'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')

    #     2.2: missing output folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        with pytest.raises(TypeError):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_tarfile_name='model')

    #     2.3: output is not a folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        output_path = os.path.join(tmpdirname, 'output_folder')
        create_dummy_file(output_path)        
        with pytest.raises(AssertionError, match='Output path should be a folder name, but got'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')
    

    #     2.4: given folders instead of files
    with tempfile.TemporaryDirectory() as tmpdirname:
        entry_path = os.path.join(tmpdirname, 'nc_infer_entry.py')
        model_path = os.path.join(tmpdirname, 'model.bin')
        yaml_path = os.path.join(tmpdirname, 'test.yaml')
        json_path = os.path.join(tmpdirname, 'test.json')

        output_path = os.path.join(tmpdirname, 'output_folder')
        os.makedirs(output_path, exist_ok=True)

        # create a entry point file as a folder
        os.makedirs(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        with pytest.raises(AssertionError, match='SageMaker entry point file, .* got a folder'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')
        os.rmdir(entry_path)

        create_dummy_file(entry_path)
        # create a model file as a folder
        os.remove(model_path)
        os.makedirs(model_path)
        create_dummy_file(yaml_path)
        create_dummy_file(json_path)
        with pytest.raises(AssertionError, match='model file, .* got a folder'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')
        os.rmdir(model_path)

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        # create a yaml file as a folder
        os.remove(yaml_path)
        os.makedirs(yaml_path)
        create_dummy_file(json_path)
        with pytest.raises(AssertionError, match='model configuration YAML file, .* got a folder'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')
        os.rmdir(yaml_path)

        create_dummy_file(entry_path)
        create_dummy_file(model_path)
        create_dummy_file(yaml_path)
        # create a json file as a folder
        os.remove(json_path)
        os.makedirs(json_path)
        with pytest.raises(AssertionError, match='graph metadata JSON file, .* got a folder'):
            wrap_model_artifacts(model_path, yaml_path, json_path, entry_path,
                                 output_path=output_path, output_tarfile_name='model')

def test_parse_s3_url():
    """ Test the parse S3 url function.
    """
    # Test case 1:  normal case, using valid S3 url.
    #       1.1: start with 's3://' or 'S3://'
    test_s3_url = 's3://graphstorm-demos/test/GSF-demo.pptx'
    bucket_name, key = parse_s3_url(test_s3_url)
    test_s3_url = 'S3://graphstorm-demos/test/GSF-demo.pptx'
    bucket_name, key = parse_s3_url(test_s3_url)
    
    assert bucket_name == 'graphstorm-demos'
    assert key == 'test/GSF-demo.pptx'
    
    #       1.2: start with 'https://'
    test_s3_url = 'https://graphstorm-demos.s3.us-west-1.amazonaws.com/test/GSF-demo.pptx'
    bucket_name, key = parse_s3_url(test_s3_url)
    
    assert bucket_name == 'graphstorm-demos.s3.us-west-1.amazonaws.com'
    assert key == 'test/GSF-demo.pptx'

    # Test case 2: abnormal cases, not start either s3:// or https://
    test_s3_url = '/graphstorm-demos/test/GSF-demo.pptx'
    with pytest.raises(AssertionError, match='Incorrect S3 *'):
        parse_s3_url(test_s3_url)

def test_check_name_format():
    """ test the check_name_format functions
    
    The naming regular expression: ^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9]).
    It means the string must start with a letter or digit. In the middle, it could be one or more
    hyphons, letters, or digits. And the string must end with a letter or digit.
    """
    # Test case 1: normal case, following the regex, including all three parts and all match
    valid_names = ['ab-cd-9',  'abc', 'a-b', 'A1-foo2', 'Z9', 'x-y-z', 'abc123']
    for valid_name in valid_names:
        assert check_name_format(valid_name) == valid_name

    # Test case 2: abnormal cases
    invalid_names = [
                "-abc",     # starts with hyphen
                "abc-",     # ends with hyphen
                "a--",      # ends with hyphen
                "a_",       # contains invalid character
                "a",        # too short to match ([...]*[a-zA-Z0-9])
                ""         # empty string
            ]
    for invalid_name in invalid_names:
        with pytest.raises(ArgumentTypeError, match='failed to satisfy regular expression pattern'):
            check_name_format(invalid_name)

@patch('launch_utils.boto3.client')
def test_check_tarfile_s3_object(mock_boto_client):
    """ The the check if tarfile object in S3 url correct and exist
    """
    mock_s3 = Mock()
    mock_boto_client.return_value = mock_s3

    # Test case 1: normal case. S3 url is right and the object ends with '.tar.gz' 
    mock_s3.head_object.return_value = "s3://graphstorm-demos/test/model.tar.gz"
    assert check_tarfile_s3_object("s3://graphstorm-demos/test/model.tar.gz") is True

    # Test case 2: abnormal cases
    #       2.1: S3 url is incorrect, not starting with s3 or https
    error_response = {'Error': {'Code': '404'}}
    mock_s3.head_object.side_effect = ClientError(error_response, 'HeadObject')
    with pytest.raises(AssertionError, match='Incorrect S3'):
        check_tarfile_s3_object('/graphstorm-demos/test/GSF-demo.pptx')

    #       2.2: S3 url is correct, but not ending with .tar.gz
    error_response = {'Error': {'Code': '404'}}
    mock_s3.head_object.side_effect = ClientError(error_response, 'HeadObject')
    with pytest.raises(AssertionError, match='not a compressed tar file'):
        check_tarfile_s3_object('s3://graphstorm-demos/test/GSF-demo.pptx')

@patch('launch_utils.S3Uploader.upload')
def test_upload_data_to_s3(mock_s3uploader):
    """ Test the upload data to S3 function.
    """
    mock_s3uploader.return_value = 's3://graphstorm-demos/test/model.tar.gz'
    
    # Test case 1: mock successful upload
    ret = upload_data_to_s3('s3://graphstorm-demos/test/', './model.tar.gz', 'session')
    mock_s3uploader.assert_called_once_with('./model.tar.gz', 's3://graphstorm-demos/test/',
                                            sagemaker_session='session')

    assert ret == 's3://graphstorm-demos/test/model.tar.gz'

    # Test case 2: mock unsucessful upload
    ret = upload_data_to_s3('s3://graphstorm-demos/test/', './model.tar.gz', 'session')
    with pytest.raises(AssertionError):
        mock_s3uploader.assert_called_once_with('s3://graphstorm-demos/test/', './model.tar.gz',
                                                sagemaker_session='session')
    

def local_test_check_tarfile_s3_object(input_value, expected_result):
    """ Test the check if a model tarfile exist in the given S3 url location.

    ## This test should NOT be ran in CI pipeline, but in local environment only! ##
    
    Testers need to provide an S3 url as the input value, and True or False as the
    expected result in the python command line.

    """
    s3_url = input_value

    if expected_result.lower() == 'true':
        assert check_tarfile_s3_object(s3_url)
    else:
        assert not check_tarfile_s3_object(s3_url)


def local_test_upload_data_to_s3(input_value, expected_result=None):
    """ Test the check if can upload a file to the given S3 url location.

    ## This test should NOT be ran in CI pipeline, but in local environment only! ##
    
    Testers need to provide an S3 url as the input value. This test function will create
    a temporary emtpy file as the data to be uploaded.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = os.path.join(tmpdirname, 'data.txt')
        create_dummy_file(data_path)

        # prepare sessions
        b3_session = boto3.Session(region_name='us-west-2')
        sm_session = sm.Session(boto_session=b3_session)

        s3_path = input_value
        ret = upload_data_to_s3(s3_path, data_path, sm_session)

    expected_result = input_value + '/data.txt'
    assert ret == expected_result

    # delete the uploaded object
    parsed_url = urlparse(ret)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')
    s3 = boto3.client('s3')
    s3.delete_object(Bucket=bucket_name, Key=object_key)


if __name__ == '__main__':

    # Should only run this in local environment where service access is configured.
    # Comment out the parser when run locally
    # parser = argparse.ArgumentParser('Local test for services')
    # parser.add_argument('--input-value', type=str, help='Input value of a test function.')
    # parser.add_argument('--expected-result', type=str, help='The expected result of a test.')
    # args = parser.parse_args()

    # Comment out all excpet the one to be tested.
    # local_test_check_tarfile_s3_object(args.input_value, args.expected_result)

    # local_test_upload_data_to_s3(args.input_value)
