"""
    Copyright Contributors

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
import pytest
import sys
import tempfile
from unittest import mock
from argparse import ArgumentTypeError
from common_parser import parse_unknown_gs_args
from launch_realtime_endpoint import (
    get_realtime_infer_parser, 
    sanity_check_realtime_infer_inputs,
    )
from config_utils import create_graph_config_json_object

def test_basic_parsing():
    args = ["--num-epochs", "1", "--use-graphbolt", "true"]
    expected = {"num-epochs": "1", "use-graphbolt": "true"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_multiple_values():
    args = [
        "--target-etype",
        "query,clicks,asin",
        "query,search,asin",
        "--feat-name",
        "ntype0:feat0",
        "ntype1:feat1",
    ]
    expected = {
        "target-etype": "query,clicks,asin query,search,asin",
        "feat-name": "ntype0:feat0 ntype1:feat1",
    }
    assert dict(parse_unknown_gs_args(args)) == expected


def test_empty_value():
    args = ["--empty-arg", "--next-arg", "value"]
    expected = {"empty-arg": "", "next-arg": "value"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_no_args():
    args = []
    result = parse_unknown_gs_args(args)
    assert len(result) == 0


def test_only_flags():
    args = ["--flag1", "--flag2", "--flag3"]
    expected = {"flag1": "", "flag2": "", "flag3": ""}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_mixed_args():
    args = ["--arg1", "value1", "--flag", "--arg2", "value2a", "value2b"]
    expected = {"arg1": "value1", "flag": "", "arg2": "value2a value2b"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_quoted_args():
    args = ["--arg", '"quoted value"', "--another-arg", "'single quoted'"]
    expected = {"arg": '"quoted value"', "another-arg": "'single quoted'"}
    assert dict(parse_unknown_gs_args(args)) == expected


@pytest.mark.parametrize(
    "input_args, expected_output",
    [
        (["--special-chars", "!@#$%^&*()"], {"special-chars": "!@#$%^&*()"}),
        (["--unicode", "こんにちは", "world"], {"unicode": "こんにちは world"}),
    ],
)
def test_parse_unknown_gs_args_parametrized(input_args, expected_output):
    assert dict(parse_unknown_gs_args(input_args)) == expected_output


def test_parse_single_string():
    """Happens when GS args are passed in quoted in bash:
    ``python launch_*.py --launch-arg 1 '--gs-arg1 2 --gs-arg2 3'``
    """
    args = ["--feat-name ntype0:feat0 ntype1:feat1"]
    expected = {"feat-name": "ntype0:feat0 ntype1:feat1"}
    assert dict(parse_unknown_gs_args(args)) == expected


def test_parse_complex_string_with_quotes():
    args = [
        (
            "--target-etype query,clicks,asin query,search,asin "
            "--feat-name ntype0:feat0 ntype1:feat1"
        )
    ]
    expected = {
        "target-etype": "query,clicks,asin query,search,asin",
        "feat-name": "ntype0:feat0 ntype1:feat1",
    }
    assert dict(parse_unknown_gs_args(args)) == expected


def test_get_realtime_infer_argparser():
    """ Test the default values of realtime parser.
    """
    test_cmd = "test_sagemaker_args.py"
    default_args = {'--instance-type': 'ml.c6i.xlarge',
                    '--instance-count': 1,
                    '--async-execution': 'true',
                    '--model-name': 'GSF-Model4Realtime',
                    '--log-level': 'INFO'}

    # Test case 1: normal cases
    #       1.1: for the three required arguments
    required_args = {'--image-uri': 'image_uri_val',
                     '--role': 'role_val',
                     '--region': 'region_val',
                     '--restore-model-path': 'model_path',
                     '--model-yaml-config-file': 'yaml_config_file',
                     '--graph-json-config-file': 'json_config_file',
                     '--upload-tarfile-s3': 'tarfile_s3',
                     '--infer-task-type': 'node_classification'}
    test_args = {**required_args, **default_args}
    test_args_str = [test_cmd] + [str(item) for pair in test_args.items() for item in pair]

    with mock.patch.object(sys, 'argv', test_args_str):
        arg_parser = get_realtime_infer_parser()
        args = arg_parser.parse_args()
        args_w_vals = {k: v for k, v in vars(args).items() if v is not None}
        expected_vals = {k.replace('-', '_')[2: ]: v for k, v in {**required_args, **default_args}.items()}
        assert args_w_vals == expected_vals

    #       1.2: for other arguments
    other_args = {'--model-name': 'test-model'     # here overwritten the default, and no '_'
                  }
    test_args = {**required_args, **default_args, **other_args}
    test_args_str = [test_cmd] + [str(item) for pair in test_args.items() for item in pair]

    with mock.patch.object(sys, 'argv', test_args_str):
        arg_parser = get_realtime_infer_parser()
        args = arg_parser.parse_args()
        args_w_vals = {k: v for k, v in vars(args).items() if v is not None}
        expected_vals = {k.replace('-', '_')[2: ]: v for k, v in test_args.items()}
        assert args_w_vals == expected_vals

    # Test case 2: abnormal cases
    #       2.1: missing a required argument
    for k, _ in required_args.items():
        new_required_args = {k1: v1 for k1, v1 in required_args.items() if k1 != k}
        test_args = {**new_required_args, **default_args}
        test_args_str = [test_cmd] + [str(item) for pair in test_args.items() for item in pair]

        with mock.patch.object(sys, 'argv', test_args_str):
            arg_parser = get_realtime_infer_parser()
            # parser will exit with non-zero code
            with pytest.raises(SystemExit) as excinfo:
                arg_parser.parse_args()
            assert excinfo.value.code != 0

    #       2.2: incorrect model_name argument
    other_args = {'--model-name': 'test_model'}     # not allow '_'
    test_args = {**required_args, **default_args, **other_args}
    test_args_str = [test_cmd] + [str(item) for pair in test_args.items() for item in pair]

    with mock.patch.object(sys, 'argv', test_args_str):
        arg_parser = get_realtime_infer_parser()
        # parser will exit with non-zero code
        with pytest.raises(SystemExit) as excinfo:
            arg_parser.parse_args()
        assert excinfo.value.code != 0

def test_sanity_check_realtime_infer_inputs():
    """ Test the argument logics defined in the sanity_check_realtime_infer_inputs function
    """
    with tempfile.TemporaryDirectory() as tmpdir:

        test_cmd = "test_sagemaker_args.py"
        default_args = {'--instance-type': 'ml.c6i.xlarge',
                        '--instance-count': 1,
                        '--async-execution': 'false',
                        '--model-name': 'GSF-Model4Realtime'}
        required_args = {'--image-uri': '123456789012.ecr.us-west-2.amazonaws.com/my-image:latest',
                        '--role': 'role_val',
                        '--region': 'us-west-2',
                        '--restore-model-path': 'model_path',
                        '--model-yaml-config-file': 'yaml_config_file',
                        '--graph-json-config-file': os.path.join(tmpdir, 'json_config_file'),
                        '--upload-tarfile-s3': 'tarfile_s3',
                        '--infer-task-type': 'node_classification'}

        # Test case 1: normal cases
        #       1.1 image url is in same region as the --region argument
        test_args = {**required_args, **default_args}
        test_args_str = [test_cmd] + [str(item) for pair in test_args.items() for item in pair]

        # create a real json file to test sanity check
        _ = create_graph_config_json_object(tmpdir, has_tokenize=False, json_fname='json_config_file')

        with mock.patch.object(sys, 'argv', test_args_str):
            arg_parser = get_realtime_infer_parser()
            args = arg_parser.parse_args()

            # should pass the check without raising errors
            sanity_check_realtime_infer_inputs(args)

        # Test case 2: abnormal cases
        #       2.1 image url is in different region from the --region argument
        other_args = {'--image-uri': '123456789012.ecr.us-west-2.amazonaws.com/my-image:latest',
                    '--region': 'us-east-1'}
        test_args = { **default_args, **required_args, **other_args}
        test_args_str = [test_cmd] + [str(item) for pair in test_args.items() for item in pair]

        with mock.patch.object(sys, 'argv', test_args_str):
            arg_parser = get_realtime_infer_parser()
            args = arg_parser.parse_args()

            with pytest.raises(ValueError, match=r'The given Docker image * '):
                sanity_check_realtime_infer_inputs(args)
