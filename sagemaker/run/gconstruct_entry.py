"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    SageMaker graph construction entry point
"""
import argparse
import sys
import subprocess

def parse_construct_args():
    """  Add arguments for launch gconstruct using SageMaker
    """
    parser = argparse.ArgumentParser(description='gs sagemaker graph construction pipeline')

    parser.add_argument("--graph-config-path", type=str,
        required=True, help="Graph configuration file")
    parser.add_argument("--input-path", type=str,
        required=True, help="Path to input graph data")
    parser.add_argument("--output-path", type=str,
        required=True, help="Path to store output")
    parser.add_argument("--graph-name", type=str,
        required=True, help="Graph name")
    return parser


if __name__ =='__main__':
    parser = parse_construct_args()
    args, unknownargs = parser.parse_known_args()

    subprocess.check_call(['ls'], cwd=args.input_path, shell=False)
    subprocess.run(["df", "-h"], check=True)

    launch_cmd = ['python3', '-m', 'graphstorm.gconstruct.construct_graph',
             '--conf-file', args.graph_config_path, '--output-dir', args.output_path,
             '--graph-name', args.graph_name] + unknownargs

    try:
        subprocess.check_call(launch_cmd, cwd=args.input_path, shell=False)
    except Exception:
        sys.exit(-1)
