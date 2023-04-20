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

    Entry point for running node regression tasks

    Run as:
    python3 -m graphstorm.run.gs_node_regression <Launch args> <Train/Infer args>
"""
import os
import logging

from .launch import get_argument_parser
from .launch import check_input_arguments
from .launch import submit_jobs
from argparse import REMAINDER

def main():
    parser = get_argument_parser()
    args, training_script_args = parser.parse_known_args()
    check_input_arguments(args)

    lib_dir = os.path.abspath(os.path.dirname(__file__))
    if args.inference:
        cmd_path = os.path.join(lib_dir, "gsgnn_np/np_infer_gnn.py")
    else:
        cmd_path = os.path.join(lib_dir, "gsgnn_np/gsgnn_np.py")
    training_script_args = [cmd_path] + training_script_args

    submit_jobs(args, training_script_args)

if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()
