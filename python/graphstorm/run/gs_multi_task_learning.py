"""
    Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Entry point for running multi-task learning.

    Run as:
    python3 -m graphstorm.run.gs_multi_task_learning <Launch args> <Train/Infer args>
"""
import os
import logging

from .launch import get_argument_parser
from .launch import check_input_arguments
from .launch import submit_jobs

def main():
    """ Main function
    """
    parser = get_argument_parser()
    args, exec_script_args = parser.parse_known_args()
    check_input_arguments(args)

    lib_dir = os.path.abspath(os.path.dirname(__file__))
    if args.inference:
        assert False, "Not implemented"
    else:
        cmd = "gsgnn_mt/gsgnn_mt.py"
    cmd_path = os.path.join(lib_dir, cmd)
    exec_script_args = [cmd_path] + exec_script_args

    if "coo" not in args.graph_format:
        args.graph_format = f"{args.graph_format},coo"
        logging.debug("Automatically add COO format to graph formats for link prediction. " + \
                "New graph_format is %s", args.graph_format)
    submit_jobs(args, exec_script_args)

if __name__ == "__main__":
    FMT = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=FMT, level=logging.INFO)
    main()
