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

    Config related utils
"""

import json

def get_graph_name(part_cnfig):
    """ Get graph name from graph partition config file

    Parameter
    ---------
    part_config: str
        Path to graph partition config file

    Return
    ------
        graph_name
    """
    with open(part_cnfig, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config["graph_name"]
