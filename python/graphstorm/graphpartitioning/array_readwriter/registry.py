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
"""
REGISTRY = {}


def register_array_parser(name):
    """Decorator function."""

    def _deco(cls):
        REGISTRY[name] = cls
        return cls

    return _deco


def get_array_parser(**fmt_meta):
    """User interfacing function to retrieve appropriate class
    for a read/write task.

    Argument
    --------
    fmt_meta: dict
        Dictionary of parameters.
    """
    cls = REGISTRY[fmt_meta.pop("name")]
    return cls(**fmt_meta)
