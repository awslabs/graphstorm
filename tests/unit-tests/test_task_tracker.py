"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

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
from argparse import Namespace

from graphstorm.tracker import get_task_tracker_class, GSSageMakerTaskTracker, GSTensorBoardTracker
from graphstorm.config import (GSConfig,
                               GRAPHSTORM_SAGEMAKER_TASK_TRACKER,
                               GRAPHSTORM_TENSORBOARD_TASK_TRACKER)
from graphstorm.gsf import create_builtin_task_tracker

def test_get_tracker_class():
    tracker_class = get_task_tracker_class(GRAPHSTORM_SAGEMAKER_TASK_TRACKER)
    assert isinstance(tracker_class, GSSageMakerTaskTracker)

    tracker_class = get_task_tracker_class(GRAPHSTORM_TENSORBOARD_TASK_TRACKER)
    assert isinstance(tracker_class, GSTensorBoardTracker)

    # default setting
    tracker_class = get_task_tracker_class("default")
    assert isinstance(tracker_class, GSSageMakerTaskTracker)

def test_create_builtin_task_tracker():
    args = Namespace(task_tracker=GRAPHSTORM_SAGEMAKER_TASK_TRACKER,
                     task_tracker_logpath=None,
                     eval_frequency=10)
    config = GSConfig(args)
    tracker = create_builtin_task_tracker(config)
    assert isinstance(tracker, GSSageMakerTaskTracker)

    args = Namespace(task_tracker=GRAPHSTORM_SAGEMAKER_TASK_TRACKER,
                     task_tracker_logpath="log",
                     eval_frequency=10)
    config = GSConfig(args)
    tracker = create_builtin_task_tracker(config)
    assert isinstance(tracker, GSSageMakerTaskTracker)

    args = Namespace(task_tracker=GRAPHSTORM_TENSORBOARD_TASK_TRACKER,
                     task_tracker_logpath=None,
                     eval_frequency=10)
    config = GSConfig(args)
    tracker = create_builtin_task_tracker(config)
    assert isinstance(tracker, GSTensorBoardTracker)

    args = Namespace(task_tracker=GRAPHSTORM_TENSORBOARD_TASK_TRACKER,
                     task_tracker_logpath="log",
                     eval_frequency=10)
    config = GSConfig(args)
    tracker = create_builtin_task_tracker(config)
    assert isinstance(tracker, GSTensorBoardTracker)

if __name__ == '__main__':
    test_get_tracker_class()
    test_create_builtin_task_tracker()