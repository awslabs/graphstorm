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

    GraphStorm task tracker

    Builtin training tracker supports:
     - GSSageMakerTaskTracker: GraphStorm SageMaker Task Tracker
"""
from .graphstorm_tracker import GSTaskTrackerAbc
from .sagemaker_tracker import GSSageMakerTaskTracker

def get_task_tracker_class(tracker_name):
    """ Get builtin task tracker

    Parameters
    ----------
    tracker_name: str
        task tracker name. 'SageMaker' for GSSageMakerTaskTracker
    """
    if tracker_name == 'SageMaker':
        # SageMaker tracker also works as normal print tracker
        return GSSageMakerTaskTracker
    # TODO: Support mlflow, etc.
    else:
        # by default use GSSageMakerTaskTracker
        return GSSageMakerTaskTracker
