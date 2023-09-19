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

    Utils for GNN distillation.
"""

import torch as th

def to_device(inputs, device='cuda'):
        """ Move the mini batch to corresponding device.

        Parameters
        ----------
        inputs: dict of tensor
            A batch from dataloader.
        device : str
            Name for the local device.

        Returns
        -------
        dict of tensor : A batch on the specified device.
        """
        if inputs is None:
            return None
        elif isinstance(inputs, th.Tensor):
            return inputs.to(device)
        elif isinstance(inputs, dict):
            outputs = {}
            for k, v in inputs.items():
                outputs[k] = to_device(v, device=device)
        elif isinstance(inputs, (list, tuple)):
            outputs = []
            for v in inputs:
                outputs.append(to_device(v, device=device))
        else:
            raise NotImplementedError
        return outputs