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

    Customized huggingface trainer for GraphStorm
"""

from transformers import Trainer
from ...dataloading import get_graph_lm_dataloader
from .utils import get_prepare_lm_input

class GsHuggingfaceTrainer(Trainer):
    """ Customize Huggingface Trainer
    """
    def __init__(self, gs_dataloader, device, gs_config, **kwargs):
        self._gs_dataloader = gs_dataloader
        self._gs_config = gs_config
        self._device = device
        super(GsHuggingfaceTrainer, self).__init__(kwargs)

    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # initialize graph dataloader
        train_dataset = self.train_dataset

        dataloader_params = {
            "device": self._device,
            "train_task": True,
            "target_idx": train_dataset.train_idxs,
            "batch_size": self.args.per_device_train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        dataloader = get_graph_lm_dataloader(self._gs_dataloader)
        prepare_input_fn = get_prepare_lm_input(self._gs_dataloader)

        return self.accelerator.prepare(dataloader(train_dataset, prepare_input_fn, **self.gs_config, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset=None):
        # initialize graph dataloader
        train_dataset = self.train_dataset

        dataloader_params = {
            "device": self._device,
            "train_task": False,
            "target_idx": train_dataset.val_idxs,
            "batch_size": self.args.per_device_eval_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        dataloader = get_graph_lm_dataloader(self._gs_dataloader)
        prepare_input_fn = get_prepare_lm_input(self._gs_dataloader)

        return self.accelerator.prepare(dataloader(train_dataset, prepare_input_fn, **self.gs_config, **dataloader_params))

    def get_test_dataloader(self, test_dataset=None):
        # initialize graph dataloader
        train_dataset = self.train_dataset

        dataloader_params = {
            "device": self._device,
            "train_task": False,
            "target_idx": train_dataset.test_idxs,
            "batch_size": self.args.per_device_eval_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        dataloader = get_graph_lm_dataloader(self._gs_dataloader)
        prepare_input_fn = get_prepare_lm_input(self._gs_dataloader)

        return self.accelerator.prepare(dataloader(train_dataset, prepare_input_fn, **self.gs_config, **dataloader_params))

