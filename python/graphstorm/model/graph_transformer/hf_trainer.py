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

class GsHuggingfaceTrainer(Trainer):
    """ Customize Huggingface Trainer
    """
    def __init__(self, train_loader, val_loader, test_loader, **kwargs):
        self._train_dataloader = train_loader
        self._val_dataloader = val_loader
        self._test_dataloader = test_loader
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

        return self.accelerator.prepare(self._train_dataloader)

    def get_eval_dataloader(self, eval_dataset=None):
        return self.accelerator.prepare(self._val_dataloader)

    def get_test_dataloader(self, test_dataset=None):
        return self.accelerator.prepare(self._test_dataloader)
