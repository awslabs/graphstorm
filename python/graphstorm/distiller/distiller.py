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

    Distill wrapper.
"""

import torch as th
from torch.nn.parallel import DistributedDataParallel
import os
import json
import logging

import graphstorm as gs
from .utils import to_device
from ..utils import barrier

class GSdistiller():
    """ GNN distiller.
    This class defines the running pipeline of distillation.
    It trains the student model and saves the checkpoint.

    Parameters
    ----------
    model : GSDistilledModel
        The student model.
    """
    def __init__(self, model):
        self.rank = gs.get_rank()
        self.model = model
        self._device = None

    def fit(
        self, 
        train_data_mgr,
        eval_data_mgr,
        distill_lr,
        saved_path,
        save_model_frequency,
        eval_frequency,
        max_distill_step,
        on_cpu=False,
    ):
        """ Distill function, which trains student model.

        Parameters
        ----------
        train_data_mgr : DataManager
            Training data manager.
        eval_data_mgr : DataManager
            Evaluation data manager.
        distill_lr : float
            Learning rate for distillation.
        saved_path : str
            Path to save the model.
        save_model_frequency : int,
            Interval to save the model checkpoint.
        eval_frequency : int,
            Interval for evaluation.
        max_distill_step : int,
            Maximum steps for distillation training.
        on_cpu : bool
            Whether the distillation will be conducted on cpu.
        """

        # TODO (HZ): add flexibility to specify different optimizers
        optimizer = th.optim.Adam(self.model.parameters(), lr=distill_lr)
        self.model.to(self.device)
        model = DistributedDataParallel(self.model, device_ids=None if on_cpu else [self.device],
                                        output_device=None if on_cpu else self.device)

        index = 0
        distill_step = 0
        complete=False
        while complete is False:
            barrier()
            logging.info(f"Train {index + 1}-th shard by trainer {self.rank}")
            complete, distill_step = self.train_shard(
                distill_step=distill_step,
                model=model, 
                optimizer=optimizer, 
                train_data_mgr=train_data_mgr, 
                eval_data_mgr=eval_data_mgr,
                saved_path=saved_path,
                save_model_frequency=save_model_frequency,
                eval_frequency=eval_frequency,
                max_distill_step=max_distill_step,
            )
            index += 1

    def save_student_model(self, model, saved_path, distill_step):
        """ Save student model

        Parameters
        ----------
        model : GSDistilledModel
            Distilled student model.
        saved_path : str
            Path to save the model.
        distill_step : int
            Distill step of the model checkpoint.
        """
        barrier()
        if self.rank == 0:
            checkpoint_path = os.path.join(saved_path, f"checkpoint-{distill_step}")
            proj_dir_loc = os.path.join(checkpoint_path, "proj")
            tokenizer_dir_loc = os.path.join(checkpoint_path, "tokenizer")
            lm_dir_loc = os.path.join(checkpoint_path, "lm")
            os.makedirs(proj_dir_loc, exist_ok=True)
            logging.info(f"Saving checkpoint to {checkpoint_path}")

            model.module.tokenizer.save_pretrained(tokenizer_dir_loc)
            model.module.lm.save_pretrained(lm_dir_loc)
            th.save(model.module.state_dict()["proj"], os.path.join(proj_dir_loc, "pytorch_model.bin"))

        return True

    def eval(self, model, eval_data_mgr, distill_step):
        """ Evaluate student model on validation set.
        The metric are mean square error (MSE).

        Parameters
        ----------
        model : GSDistilledModel
            Distilled student model.
        eval_data_mgr : DataManager
            Data manager for validation data.
        distill_step : int
            Distill step of the model checkpoint.
        """
        model.eval()
        index = 0
        batch_index = 0
        total_mse = 0
        while True:
            dataset_iterator = eval_data_mgr.get_iterator()
            if dataset_iterator is None:
                break
            logging.info(f"Eval {index + 1}-th shard by trainer {self.rank}")
            with th.no_grad():
                for _, batch in enumerate(dataset_iterator):
                    batch = to_device(batch, self.device)  # Move to device
                    mse = model.module(batch["input_ids"],
                        batch["attention_mask"],
                        batch["labels"])
                    total_mse += mse.item()
                    batch_index += 1
            index += 1

        mean_total_mse = total_mse / batch_index
        logging.info(f"Eval MSE at step {distill_step}: {mean_total_mse}")
        model.train()
        eval_data_mgr.release_iterator()

    def train_shard(
        self, 
        distill_step, 
        model, 
        optimizer, 
        train_data_mgr, 
        eval_data_mgr, 
        saved_path,
        save_model_frequency,
        eval_frequency,
        max_distill_step,
    ):
        """
        Train using one shard from train_data_mgr.
        Parameters
        ----------
        distill_step : int
            Distill step of the model checkpoint.
        model : GSDistilledModel
            Distilled student model.
        optimizer : torch optimizer
            optimizer for distillation.
        train_data_mgr : DataManager
            Data manager for training data.
        eval_data_mgr : DataManager
            Data manager for validation data.
        saved_path : str
            Path to save the model.
        save_model_frequency : int,
            Interval to save the model checkpoint.
        eval_frequency : int,
            Interval for evaluation.
        max_distill_step : int,
            Maximum steps for distillation training.
    
        Returns
        -------
        bool : whether to stop distillation.
        int : Distill step of the model checkpoint.
        """
        dataset_iterator = train_data_mgr.get_iterator()
        if not dataset_iterator:
            raise RuntimeError("No training data. Check training dataset as the data sampler"
                                    "is supposed to be infinite.")

        # TODO (HZ): support prefetch to speed up the training
        model.train()
        shard_loss = 0
        complete = False

        for batch_num, batch in enumerate(dataset_iterator):
            try:
                batch = to_device(batch, self.device)  # Move to device

                loss = model(batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"])

                shard_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # TODO (HZ): Need to review how to report loss.
                mean_shard_loss = shard_loss / (batch_num + 1)
                if distill_step % 20 == 0:
                    logging.info(
                        f"Loss for shard {train_data_mgr.get_iterator_name()}"
                        f" at step {distill_step} = {mean_shard_loss}"
                        f" from trainer {self.rank}"
                    )

                if distill_step % save_model_frequency == 0 and distill_step != 0:
                    self.save_student_model(model, saved_path, distill_step)

                if distill_step % eval_frequency == 0 and distill_step != 0:
                    barrier()
                    # TODO (HZ): implement distributed evaluation by communicating with all trainers
                    if self.rank == 0:
                        self.eval(model, eval_data_mgr, distill_step)
                    barrier()

                if distill_step == max_distill_step:
                    complete = True
                    break
                distill_step += 1
            except StopIteration:
                continue

        # release the memory
        train_data_mgr.release_iterator()

        return complete, distill_step

    def setup_device(self, device):
        """ Set up the device for the distillation.
        The device is set up based on the local rank.
        Parameters
        ----------
        device :
            The device for distillation.
        """
        self._device = th.device(device)

    @property
    def device(self):
        """ The device associated with the inference.
        """
        return self._device