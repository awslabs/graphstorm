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
import os
import json
import logging
import torch as th
from torch.nn.parallel import DistributedDataParallel

import graphstorm as gs
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
    ):
        """ Distill function, which trains student model.

        Parameters
        ----------
        train_data_mgr : DistillDataManager
            Training data manager.
        eval_data_mgr : DistillDataManager
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
        """

        # TODO (HZ): add flexibility to specify different optimizers
        optimizer = th.optim.Adam(self.model.parameters(), lr=distill_lr)
        self.model.to(self.device)
        model = DistributedDataParallel(self.model, device_ids=[self.device],
                                        output_device=self.device)

        index = 0
        distill_step = 0
        while distill_step < max_distill_step:
            barrier()
            logging.info("Train %s-th shard by trainer %s", \
                index+1, self.rank)
            distill_step = self.train_shard(
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
            logging.info("Saving checkpoint to %s", checkpoint_path)
            model.module.save_gs_checkpoint(checkpoint_path)
        return True

    def eval(self, model, eval_data_mgr, distill_step):
        """ Evaluate student model on validation set.
        The metric are mean square error (MSE).

        Parameters
        ----------
        model : GSDistilledModel
            Distilled student model.
        eval_data_mgr : DistillDataManager
            Data manager for validation data.
        distill_step : int
            Distill step of the model checkpoint.
        """
        model.eval()
        batch_index = 0
        total_mse = 0
        for index, dataset_iterator in enumerate(eval_data_mgr):
            if dataset_iterator is None:
                break
            logging.info("Eval %s-th shard by trainer %s", \
                index+1, self.rank)
            with th.no_grad():
                for _, batch in enumerate(dataset_iterator):
                    mse = model.module(batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                        batch["labels"].to(self.device))
                    total_mse += mse.item()
                    batch_index += 1

        mean_total_mse = total_mse / batch_index
        logging.info("Eval MSE at step %s: %s", distill_step, mean_total_mse)
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
        train_data_mgr : DistillDataManager
            Data manager for training data.
        eval_data_mgr : DistillDataManager
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
        int : Distill step of the model checkpoint.
        """
        dataset_iterator = train_data_mgr.get_iterator()
        if not dataset_iterator:
            raise RuntimeError("No training data. Check training dataset as the data sampler"
                                    "is supposed to be infinite.")

        # TODO (HZ): support prefetch to speed up the training
        model.train()
        shard_loss = 0

        for batch_num, batch in enumerate(dataset_iterator):
            try:
                loss = model(batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["labels"].to(self.device))

                shard_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # TODO (HZ): Need to review how to report loss.
                mean_shard_loss = shard_loss / (batch_num + 1)
                if distill_step % 20 == 0:
                    logging.info(
                        "Loss for shard {train_data_mgr.get_iterator_name()}" \
                        " at step %s = %s" \
                        " from trainer %s", distill_step, mean_shard_loss, \
                        self.rank)

                if distill_step % save_model_frequency == 0 and distill_step != 0:
                    # TODO (HZ): implement save_topk_models based on val MSE scores
                    self.save_student_model(model, saved_path, distill_step)

                if distill_step % eval_frequency == 0 and distill_step != 0:
                    barrier()
                    # TODO (HZ): implement distributed evaluation by communicating with all trainers
                    if self.rank == 0:
                        self.eval(model, eval_data_mgr, distill_step)
                    barrier()

                if distill_step == max_distill_step:
                    break
                distill_step += 1
            except StopIteration:
                continue

        # release the memory
        train_data_mgr.release_iterator()

        return distill_step

    def setup_device(self, device):
        """ Set up the device for the distillation.
        The device is set up based on the local rank.
        Parameters
        ----------
        device :
            The device for distillation.
        """
        self._device = device

    @property
    def device(self):
        """ The device associated with the inference.
        """
        return self._device
