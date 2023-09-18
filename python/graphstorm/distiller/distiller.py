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

from ..model.gnn_distill import GSDistilledModel
from ..dataloading import DataloaderGenerator, DataManager

class GSdistiller():
    """ GNN distiller.

    Parameters
    ----------
    rank : int
        The rank.
        Configs for GNN distillation
    """
    def __init__(self, rank):
        self._rank = rank
        self._device = -1

    def distill(
        self, 
        lm_name, 
        pre_trained_name,
        data_dir,
        batch_size,
        max_seq_len,
        distill_lr,
        saved_path,
        save_model_frequency,
        eval_frequency,
        max_global_step,
        on_cpu=False,
    ):
        """ Distill function.
        Each worker initializes student model, data provider,
        and start training.

        Parameters
        ----------
        lm_name : str
            Model name for Transformer-based student model.
        pre_trained_name : str
            Name for pre-trained model weights.
        gnn_embed_dim : int
            Size of GNN embeddings.
        data_dir : str
            Directory for distillation data.
        batch_size : int
            Batch size for distillation.
        max_seq_len : int
            Maximum sequence length.
        distill_lr : float
            Learning rate for distillation.
        saved_path : str
            Path to save the model.
        save_model_frequency : int,
            Interval to save the model checkpoint.
        eval_frequency : int,
            Interval for evaluation.
        max_global_step : int,
            Maximum steps for distillation training.
        on_cpu : bool
            Whether the distillation will be conducted on cpu.
        """
        self.student = GSDistilledModel(lm_name=lm_name, pre_trained_name=pre_trained_name)
        dataloader_generator = DataloaderGenerator(tokenizer=self.student.tokenizer, 
            max_seq_len = max_seq_len,
            device=self.device, 
            batch_size=batch_size,
        )
        train_data_provider = DataManager(
            dataloader_generator,
            dataset_path=os.path.join(data_dir, 'train'),
            local_rank=self.rank,
            world_size=th.distributed.get_world_size(),
            is_train=True,
        )
        eval_data_provider = DataManager(
            dataloader_generator,
            dataset_path=os.path.join(data_dir, 'val'),
            local_rank=self.rank,
            world_size=th.distributed.get_world_size(),
            is_train=False,
        )

        # get GNN embed dim
        dataset_iterator = eval_data_provider.get_iterator()
        if not dataset_iterator:
            raise RuntimeError("No validation data") 
        batch = next(iter(dataset_iterator))
        gnn_embed_dim = batch["labels"].shape[1]
        self.student.init_proj_layer(gnn_embed_dim=gnn_embed_dim)

        # TODO (HZ): add flexibility to specify different optimizers
        optimizer = th.optim.Adam(self.student.parameters(), lr=distill_lr)
        self.student.to(self.device)
        student = DistributedDataParallel(self.student, device_ids=None if on_cpu else [self.device],
                                        output_device=None if on_cpu else self.device)

        index = 0
        global_step = 0
        while True:
            th.distributed.barrier()
            print (f"Train {index + 1}-th shard by trainer {self.rank}")
            complete, global_step = self.train_shard(
                global_step=global_step,
                model=student, 
                optimizer=optimizer, 
                train_dataset_provider=train_data_provider, 
                eval_dataset_provider=eval_data_provider,
                saved_path=saved_path,
                save_model_frequency=save_model_frequency,
                eval_frequency=eval_frequency,
                max_global_step=max_global_step,
            )
            is_first_iteration = False

            index += 1
            if complete:
                break

    def save_student_model(self, model, saved_path, global_step):
        """ Save student model

        Parameters
        ----------
        model : GSDistilledModel
            Distilled student model.
        saved_path : str
            Path to save the model.
        global_step : int
            Global step of the model checkpoint.
        """
        th.distributed.barrier()
        if self.rank == 0:
            checkpoint_path = os.path.join(saved_path, f"checkpoint-{global_step}")
            proj_dir_loc = os.path.join(checkpoint_path, "proj")
            tokenizer_dir_loc = os.path.join(checkpoint_path, "tokenizer")
            lm_dir_loc = os.path.join(checkpoint_path, "lm")
            os.makedirs(proj_dir_loc, exist_ok=True)
            print (f"Saving checkpoint to {checkpoint_path}")

            # TODO (HZ): need to test if the saved model can be successfully loaded
            model.module.tokenizer.save_pretrained(tokenizer_dir_loc)
            model.module.lm.save_pretrained(lm_dir_loc)
            th.save(model.module.state_dict()["proj"], os.path.join(proj_dir_loc, "pytorch_model.bin"))

        return True

    def to_device(self, inputs, device='cuda'):
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
                outputs[k] = self.to_device(v, device=device)
        elif isinstance(inputs, (list, tuple)):
            outputs = []
            for v in inputs:
                outputs.append(self.to_device(v, device=device))
        else:
            raise NotImplementedError
        return outputs

    def eval(self, model, eval_dataset_provider, global_step):
        """ Evaluate student model on validation set.
        The metric are mean square error (MSE).

        Parameters
        ----------
        model : GSDistilledModel
            Distilled student model.
        eval_dataset_provider : DataProvider
            Data provider for validation data.
        global_step : int
            Global step of the model checkpoint.
        """
        model.eval()
        index = 0
        batch_index = 0
        total_loss = 0
        while True:
            dataset_iterator = eval_dataset_provider.get_iterator()
            if dataset_iterator is None:
                break
            print (f"Eval {index + 1}-th shard by trainer {self.rank}")
            with th.no_grad():
                for batch_num, batch in enumerate(dataset_iterator):
                    batch = self.to_device(batch, self.device)  # Move to device
                    loss = model.module(batch["input_ids"],
                        batch["attention_mask"],
                        batch["labels"])
                    total_loss += loss.item()
                    batch_index += 1
            index += 1

        mean_total_loss = total_loss / batch_index
        print (f"Eval MSE at step {global_step}: {mean_total_loss}")
        model.train()
        eval_dataset_provider.release_iterator()

    def train_shard(
        self, 
        global_step, 
        model, 
        optimizer, 
        train_dataset_provider, 
        eval_dataset_provider, 
        saved_path,
        save_model_frequency,
        eval_frequency,
        max_global_step,
    ):
        """
        Train using one shard from train_dataset_provider.
        Parameters
        ----------
        global_step : int
            Global step of the model checkpoint.
        model : GSDistilledModel
            Distilled student model.
        optimizer : torch optimizer
            optimizer for distillation.
        train_dataset_provider : DataProvider
            Data provider for training data.
        eval_dataset_provider : DataProvider
            Data provider for validation data.
        saved_path : str
            Path to save the model.
        save_model_frequency : int,
            Interval to save the model checkpoint.
        eval_frequency : int,
            Interval for evaluation.
        max_global_step : int,
            Maximum steps for distillation training.
    
        Returns
        -------
        bool : whether to stop distillation.
        int : Global step of the model checkpoint.
        """
        dataset_iterator = train_dataset_provider.get_iterator()
        if not dataset_iterator:
            raise RuntimeError("No training data. Check training dataset as the data sampler"
                                    "is supposed to be infinite.")

        # TODO (HZ): support prefetch to speed up the training
        model.train()
        shard_loss = 0
        complete = False

        for batch_num, batch in enumerate(dataset_iterator):
            try:
                batch = self.to_device(batch, self.device)  # Move to device

                loss = model(batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"])

                shard_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_shard_loss = shard_loss / (batch_num + 1)
                if global_step % 20 == 0:
                    print (
                        f"Loss for shard {train_dataset_provider.get_iterator_name()}"
                        f" at step {global_step} = {mean_shard_loss}"
                        f" from trainer {self.rank}"
                    )

                if global_step % save_model_frequency == 0 and global_step != 0:
                    self.save_student_model(model, saved_path, global_step)

                if global_step % eval_frequency == 0 and global_step != 0:
                    th.distributed.barrier()
                    # TODO (HZ): implement distributed evaluation by communicating with all trainers
                    if self.rank == 0:
                        self.eval(model, eval_dataset_provider, global_step)
                    th.distributed.barrier()

                if global_step == max_global_step:
                    complete = True
                    break
                global_step += 1
            except StopIteration:
                continue

        # release the memory
        train_dataset_provider.release_iterator()

        return complete, global_step

    def setup_device(self, device):
        """ Set up the device for the distillation.

        The CUDA device is set up based on the local rank.

        Parameters
        ----------
        device :
            The device for distillation.
        """
        self._device = th.device(device)

    @property
    def rank(self):
        """ Get the rank for the distillation.
        """
        return self._rank

    @property
    def device(self):
        """ The device associated with the inference.
        """
        return self._device