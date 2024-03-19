from typing import Any, Dict, Optional

import graphstorm as gs
import graphstorm_lightning as gsl
import pytorch_lightning as pl
import torch
import contextlib


class GSgnnNodeTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        trainer: pl.Trainer,
        config: Dict[str, Any],
        graph_data_uri: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.trainer = trainer

    def _device(self) -> torch.device:
        return self.trainer.strategy.root_device

    def prepare_data(self) -> None:
        gsl.utils.load_data(self.trainer, self.hparams.config, self.hparams.graph_data_uri)

    def setup(self, stage: str) -> None:
        ip_config = gsl.utils.initialize_dgl(self.trainer, self.hparams.config)
        with ip_config or contextlib.nullcontext():
            if ip_config:
                self.hparams.config["gsf"]["basic"]["ip_config"] = ip_config.name
            self.config = config = gsl.utils.get_config(self.trainer, self.hparams.config)
            self.gnn = gs.dataloading.GSgnnNodeTrainData(
                config.graph_name,
                config.part_config,
                train_ntypes=config.target_ntype,
                eval_ntypes=config.eval_target_ntype,
                node_feat_field=config.node_feat_name,
                label_field=config.label_field,
                lm_feat_ntypes=gs.utils.get_lm_ntypes(config.node_lm_configs),
            )

    def _dataloader(self, target_idxs: Dict[str, torch.Tensor]) -> gs.dataloading.GSgnnNodeDataLoader:
        return gs.dataloading.GSgnnNodeDataLoader(
            self.gnn,
            target_idxs,
            fanout=self.hparams.fanout,
            batch_size=self.hparams.batch_size,
            device=self._device(),
            train_task=self.hparams.train_task,
            construct_feat_ntype=self.hparams.construct_feat_ntype,
            construct_feat_fanout=self.hparams.construct_feat_fanout,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int):
        data = self.gnn

        # from source
        input_nodes, seeds, blocks = batch
        if not isinstance(input_nodes, dict):
            assert len(data.g.ntypes) == 1
            input_nodes = {data.g.ntypes[0]: input_nodes}
        # make sure input_nodes and seeds are on CPU, since they get converted to NumPy by DGL internally
        input_feats = data.get_node_feats(input_nodes, device)
        lbl = data.get_labels(seeds, device)
        blocks = pl.utilities.move_data_to_device(blocks, device)
        return (input_nodes, lbl, blocks, input_feats)

    def train_dataloader(self) -> gs.dataloading.GSgnnNodeDataLoader:  # type: ignore
        train_data = self.gnn
        config = self.config
        # from https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/run/gsgnn_np/gsgnn_np.py
        dataloader = None
        if config.use_pseudolabel:
            # Use nodes not in train_idxs as unlabeled node sets
            unlabeled_idxs = train_data.get_unlabeled_idxs()
            # semi-supervised loader
            dataloader = gs.dataloading.GSgnnNodeSemiSupDataLoader(
                train_data,
                train_data.train_idxs,
                unlabeled_idxs,
                fanout=config.fanout,
                batch_size=config.batch_size,
                device=self._device(),
                train_task=True,
                construct_feat_ntype=config.construct_feat_ntype,
                construct_feat_fanout=config.construct_feat_fanout,
            )
        else:
            dataloader = gs.dataloading.GSgnnNodeDataLoader(
                train_data,
                train_data.train_idxs,
                fanout=config.fanout,
                batch_size=config.batch_size,
                device=self._device(),
                train_task=True,
                construct_feat_ntype=config.construct_feat_ntype,
                construct_feat_fanout=config.construct_feat_fanout,
            )
        return dataloader

    def val_dataloader(self) -> Optional[gs.dataloading.GSgnnNodeDataLoader]:  # type: ignore
        train_data = self.gnn
        config = self.config
        # from https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/run/gsgnn_np/gsgnn_np.py
        test_dataloader = None
        if len(train_data.test_idxs) > 0:
            fanout = config.eval_fanout if config.use_mini_batch_infer else []
            test_dataloader = gs.dataloading.GSgnnNodeDataLoader(
                train_data,
                train_data.test_idxs,
                fanout=fanout,
                batch_size=config.eval_batch_size,
                device=self._device(),
                train_task=False,
                construct_feat_ntype=config.construct_feat_ntype,
                construct_feat_fanout=config.construct_feat_fanout,
            )
        return test_dataloader

    def test_dataloader(self) -> Optional[gs.dataloading.GSgnnNodeDataLoader]:  # type: ignore
        train_data = self.gnn
        config = self.config
        # from https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/run/gsgnn_np/gsgnn_np.py
        test_dataloader = None
        if len(train_data.test_idxs) > 0:
            # we don't need fanout for full-graph inference
            fanout = config.eval_fanout if config.use_mini_batch_infer else []
            test_dataloader = gs.dataloading.GSgnnNodeDataLoader(
                train_data,
                train_data.test_idxs,
                fanout=fanout,
                batch_size=config.eval_batch_size,
                device=self._device(),
                train_task=False,
                construct_feat_ntype=config.construct_feat_ntype,
                construct_feat_fanout=config.construct_feat_fanout,
            )
        return test_dataloader
