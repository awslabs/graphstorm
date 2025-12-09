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

    Mitra Embedding Generation Tool for GraphStorm

    This tool generates Mitra embeddings from graph data for use in GraphStorm.
    It supports:
    1. MovieLens-100k dataset for User gender classification (with automatic download)
    2. Custom DGL graphs constructed from parquet/CSV files using gconstruct

    Important Limitation
    --------------------
    The current version of Mitra TFM (Tabular Foundation Model) requires the number 
    of classes for classification problems to be no more than 10. Datasets exceeding 
    this limit are not supported and will cause training to fail.

    Installation of Mitra:
        pip install autogluon.tabular[mitra]
    -----------
"""

import os
import sys
import gc
import torch
import tempfile
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import dgl
import urllib.request
import zipfile

# Add parent directory to path to import graphstorm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune
from autogluon.tabular.models.mitra._internal.config.enums import Task, LossName, ModelName
from autogluon.tabular.models.mitra._internal.core.trainer_finetune import CollatorWithPadding

from graphstorm.data.movielens import MovieLens100kNCDataset
from graphstorm.data.dataset import ConstructedGraphDataset


def download_movielens_100k(raw_dir):
    """
    Download and extract MovieLens 100k dataset if not already present.
    
    Parameters
    ----------
    raw_dir : str
        Directory where the dataset should be stored
        
    Returns
    -------
    str
        Path to the directory containing the extracted ml-100k folder
    """
    ml_dir = os.path.join(raw_dir, 'ml-100k')
    
    # Check if already downloaded
    if os.path.exists(os.path.join(ml_dir, 'u.user')):
        print(f"MovieLens 100k dataset already exists at {ml_dir}")
        return raw_dir
    
    # Create directory if it doesn't exist
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download URL
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(raw_dir, 'ml-100k.zip')
    
    print(f"Downloading MovieLens 100k dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        # Extract zip file
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        # Remove zip file
        os.remove(zip_path)
        print(f"MovieLens 100k dataset extracted to {ml_dir}")
        
        return raw_dir
    except Exception as e:
        raise RuntimeError(f"Failed to download MovieLens 100k dataset: {e}")


def save_input_embeddings_hook(name, hidden_embeddings_container):
    """
    Create a hook function that captures intermediate representations from model layers.
    
    Parameters
    ----------
    name : str
        Name identifier for the layer being hooked
    hidden_embeddings_container : dict
        Dictionary to store captured embeddings, keyed by layer name
        
    Returns
    -------
    function
        Hook function that can be registered to a PyTorch module
    """
    def hook_fn(module, inp, out):
        if torch.is_tensor(inp):
            hidden_embeddings_container[name] = inp.detach().cpu()
        elif isinstance(inp, (tuple, list)):
            hidden_embeddings_container[name] = [x.detach().cpu() if torch.is_tensor(x) else x for x in inp]
        elif isinstance(inp, dict):
            hidden_embeddings_container[name] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in inp.items()}
        else:
            hidden_embeddings_container[name] = inp
    return hook_fn


def register_hooks_for_embeddings(model, hidden_embeddings_container, layer_names=None):
    """
    Register forward hooks to capture intermediate representations from model layers.
    
    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to register hooks on
    hidden_embeddings_container : dict
        Dictionary to store captured embeddings from each layer
    layer_names : list of str, optional
        Specific layer names to hook. If None, hooks all named modules
        
    Returns
    -------
    list
        List of registered hook handles that can be removed later
    """
    hooks = []
    
    if layer_names is None:
        for name, module in model.named_modules():
            if name:
                hook = module.register_forward_hook(save_input_embeddings_hook(name, hidden_embeddings_container))
                hooks.append(hook)
    else:
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(save_input_embeddings_hook(name, hidden_embeddings_container))
                hooks.append(hook)
    return hooks


def get_instance_embeddings(fold_predictor, batch_data, save_dir=None) -> tuple:
    """
    Generate embeddings from a trained Mitra predictor for input data.
    
    This function processes data through a trained Mitra model and extracts
    embeddings from the final layer normalization. It handles large datasets
    by saving intermediate results to disk.
    
    Parameters
    ----------
    fold_predictor : TabularPredictor
        Trained AutoGluon TabularPredictor with Mitra model
    batch_data : pd.DataFrame or np.ndarray
        Input data to generate embeddings for
    save_dir : str, optional
        Directory to save intermediate embedding files. If None, uses a temporary directory
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - embeddings: Instance embeddings from the model [N, embedding_dim]
        - avg_embeddings: Average embeddings across features [N, embedding_dim]
        Returns empty arrays if no embeddings are generated
        
    Notes
    -----
    - Embeddings are saved in chunks to manage memory for large datasets
    - Temporary files are automatically cleaned up after merging
    - Uses GPU if available for faster processing
    """
    if fold_predictor.transform_features is not None:
        batch_data = fold_predictor._learner.transform_features(batch_data)
    fold_predictor_trainer = fold_predictor._learner.load_trainer()
    ensemble_model = fold_predictor_trainer.load_model("Mitra")
    batch_data = ensemble_model.preprocess(batch_data)
    model = ensemble_model.model
        
    assert len(model.trainers) == 1, "the number of trainer should be one instead of a ensembled one"
    trainer = model.trainers[0]
    if isinstance(batch_data, pd.DataFrame):
        batch_data = batch_data.values

    self = trainer
    x_support, y_support, x_query = model.X, model.y, batch_data
    x_support_transformed = self.preprocessor.transform_X(x_support)
    y_support_transformed = self.preprocessor.transform_y(y_support)
    x_query_transformed = self.preprocessor.transform_X(x_query)

    dataset = DatasetFinetune(                                                             
        self.cfg,                                                                          
        x_support=x_support_transformed,
        y_support=y_support_transformed,                      
        x_query=x_query_transformed,
        y_query=None,
        max_samples_support=self.cfg.hyperparams['max_samples_support'],
        max_samples_query=self.cfg.hyperparams['max_samples_query'],
        rng=self.rng,
    )                                                        

    if self.cfg.model_name == ModelName.TABPFN:
        pad_to_max_features = True
    elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
        pad_to_max_features = False
    else:
        raise NotImplementedError(f"Model {self.cfg.model_name} not implemented")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
        collate_fn=CollatorWithPadding(
            max_features=self.cfg.hyperparams['dim_embedding'],
            pad_to_max_features=pad_to_max_features
        ),
    )
    self.model.eval()

    if save_dir is None:
        save_dir = tempfile.mkdtemp()
    
    os.makedirs(save_dir, exist_ok=True)
    
    embeddings = []
    avg_embeddings = []
    save_interval = 1024 * 10
    file_counter = 0
    temp_files = []
    
    with torch.no_grad():                                                                  
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches for mitra embeddings")):
            cached_hidden_embeddings = {}
            hooks = register_hooks_for_embeddings(self.model, cached_hidden_embeddings, ['final_layer_norm'])
            
            try:
                with torch.autocast(device_type=self.device, dtype=getattr(torch, self.cfg.hyperparams['precision'])):
                    x_s = batch['x_support'].to(self.device, non_blocking=True)
                    y_s = batch['y_support'].to(self.device, non_blocking=True)
                    x_q = batch['x_query'].to(self.device, non_blocking=True)
                    padding_features = batch['padding_features'].to(self.device, non_blocking=True)
                    padding_obs_support = batch['padding_obs_support'].to(self.device, non_blocking=True)
                    padding_obs_query = batch['padding_obs_query'].to(self.device, non_blocking=True)
                    
                    if self.cfg.task == Task.REGRESSION and self.cfg.hyperparams['regression_loss'] == LossName.CROSS_ENTROPY:
                        y_s = torch.bucketize(y_s, self.bins) - 1
                        y_s = torch.clamp(y_s, 0, self.cfg.hyperparams['dim_output']-1).to(torch.int64)

                    if self.cfg.model_name == ModelName.TABPFN:
                        _ = self.model(x_s, y_s, x_q, task=self.cfg.task).squeeze(-1)
                    elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
                        _ = self.model(x_s, y_s, x_q, padding_features, padding_obs_support, padding_obs_query)
                
                if 'final_layer_norm' in cached_hidden_embeddings.keys():
                    embedding_slice = cached_hidden_embeddings['final_layer_norm'][0][0, :, 0].detach().cpu().numpy()
                    avg_embedding_slice = cached_hidden_embeddings['final_layer_norm'][0][0, :, 1:].detach().mean(dim=-1).cpu().numpy()
                    
                    embeddings.append(embedding_slice.astype(np.float32))
                    avg_embeddings.append(avg_embedding_slice.astype(np.float32))
                    
                    if len(embeddings) >= save_interval:
                        temp_emb_file = os.path.join(save_dir, f'embeddings_part_{file_counter}.npy')
                        temp_avg_file = os.path.join(save_dir, f'avg_embeddings_part_{file_counter}.npy')
                        
                        try:
                            np.save(temp_emb_file, np.concatenate(embeddings, axis=0))
                            np.save(temp_avg_file, np.concatenate(avg_embeddings, axis=0))
                        except ValueError:
                            np.save(temp_emb_file, np.vstack(embeddings))
                            np.save(temp_avg_file, np.vstack(avg_embeddings))
                        
                        temp_files.append((temp_emb_file, temp_avg_file))
                        file_counter += 1
                        embeddings = []
                        avg_embeddings = []
                        gc.collect()
                    
                    del cached_hidden_embeddings['final_layer_norm']
                
            finally:
                for hook in hooks:
                    hook.remove()
                cached_hidden_embeddings.clear()
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if len(embeddings) > 0:
        temp_emb_file = os.path.join(save_dir, f'embeddings_part_{file_counter}.npy')
        temp_avg_file = os.path.join(save_dir, f'avg_embeddings_part_{file_counter}.npy')
        
        try:
            np.save(temp_emb_file, np.concatenate(embeddings, axis=0))
            np.save(temp_avg_file, np.concatenate(avg_embeddings, axis=0))
        except ValueError:
            np.save(temp_emb_file, np.vstack(embeddings))
            np.save(temp_avg_file, np.vstack(avg_embeddings))
        
        temp_files.append((temp_emb_file, temp_avg_file))
        embeddings = []
        avg_embeddings = []
        gc.collect()
    
    if temp_files:
        all_embeddings = []
        all_avg_embeddings = []
        
        for emb_file, avg_file in temp_files:
            emb_data = np.load(emb_file)
            avg_data = np.load(avg_file)
            all_embeddings.append(emb_data)
            all_avg_embeddings.append(avg_data)
            
            try:
                os.remove(emb_file)
                os.remove(avg_file)
            except:
                pass
        
        try:
            embeddings_result = np.concatenate(all_embeddings, axis=0)
            avg_embeddings_result = np.concatenate(all_avg_embeddings, axis=0)
        except ValueError:
            embeddings_result = np.vstack(all_embeddings)
            avg_embeddings_result = np.vstack(all_avg_embeddings)
        
        del all_embeddings, all_avg_embeddings
        gc.collect()
        
        return embeddings_result, avg_embeddings_result
    else:
        return np.array([]), np.array([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tools to generate Mitra embeddings from GraphStorm datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # MovieLens dataset - user gender classification
            python gen_mitra_embedding.py --dataset movie-len --dataset_path data/ml-100k \
                    --savedir output --target-ntype user --label-name gender
            
            # Custom DGL graph constructed from parquet files using gconstruct
            python gen_mitra_embedding.py --dataset my-graph --dataset_path data/constructed --savedir output
            
            # For parquet input, first use gconstruct to create a DGL graph:
            python -m graphstorm.gconstruct.construct_graph --conf-file config.json --output-dir data/constructed
            # Then use this tool to generate embeddings from the constructed graph
        """
    )
    parser.add_argument("--savedir", type=str, required=True, 
                       help="Path to the directory to save embeddings")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", 
                       help="Dataset name: 'movie-len' for MovieLens or custom name for DGL graphs")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset: MovieLens raw data dir or DGL graph directory")
    parser.add_argument("--target-ntype", type=str, default="_N",     
                       help="Target node type (for heterogeneous graphs)")
    parser.add_argument("--feat-name", type=str, default="feat",   
                       help="Feature name in graph node data")
    parser.add_argument("--label-name", type=str, default="labels", 
                       help="Label name in graph node data")
    args = parser.parse_args()

    # Load graph data
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'movie-len':
        # Load MovieLens dataset without text features
        print(f"Loading MovieLens dataset from: {args.dataset_path}")
        # Download MovieLens if not present
        dataset_path = download_movielens_100k(args.dataset_path)
        # Load dataset for gender classification
        print("Setting up for user gender classification...")
        dataset = MovieLens100kNCDataset(dataset_path, use_text_feat=False)
        g = dataset[0]
        
        # Extract individual features from user nodes
        user_feat = g.nodes['user'].data['feat']
        g.nodes['user'].data['age']         = user_feat[:, 0:1]  
        g.nodes['user'].data['gender']      = user_feat[:, 1:2]
        g.nodes['user'].data['occupation']   = user_feat[:, 2:]
        
        # Use age and occupation as features (gender is now the label)
        g.nodes['user'].data['feat'] = torch.cat(
            [g.nodes['user'].data['age'], g.nodes['user'].data['occupation']], dim=1)
        print(f"Gender classification setup: "
            f"{g.nodes['user'].data['feat'].shape[0]} users, "
            f"features shape: {g.nodes['user'].data['feat'].shape}"
        )
    else:
        # Load custom DGL graph constructed from parquet/csv files using gconstruct
        # The graph should be saved as <dataset_name>.dgl in dataset_path
        print(f"Loading constructed DGL graph: {args.dataset}")
        print(f"Graph directory: {args.dataset_path}")
        dataset = ConstructedGraphDataset(args.dataset, args.dataset_path)
        g = dataset[0]
    
    print(f"Graph loaded successfully: {g}")
    
    # Get features and labels from the DGL graph
    # Handle both homogeneous and heterogeneous graphs
    if len(g.ntypes) == 1:
        # Homogeneous graph
        target_ntype = g.ntypes[0]
        print(f"Homogeneous graph detected, using node type: {target_ntype}")
    else:
        # Heterogeneous graph
        target_ntype = args.target_ntype
        if target_ntype not in g.ntypes:
            print(f"Target node type '{target_ntype}' not found. Searching for valid node type...")
            for ntype in g.ntypes:
                if args.feat_name in g.nodes[ntype].data and args.label_name in g.nodes[ntype].data:
                    target_ntype = ntype
                    print(f"Using node type: {target_ntype}")
                    break
            else:
                raise ValueError(f"No node type found with '{args.feat_name}' and '{args.label_name}'. "
                               f"Available node types: {g.ntypes}")
    
    # Verify features and labels exist
    if args.feat_name not in g.nodes[target_ntype].data:
        raise ValueError(f"Node type '{target_ntype}' does not have '{args.feat_name}' in data. "
                        f"Available keys: {list(g.nodes[target_ntype].data.keys())}")
    if args.label_name not in g.nodes[target_ntype].data:
        raise ValueError(f"Node type '{target_ntype}' does not have '{args.label_name}' in data. "
                        f"Available keys: {list(g.nodes[target_ntype].data.keys())}")
    
    target_features = g.nodes[target_ntype].data[args.feat_name]
    target_labels   = g.nodes[target_ntype].data[args.label_name]
    print(f"Using node type '{target_ntype}' with {target_features.shape[0]} nodes")
    
    # Create DataFrame for Mitra
    print(f"Creating DataFrame with {target_features.shape[0]} samples and {target_features.shape[1]} features")
    table = pd.DataFrame(target_features.cpu().numpy())
    
    # Ensure labels are 0-indexed for Mitra
    labels_np = target_labels.cpu().numpy().flatten()
    unique_labels = np.unique(labels_np)
    print(f"Original label range: {labels_np.min()} to {labels_np.max()}")
    
    if labels_np.min() != 0 or not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        print("Remapping labels to 0-indexed...")
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels_np = np.array([label_map[label] for label in labels_np])
    
    table['y'] = labels_np.astype(int)
    print(f"Final label range: {labels_np.min()} to {labels_np.max()}")
    
    # Get number of unique classes
    num_classes = len(table['y'].unique())
    print(f"Number of classes: {num_classes}")
    
    # Check if number of classes exceeds Mitra's limitation
    if num_classes > 10:
        print(f"\n{'='*70}")
        print(f"ERROR: Dataset has {num_classes} classes")
        print(f"{'='*70}")
        print(f"\nThe current version of Mitra TFM (Tabular Foundation Model) supports")
        print(f"a maximum of 10 classes for multiclass classification problems.")
        print(f"\nYour dataset has {num_classes} classes, which exceeds this limitation.")
        print(f"\nPossible solutions:")
        print(f"  1. Use a different embedding method (e.g., traditional GNN encoders)")
        print(f"  2. Reduce the number of classes through label grouping/merging")
        print(f"  3. Convert to a binary or regression task if applicable")
        print(f"  4. Wait for future versions of AutoGluon with expanded class support")
        print(f"\n{'='*70}")
        sys.exit(1)
    
    # Initialize and fit Mitra predictor
    print("Initializing Mitra predictor...")
    mitra_predictor = TabularPredictor(label='y', problem_type='multiclass')
    print("Fitting Mitra model...")
    
    mitra_predictor.fit(
        table,
        hyperparameters={
            'MITRA': {
                'fine_tune': False, 
                'ag.max_memory_usage_ratio': 1000000, 
                'ag.max_rows': None,
            },
        },
    )
    
    # Generate embeddings
    print("Generating Mitra embeddings...")
    mitra_embeddings, _ = get_instance_embeddings(mitra_predictor, table, save_dir=args.savedir)
    
    # Save embeddings
    output_path = os.path.join(args.savedir, "mitra_embeddings.pt")
    print(f"Saving embeddings to {output_path}")
    torch.save(torch.from_numpy(mitra_embeddings), output_path)
    print("Done!")
