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

    This tool generates Mitra embeddings from custom datasets in parquet format
    to be used in GraphStorm.

    Expected Input Directory Structure
    -----------------------------------
    The tool expects parquet files organized by node types:
    
    dataset_path/
    └── target_ntype/              # Node type directory (e.g., 'user', 'product', 'movie')
        ├── data.parquet           # Single parquet file, OR
        ├── part-00000.parquet     # Multiple parquet files
        ├── part-00001.parquet
        └── ...
    
    Example structures:
    
 Single node type:
       data/
      user/
           └── users.parquet
    
    2. Multiple node types:
       data/
       ├── user/
       │   └── users.parquet
       └── product/
           └── products.parquet
    
    3. Partitioned data:
       data/
       └── target_ntype/
           ├── part-00000.parquet
           ├── part-00001.parquet
           └── part-00002.parquet
    
    Parquet File Requirements
    -------------------------
    Each parquet file must contain:
    - Feature columns: Numeric columns used for embedding generation
    - Label column: Target column for classification (specified by --label-name)
    - Node ID column: Unique identifier for each node (specified by --node-id-col)
      If not present, sequential IDs will be auto-generated
    
    Output Structure
    ----------------
    The tool generates embeddings in the following structure:
    
    data/
    └── mitra_emb/
        └── mitra_embeddings.parquet
    
    The output parquet contains:
    - node_id: Node identifiers
    - label_column: Original label values
    - 0, 1, 2, ...: Embedding dimensions (512-dimensional by default)

    Limitation:
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
from pathlib import Path

# Add parent directory to path to import graphstorm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune
from autogluon.tabular.models.mitra._internal.config.enums import Task, LossName, ModelName
from autogluon.tabular.models.mitra._internal.core.trainer_finetune import CollatorWithPadding


def load_parquet_data(data_path, feature_cols=None, label_col='label', node_id_col='node_id'):
    """
    Load custom data from parquet directory.
    
    Parameters
    ----------
    data_path : str
        Path to directory containing parquet files
    feature_cols : list of str, optional
        List of column names of features. If None, uses all columns except label_col and node_id_col
    label_col : str
        Column name to use as label
    node_id_col : str
        Column name for node IDs. If not present, sequential IDs will be created
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        DataFrame with features and label column 'y', and Series with node_ids
    """
    data_path = Path(data_path)
    
    # Only support directory input
    if not data_path.is_dir():
        raise ValueError(f"data_path must be a directory: {data_path}")
    
    # Load all parquet files in directory (exclude mitra_embeddings.parquet to avoid loading previous outputs)
    parquet_files = sorted([f for f in data_path.glob("*.parquet") if f.name != "mitra_embeddings.parquet"])
    if not parquet_files:
        parquet_files = sorted([f for f in data_path.glob("part-*.parquet") if f.name != "mitra_embeddings.parquet"])
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path} (excluding mitra_embeddings.parquet)")
    
    print(f"Loading data from {len(parquet_files)} parquet files in {data_path}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Extract or create node IDs
    if node_id_col in df.columns:
        node_ids = df[node_id_col].copy()
        print(f"Using existing node ID column: {node_id_col}")
    else:
        node_ids = pd.Series(range(len(df)), name='node_id')
        print(f"Node ID column '{node_id_col}' not found, creating sequential IDs")
    
    # Check if label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data. Available columns: {list(df.columns)}")
    
    # Select features
    if feature_cols is None:
        # Use all columns except label and node_id as features
        feature_cols = [col for col in df.columns if col not in [label_col, node_id_col]]
    else:
        # Verify all feature columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
    
    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")
    
    # Create features dataframe
    features_df = df[feature_cols].copy()
    features_df['y'] = df[label_col]
    
    return features_df, node_ids


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
        description='Tools to generate Mitra embeddings from custom parquet data directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Expected Directory Structure:
            dataset_path/
            └── target_ntype/
                └── *.parquet files

        Examples:
            # Auto-detect all feature columns excluding the mitra_embedding.parquet
            # Reads from: data/target_ntype/*.parquet
            # Writes to:  output/target_ntype/mitra_embeddings.parquet
            python tools/gen_mitra_embedding.py \\
                --dataset_path data \\
                --target-ntype target_ntype \\
                --label-name target_label
                --node-id-col node_id
                """
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Base path to dataset directory. Tool will look for parquet files in dataset_path/target-ntype/")
    parser.add_argument("--target-ntype", type=str, required=True,
                       help="Target node type. Tool will read and write in dataset_path/target-ntype/")
    parser.add_argument("--label-name", type=str, required=True, 
                       help="Label column name in parquet files (used for classification task)")
    parser.add_argument("--feature-cols", type=str, default=None,
                       help="Comma-separated list of feature column names. If not specified, uses all columns except label and node_id")
    parser.add_argument("--node-id-col", type=str, default="node_id",
                       help="Node ID column name in parquet files. If not present, sequential IDs will be created (default: 'node_id')")
    args = parser.parse_args()

    # Construct path: dataset_path/target-ntype/
    data_path_with_ntype = os.path.join(args.dataset_path, args.target_ntype)
    print(f"{'='*70}")
    print(f"Mitra Embedding Generation")
    print(f"{'='*70}")
    print(f"Input directory:  {data_path_with_ntype}")
    print(f"Output directory: {data_path_with_ntype}")
    print(f"Node type:        {args.target_ntype}")
    print(f"Label column:     {args.label_name}")
    print(f"Node ID column:   {args.node_id_col}")
    print(f"{'='*70}\n")
    
    # Verify input directory exists
    if not os.path.exists(data_path_with_ntype):
        print(f"ERROR: Input directory does not exist: {data_path_with_ntype}")
        print(f"\nExpected structure:")
        print(f"  {args.dataset_path}/")
        print(f"  └── {args.target_ntype}/")
        print(f"      └── *.parquet files")
        sys.exit(1)
    
    print(f"Loading custom data from: {data_path_with_ntype}")
    feature_cols = None
    if args.feature_cols:
        feature_cols = [col.strip() for col in args.feature_cols.split(',')]
    
    table, node_ids = load_parquet_data(
        data_path_with_ntype, feature_cols=feature_cols, 
        label_col=args.label_name, node_id_col=args.node_id_col
    )
    
    print(f"  Data loaded successfully:")
    print(f"  Total samples: {len(table)}")
    print(f"  Feature columns: {len(table.columns) - 1}")
    print(f"  Label column: 'y'")
    
    # Ensure labels are 0-indexed for Mitra
    labels_np = table['y'].values
    unique_labels = np.unique(labels_np)
    if labels_np.min() != 0 or not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        print("  Remapping labels to 0-indexed...")
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        table['y'] = table['y'].map(label_map)
        labels_np = table['y'].values
        print(f"  Remapped label range: {labels_np.min()} to {labels_np.max()}")
    
    # Get number of unique classes
    num_classes = len(table['y'].unique())
    print(f"  Number of classes: {num_classes}")
    
    # Check if number of classes exceeds Mitra's limitation
    if num_classes > 10:
        print(f"{'='*70}")
        print(f"ERROR: Dataset has {num_classes} classes")
        print(f"{'='*70}")
        print(f"The current version of Mitra TFM (Tabular Foundation Model) supports")
        print(f"a maximum of 10 classes for multiclass classification problems.")
        print(f"Your dataset has {num_classes} classes, which exceeds this limitation.")
        print(f"Possible solutions:")
        print(f"  1. Use a different embedding method (e.g., traditional GNN encoders)")
        print(f"  2. Reduce the number of classes through label grouping/merging")
        print(f"  3. Convert to a binary or regression task if applicable")
        print(f"  4. Wait for future versions of AutoGluon with expanded class support")
        print(f"{'='*70}")
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
    mitra_embeddings, _ = get_instance_embeddings(mitra_predictor, table, save_dir=data_path_with_ntype)
    
    # Save embeddings as parquet with node IDs
    # Create subdirectory for node type
    output_dir = os.path.join(data_path_with_ntype, args.target_ntype)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine target column name
    target_col_name = args.label_name
    
    # Create DataFrame with node_id, target label, and embeddings
    embedding_df = pd.DataFrame(mitra_embeddings)
    embedding_df.insert(0, 'node_id', node_ids.values)
    embedding_df.insert(1, target_col_name, table['y'].values)
    
    # Save as parquet
    output_path = os.path.join(output_dir, "mitra_embeddings.parquet")
    print(f"\n{'='*70}")
    print(f"Saving embeddings to: {output_path}")
    print(f"{'='*70}")
    print(f"  Node type:        {args.target_ntype}")
    print(f"  Embedding shape:  {mitra_embeddings.shape}")
    print(f"  Number of nodes:  {len(node_ids)}")
    print(f"  Target column:    {target_col_name}")
    embedding_df.to_parquet(output_path, index=False)
    print(f"{'='*70}")
    print(f"SUCCESS: Embeddings saved successfully!")
    print(f"{'='*70}")
