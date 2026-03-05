# --- This MUST be at the top ---
import sys, os
from pathlib import Path

# Path to the script itself
script_path = Path(__file__).resolve()

# Path to .../QTransformer
project_dir = script_path.parent.parent.parent

print(f"Adding project directory to sys.path: {project_dir}")
# Add BOTH directories to sys.path
sys.path.append(str(project_dir))  # Adds /home/carlosR/QTransformer

from mi_quantum.mkexperiments import make_experiment_transformer
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from benri.data import aggregate_and_save_top_configs


if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
        'repeat_selector'       : False,         # True to train autoencoder each time for more variability
        'send_telegram'         : True,
        'num_experiments'       : 5,
        'num_classes'           : 7,
        'trained_selector_once' : False,
        'pixels'                : 64,
        'square'                : False,
        'experiment_name'       : '64PixelsSOAP_QPreprocess',
        'experiment_id'         : 'transformer_results/ExperimentsForPaper/64PixelsSOAP_QPreprocess',
        'variant'               : 'transformer',
        'B'                     : 256,
        'special_batch_for_data': False,
        'rewind_channels'       : False,
        'N'                     : 125,
        'N2'                    : 30,
        'q_config'              : {'none', 'quantum'},
        'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'second_at_a_time'      : False,
        'augmenting'            : True,
        'concatenate_original'  : False,
        'verbose'               : True,
        'optimizer'             : 'SOAP',
        
    }

    w = torch.pi/3
    p_base = {
        'learning_rate': 0.0025, 'hidden_size': 384, 'dropout': 0.285,
        'quantum' : False, 'num_head': 16, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 7, 'patch_size': 8, 'weight_decay': 1e-7, 'attention_selection': 'filter', 
        'selection_amount': 20, 'RD': 1, 'connectivity' : 'chain' ,'entangle_method' : 'edges', 'special_cls' : 'none', 'parallel': 1, 'patience': -1, 
        'scheduler_factor': 0.985, 'q_stride': 1, 'ancilla' : 1, 'channels_out' :  [ [0], [1], [2], [3], [4], [5], [6], [7], [8] ], 'augmentation_prob' : 1, 'val_train_pond' : 1,
        'flatten_extra_channels' : False, 'quanv_kernel_size' : 3,
        'stride' : 1,
        'channels_out' : [1],
        'ancilla': 0,
        'graphs' : {'edges': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [3, 4], [6, 4]], 'weights': [w,w,w,w,w,w,w,w,w,w]},
        'entangle_method' : 'CRX',
        'invert_embedding' : True
    }

    # 2. Define Iterables
    all_iter = {

    }
    
    data_iter = {
        'channels_out': [ [0], [1], [2], [3], [4], [5], [6], [7], [8] ]
    }

    model_iter = {
       
    }

    graph_columns = ['q_config', 'channels_out', 'test_auc']

    # 3. Run Experiment
    print("--- Starting Refactored Transformer Experiment ---")
    df_results = make_experiment_transformer(
        exp_config_base, 
        p_base, 
        all_iter=all_iter, 
        data_iter= data_iter,
        model_iter=model_iter,
        graph_columns=graph_columns
    )

    print("--- Refactored Experiment Finished ---")

    # 4. Make plot and save it
    table_dir = script_path.parent / "aggregated_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate: group by all graph_columns except the last (value column)
    value_column = graph_columns[-1]
    group_cols = graph_columns[:-1]

    # Use the reusable aggregation/top-n function
    agg, top_n = aggregate_and_save_top_configs(df = df_results, group_cols = group_cols, value_column = value_column, table_dir = table_dir, n=5)
        