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
        'experiment_id': 'transformer_results/ExperimentsForThesis/28x28/q_preprocessor_nt_mod_king',
        'experiment_name': '11_Quantum_Preprocessor_NT_mod_king',
        'B': 256,
        'N': 100, # Num epochs
        'num_experiments': 50,
        'num_classes': 7,
        'square' : False,
        'pixels' : 28,
        'q_config' : {'quantum', 'none'},
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : True,
        'send_telegram': True,
        'verbose': True,
        'concatenate_original' : False
    }
    w = torch.pi/3
    p_base = {
        'patch_size': 4,
        'num_head': 16,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 49,
        'special_cls': 'false',
        'mlp_size': 3,
        'quantum': False,
        'train_q' : False,
        'U3_layers' : 0,
        'entangling_layers' : 1,
        'dropout': 0.175,
        'parallel': 1,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 1,
        'connectivity': {'edges': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1],
                                         [3, 4], [6, 4]], 'weights': [w,w,w,w,w,w,w,w,w,w]}, # Modified king
        'learning_rate': 0.0025,
        'hidden_size': 48,          # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.985,
        'augmentation_prob' : 0, 
        'val_train_pond' : 1,
        'quanv_kernel_size' : 3,
        'stride' : 1,
        'channels_out' : [1],
        'ancilla' : 0,
        'graphs' : {'edges': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1],
                                         [3, 4], [6, 4]], 'weights': [w,w,w,w,w,w,w,w,w,w]}, # Modified King
        'entangle_method' : 'CRX', 
        'invert_embedding' : True,

    }

    # 2. Define Iterables
    all_iter = {
        
    }
    
    data_iter = {
        'channels_out' : [ [0], [1], [2], [3], [4], [5], [6], [7], [8] ]
    }

    model_iter = {
        
    }

    graph_columns = ['preprocessor', 'test_auc']

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
        