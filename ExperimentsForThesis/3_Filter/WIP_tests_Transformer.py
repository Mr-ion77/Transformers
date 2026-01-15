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
        'experiment_id': 'transformer_results/ExperimentsForThesis/28x28/filter',
        'experiment_name': 'Attention Filter or Not',
        'B': 256,
        'N': 125, # Num epochs
        'num_experiments': 30,
        'num_classes': 7,
        'square' : False,
        'pixels' : 28,
        'q_config' : {'none'},
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : True,
        'send_telegram': True
    }

    p_base = {
        'patch_size': 4,
        'num_head': 16,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 49,
        'special_cls': 'false',
        'mlp_size': 3,
        'quantum': False,
        'U3_layers' : False,
        'dropout': 0.175,
        'parallel': 1,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 1,
        'connectivity': {'edges': [[0, 1], [1, 2], [2, 0]], 'weights': [torch.pi/3] * 3},
        'learning_rate': 0.0025,
        'hidden_size': 48,          # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.985,
        'augmentation_prob' : 0, 
        'val_train_pond' : 1,
        'quanv_kernel_size' : 2,
        'stride' : 1,
        'channels_out' : [1],
        'ancilla': 0,
        'graphs' : 'star',
        'entangle_method' : 'CNOT',
        'invert_embedding' : False
    }

    # 2. Define Iterables
    all_iter = {

    }
    
    data_iter = {
    
    }

    model_iter = {
        'selection_amount' : [49, 40, 30, 25, 20, 15], 
    }

    graph_columns = ['selection_amount', 'test_auc']

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
        