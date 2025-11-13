from mi_quantum.mkexperiments import make_experiment_transformer
import torch

if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'experiment_id': 'Transformer_Test_v1',
        'experiment_name': 'Test ViT Experiment',
        'B': 256,
        'N': 100, # Num epochs
        'num_experiments': 10,
        'num_classes': 7,
        'channels_last': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'send_telegram': True
    }

    p_base = {
        'patch_size': 4,
        'num_head': 2,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 20,
        'special_cls': True,
        'mlp_size': 5,
        'quantum': False,
        'dropout': 0.225,
        'paralel': 2,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 1,
        'connectivity': 'chain',
        'learning_rate': 0.0025,
        'hidden_size': 48, # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.995
    }

    # 2. Define Iterables
    all_iter = {
        'learning_rate': [0.0025, 0.005, 0.01],
    }
    
    model_iter = {
        'num_transf': [1, 2]
    }

    # 3. Run Experiment
    print("--- Starting Refactored Transformer Experiment ---")
    make_experiment_transformer(
        exp_config_base, 
        p_base, 
        all_iter=all_iter, 
        model_iter=model_iter
    )
    print("--- Refactored Experiment Finished ---")