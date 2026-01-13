from mi_quantum.mkexperiments import make_experiment_transformer
import torch

if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'experiment_id': '128x128/quantum_vs_none',
        'experiment_name': 'Quantum vs None (CRX)',
        'B': 256,
        'N': 125, # Num epochs
        'num_experiments': 20,
        'num_classes': 7,
        'square' : True,
        'pixels' : 128,
        'q_config' : {'none', 'quantum'},
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : True,
        'send_telegram': True
    }

    p_base = {
        'patch_size': 16,
        'num_head': 16,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 32,
        'special_cls': 'false',
        'mlp_size': 5,
        'quantum': False,
        'dropout': 0.175,
        'parallel': 2,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 1,
        'connectivity': {'edges': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], 'weights': [torch.pi/3] * 6},  # Example, might be derived
        'learning_rate': 7.5e-5,
        'hidden_size': 48,          # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.965,
        'augmentation_prob' : 0, 
        'val_train_pond' : 1,
        'quanv_kernel_size' : 2,
        'stride' : 1,
        'channels_out' : [0],
        'ancilla': 0,
        'graph' : 'star',
        'entangle_method' : 'CRX',
        'invert_embedding' : True
    }

    # 2. Define Iterables
    all_iter = {

    }

    # Repetir amb més dropout
    
    data_iter = {
        'channels_out' : [ [0], [1], [2], [3]]
    }

    model_iter = {

    }

    graph_columns = ['q_config', 'channels_out', 'test_auc']

    # 3. Run Experiment
    print("--- Starting Refactored Transformer Experiment ---")
    make_experiment_transformer(
        exp_config_base, 
        p_base, 
        all_iter=all_iter, 
        data_iter= data_iter,
        model_iter=model_iter,
        graph_columns=graph_columns
    )
    print("--- Refactored Experiment Finished ---")