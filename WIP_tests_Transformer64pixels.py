from mi_quantum.mkexperiments import make_experiment_transformer
import torch

if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'experiment_id': 'wd_after_edges_drop_and_lr',
        'experiment_name': 'Edges search: Dropout and Learning Rate with Weight Decay',
        'B': 256,
        'N': 125, # Num epochs
        'num_experiments': 20,
        'num_classes': 7,
        'square' : True,
        'pixels' : 64,
        'q_config' : {'quantum'},
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : False,
        'send_telegram': True
    }

    p_base = {
        'patch_size': 8,
        'num_head': 8,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 32,
        'special_cls': 'false',
        'mlp_size': 4,
        'quantum': False,
        'dropout': 0.225,
        'parallel': 2,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 1,
        'connectivity': 'star',
        'learning_rate': 0.0025,
        'hidden_size': 48,          # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.965,
        'augmentation_prob' : 0, 
        'val_train_pond' : 1,
        'quanv_kernel_size' : 2,
        'stride' : 1,
        'channels_out' : [3],
        'ancilla': 0,
        'graph' : 'star',
        'entangle_method' : 'edges',
        'invert_embedding' : True
    }

    # 2. Define Iterables
    all_iter = {

    }

    # Repetir amb més dropout
    
    data_iter = {
        'entangle_method' : [ 'edges', 'CRX' ], 'channels_out' : [ [3], [1] ], 'invert_embedding' : [True, False]
    }

    model_iter = {
       'dropout': [0.225], 'learning_rate': [  7.5e-4  ], 'weight_decay': [ 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 ]
    }

    graph_columns = ['invert_embedding', 'entangle_method', 'weight_decay', 'test_auc']

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