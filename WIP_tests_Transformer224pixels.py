from mi_quantum.mkexperiments import make_experiment_transformer
import torch

if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'experiment_id': '224x224/number_of_heads',
        'experiment_name': 'First Search for 224 pixels Transformer',
        'B': 256,
        'N': 125, # Num epochs
        'num_experiments': 20,
        'num_classes': 7,
        'square' : True,
        'pixels' : 224,
        'q_config' : {'none'},
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : True,
        'send_telegram': True
    }

    p_base = {
        'patch_size': 28,
        'num_head': 28*4,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 32,
        'special_cls': 'false',
        'mlp_size': 8,
        'quantum': False,
        'dropout': 0.175,
        'parallel': 1,
        'attention_selection': 'filter',
        'RD': 2,
        'q_stride': 1,
        'connectivity': {'edges': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], 'weights': [torch.pi/3] * 6},
        'learning_rate': 5e-5,
        'hidden_size': 28*28*3,                 # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.965,
        'augmentation_prob' : 0, 
        'val_train_pond' : 1,
        'quanv_kernel_size' : 2,
        'stride' : 1,
        'channels_out' : [1],
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

    }

    model_iter = {
        'dropout': [ 0.175, 0.25, 0.325 ], 'learning_rate': [5e-5, 1e-4, 5e-4]
    }

    graph_columns = ['dropout' ,'learning_rate', 'test_auc']

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