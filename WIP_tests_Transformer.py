from mi_quantum.mkexperiments import make_experiment_transformer
import torch

if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'experiment_id': 'crx_vs_cosine',
        'experiment_name': 'Is really CRX better than just cosine?',
        'B': 256,
        'N': 125, # Num epochs
        'num_experiments': 20,
        'num_classes': 7,
        'square' : True,
        'pixels' : 28,
        'q_config' : {'cosine2'},
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : True,
        'send_telegram': True
    }

    p_base = {
        'patch_size': 4,
        'num_head': 4,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 25,
        'special_cls': 'false',
        'mlp_size': 6,
        'quantum': False,
        'dropout': 0.225,
        'parallel': 2,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 2,
        'connectivity': 'star',
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
        'graph' : 'star',
        'entangle_method' : 'CRX',
        'invert_embedding' : True
    }

    # 2. Define Iterables
    all_iter = {

    }
    
    data_iter = {
       
    }

    model_iter = {
        'dropout': [0.225, 0.3, 0.375],
    }

    graph_columns = ['q_config', 'dropout', 'test_auc']

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