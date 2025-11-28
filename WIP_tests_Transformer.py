from mi_quantum.mkexperiments import make_experiment_transformer
import torch

if __name__ == "__main__":
    
    # 1. Define Base Configs
    exp_config_base = {
        'experiment_id': 'q_stride_redo',
        'experiment_name': 'Striding effect on quantum transformer',
        'B': 256,
        'N': 125, # Num epochs
        'num_experiments': 20,
        'num_classes': 7,
        'square' : True,
        'channels_last': False,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'second_at_a_time' : False,
        'send_telegram': True
    }

    p_base = {
        'patch_size': 4,
        'num_head': 4,
        'Attention_N': 2,
        'num_transf': 2,
        'selection_amount': 25,
        'special_cls': 'false',
        'mlp_size': 4,
        'quantum': True,
        'dropout': 0.225,
        'parallel': 2,
        'attention_selection': 'filter',
        'RD': 1,
        'q_stride': 1,
        'connectivity': 'star',
        'learning_rate': 0.0025,
        'hidden_size': 48, # Example, might be derived
        'weight_decay': 1e-7,
        'patience': -1,
        'scheduler_factor': 0.985,
        'augmentation_prob' : 0, 
        'val_train_pond' : 1,
    }

    # 2. Define Iterables
    all_iter = {

    }
    
    model_iter = {
        'quantum' : [True, False],
        'q_stride': [1, 2, 3, 4],
    }

    graph_columns = ['quantum', 'q_stride', 'test_auc']

    # 3. Run Experiment
    print("--- Starting Refactored Transformer Experiment ---")
    make_experiment_transformer(
        exp_config_base, 
        p_base, 
        all_iter=all_iter, 
        model_iter=model_iter,
        graph_columns=graph_columns
    )
    print("--- Refactored Experiment Finished ---")