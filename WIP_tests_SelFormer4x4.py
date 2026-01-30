import torch
from mi_quantum.mkexperiments import make_experiment_selformer

king_measurers = [[4], [4, 5], [4, 5, 7], [4, 5, 7, 0], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8, 1], [4, 5, 7, 0, 8, 1, 3] , [4, 5, 7, 0, 8, 1, 3, 6], [4, 5, 7, 0, 8, 1, 3, 6, 2]]
twox2measurers = [[3], [3, 2], [3, 2, 1], [3, 2, 1, 0]]
graph = [ [0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4] ]

# Hyperparams
p1 = {
    '1_learning_rate': 0.0025, '1_hidden_size': 48, '1_dropout': 0.175,
    '1_quantum' : False, '1_num_head': 4, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 5, '1_patch_size': 4, '1_weight_decay': 1e-7, '1_attention_selection': 'none', 
    '1_selection_amount': 25, '1_RD': 1, '1_connectivity' : 'star' ,'1_entangle_method' : 'CRX', '1_special_cls' : 'none', '1_parallel': 2, '1_patience': -1, 
    '1_scheduler_factor': 0.985, '1_q_stride': 1, '1_ancilla' : 0, '1_channels_out' : [1], '1_augmentation_prob' : 0, '1_val_train_pond' : 1,
    '1_flatten_extra_channels' : False, '1_quanv_kernel_size' : 2
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': 0.175,
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 5, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'selection_amount': 25, 'RD': 1, 'special_cls' : 'none', 'parallel': 2, 'parallel_mode' : 'quantum', 'patience': -1, 'scheduler_factor': 0.975, 'q_stride': 1, 'augmentation_prob' : 0,
    'val_train_pond' : 1, 'len_channels_scaler' : 0
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : True,
    'num_experiments'       : 2,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'use_selector_as_class' : True,
    'pixels'                : 28,
    'experiment_name'       : 'Concatenate Original 4x4 Patches with 2x2 Kernel, but parallel',
    'experiment_id'         : 'final_stand/2x2kernel/4x4patches/classical_sel_as_class',
    'variant'               : 'selformer',
    'B'                     : 256,
    'special_batch_for_data': False,
    'rewind_channels'       : False,
    'N1'                    : 50,
    'N2'                    : 150,
    'q_config'              : {'none'},
    'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'second_at_a_time'      : True,
    'augmenting'            : p2['augmentation_prob'] > 0,
    'concatenate_original'  : False
}


all_iter = { }
data_iter = { 'use_selector_as_class' : [True, False] } # ,  [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
m2_iter = { 'aux' : list(range(10)) } 
graph_columns = [ 'use_selector_as_class', 'test_auc']

make_experiment_selformer(exp_config, p1, p2, all_iter = all_iter, data_iter = data_iter, m2_iter= m2_iter, graph_columns = graph_columns)

