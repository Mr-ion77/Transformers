import torch
from mi_quantum.mkexperiments import make_experiment_selformer

king_measurers = [[4], [4, 5], [4, 5, 7], [4, 5, 7, 0], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8, 1], [4, 5, 7, 0, 8, 1, 3] , [4, 5, 7, 0, 8, 1, 3, 6], [4, 5, 7, 0, 8, 1, 3, 6, 2]]
twox2measurers = [[3], [3, 2], [3, 2, 1], [3, 2, 1, 0]]

# Hyperparams
p1 = {
    '1_learning_rate': 0.00025, '1_hidden_size': 147, '1_dropout': 0.075,
    '1_quantum' : False, '1_num_head': 3, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 35, '1_patch_size': 7, '1_weight_decay': 1e-7, '1_attention_selection': 'none', 
    '1_selection_amount': 8, '1_RD': 1, '1_connectivity' : 'king' ,'1_entangle_method' : 'CRX', '1_special_cls' : False, '1_parallel': 1, '1_patience': -1, 
    '1_scheduler_factor': 0.975, '1_q_stride': 1, '1_ancilla' : 0, '1_channels_out' : king_measurers[ -1 ], '1_augmentation_prob' : 0, '1_val_train_pond' : 1,
    '1_flatten_extra_channels' : False, '1_quanv_kernel_size' : 3
}

p2 = {
    'learning_rate': 0.00025, 'hidden_size': 147, 'dropout': 0.075,
    'quantum' : False, 'num_head': 3, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 35, 'patch_size': 7, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'selection_amount': 8, 'RD': 1, 'special_cls' : False, 'parallel': 2, 'patience': -1, 'scheduler_factor': 0.975, 'q_stride': 1, 'augmentation_prob' : 1,
    'val_train_pond' : 1, 'len_channels_scaler' : 1
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : True,
    'num_experiments'       : 20,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'pixels'                : 28,
    'experiment_name'       : 'Dropout and Channels Grid Search + Extra Channels as Extra Patches Kernel 3x3 Patches 7x7',
    'experiment_id'         : 'pixel',
    'variant'               : 'selformer',
    'B'                     : 256,
    'special_batch_for_data': False,
    'rewind_channels'       : False,
    'N1'                    : 125,
    'N2'                    : 125,
    'q_config'              : {'patchwise', 'none'},
    'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'second_at_a_time'      : False,
    'augmenting'            : True,
}


all_iter = {  }
data_iter = {  }
m2_iter = { 'learning_rate': [ 2.5e-5 , 2.5e-4, 2.5e-3] , 'augmentation_prob': [1, 0.5, 0] } 
graph_columns = ['augmentation_prob', 'channels_out', 'test_auc']

make_experiment_selformer(exp_config, p1, p2, all_iter = all_iter, data_iter = data_iter, m2_iter= m2_iter, graph_columns = graph_columns)