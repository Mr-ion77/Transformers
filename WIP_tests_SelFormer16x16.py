import torch
from mi_quantum.mkexperiments import make_experiment_selformer

king_measurers = [[4], [4, 5], [4, 5, 7], [4, 5, 7, 0], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8, 1], [4, 5, 7, 0, 8, 1, 3] , [4, 5, 7, 0, 8, 1, 3, 6], [4, 5, 7, 0, 8, 1, 3, 6, 2]]
twox2measurers = [[3], [3, 2], [3, 2, 1], [3, 2, 1, 0]]

# Hyperparams
p1 = {
    '1_learning_rate': 0.00025, '1_hidden_size': 768, '1_dropout': 0.3,
    '1_quantum' : False, '1_num_head': 8, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 5, '1_patch_size': 16, '1_weight_decay': 1e-4, '1_attention_selection': 'none', 
    '1_selection_amount': 98, '1_RD': 1, '1_connectivity' : 'king' ,'1_entangle_method' : 'CRX', '1_special_cls' : 'none', '1_paralel': 1, '1_patience': -1, 
    '1_scheduler_factor': 0.955, '1_q_stride': 1, '1_ancilla' : 0, '1_channels_out' : king_measurers[ -1 ], '1_augmentation_prob' : 0, '1_val_train_pond' : 1,
    '1_flatten_extra_channels' : False, '1_quanv_kernel_size' : 3
}

p2 = {
    'learning_rate': 0.00025, 'hidden_size': 768, 'dropout': 0.3,
    'quantum' : False, 'num_head': 8, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 5, 'patch_size': 16, 'weight_decay': 1e-4, 'attention_selection': 'filter',
    'selection_amount': 98, 'RD': 1, 'special_cls' : 'none', 'paralel': 1, 'patience': -1, 'scheduler_factor': 0.955, 'q_stride': 1, 'augmentation_prob' : 1,
    'val_train_pond' : 1, 'len_channels_scaler' : 1
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : True,
    'num_experiments'       : 20,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'pixels'                : 224,
    'experiment_name'       : 'Resolution224/16x16patches/kernel3x3 Selformer',
    'experiment_id'         : 'resolution224/16x16patches/kernel3x3/dropout_lr',
    'variant'               : 'selformer',
    'B'                     : 128,
    'special_batch_for_data': False,
    'rewind_channels'       : False,
    'N1'                    : 100,
    'N2'                    : 100,
    'q_config'              : {'none'},
    'device'                : torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'second_at_a_time'      : False,
    'augmenting'            : False,
}


all_iter = {  }
data_iter = {  }
m2_iter = { 'learning_rate': [ 2.5e-4 ], 'dropout' :  [0.525, 0.625, 0.725]  } 
graph_columns = ['dropout', 'test_auc']

make_experiment_selformer(exp_config, p1, p2, all_iter = all_iter, data_iter = data_iter, m2_iter= m2_iter, graph_columns = graph_columns)