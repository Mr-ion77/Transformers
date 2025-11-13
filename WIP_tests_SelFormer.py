import torch
from mi_quantum.mkexperiments import make_experiment_selformer
# Hyperparams
p1 = {
    '1_learning_rate': 0.0025, '1_hidden_size': 48, '1_dropout': 0.225,
    '1_quantum' : False, '1_num_head': 4, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 5, '1_patch_size': 4, '1_weight_decay': 1e-7, '1_attention_selection': 'none', 
    '1_selection_amount': 49, '1_RD': 1, '1_connectivity' : 'star' ,'1_entangle_method' : 'CRX', '1_special_cls' : False, '1_paralel': 1, '1_patience': -1, 
    '1_scheduler_factor': 0.985, '1_q_stride': 1, '1_ancilla' : 0, '1_channels_out' : [-1],'1_augmentation_prob' : 1, '1_val_train_pond' : 1,
    '1_flatten_extra_channels' : False, '1_quanv_kernel_size' : 2
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': 0.3,
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 5, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'selection_amount': 20, 'RD': 1, 'special_cls' : False, 'paralel': 2, 'patience': -1, 'scheduler_factor': 0.975, 'q_stride': 1, 'augmentation_prob' : 0,
    'val_train_pond' : 0.5, 'len_channels_scaler' : 3
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : True,
    'num_experiments'       : 20,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'experiment_name'       : 'Dropout and Channels Grid Search + Extra Channels as Extra Patches Kernel 3x3',
    'experiment_id'         : 'final_stand/3x3/dropout_channels/extra_patches/dinamic_sel_amount/',
    'variant'               : 'selformer',
    'B'                     : 256,
    'N1'                    : 125,
    'N2'                    : 105,
    'q_config'              : {'none', 'patchwise'},
    'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

king_measurers = [[4], [4, 5], [4, 5, 7], [4, 5, 7, 0], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8, 1], [4, 5, 7, 0, 8, 1, 3] , [4, 5, 7, 0, 8, 1, 3, 6], [4, 5, 7, 0, 8, 1, 3, 6, 2]]

all_iter = {'1_selection_amount' : [20] }
data_iter = { '1_channels_out' : [ king_measurers[i] for i in [0, 2, 4, 6, 8] ], '1_flatten_extra_channels':[False] }
m2_iter = {'dropout' : [0.275, 0.325, 0.375, 0.425, 0.475], 'len_channels_scaler': [0, 1, 2, 3, 4] } # [ 0.275, 0.375, 0.425, 0.475 ]
graph_columns = ['len_channels_scaler', 'channels_out', 'test_auc']

make_experiment_selformer(exp_config, p1, p2, all_iter = all_iter, data_iter = data_iter, m2_iter= m2_iter, graph_columns = graph_columns)