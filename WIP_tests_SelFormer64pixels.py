import torch
from mi_quantum.mkexperiments import make_experiment_selformer2
from benri.data import aggregate_and_save_top_configs
from pathlib import Path

king_measurers = [[4], [4, 5], [4, 5, 7], [4, 5, 7, 0], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8], [4, 5, 7, 0, 8, 1], [4, 5, 7, 0, 8, 1, 3] , [4, 5, 7, 0, 8, 1, 3, 6], [4, 5, 7, 0, 8, 1, 3, 6, 2]]
twox2measurers = [[3], [3, 2], [3, 2, 1], [3, 2, 1, 0]]

# Hyperparams
p1 = {
    '1_learning_rate': 0.0025, '1_hidden_size': 64*3, '1_dropout': 0.225,
    '1_quantum' : False, '1_num_head': 64, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 4, '1_patch_size': 8, '1_weight_decay': 1e-5, '1_attention_selection': 'filter', 
    '1_selection_amount': 32, '1_RD': 1,  '1_special_cls' : 'none', '1_parallel': 1, '1_patience': -1, 
    '1_scheduler_factor': 0.955, '1_q_stride': 1,  '1_augmentation_prob' : 0, '1_val_train_pond' : 1,
    
}

data = {
    'connectivity' : 'chain' ,'entangle_method' : 'CRX', 'ancilla' : 0, 'channels_out' : [1], 'selection_amount': 32,
    'flatten_extra_channels' : False, 'quanv_kernel_size' : 2, 'invert_embedding' : False    
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 64*3, 'dropout': 0.175,
    'quantum' : False, 'num_head': 64, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 10, 'patch_size': 8, 'weight_decay': 1e-4, 'attention_selection': 'none',
    'selection_amount': 64, 'RD': 1, 'special_cls' : 'none', 'parallel': 1, 'patience': -1, 'scheduler_factor': 0.955, 'q_stride': 1, 'augmentation_prob' : 0,
    'val_train_pond' : 1, 'len_channels_scaler' : 1
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : False,
    'num_experiments'       : 20,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'pixels'                : 64,
    'experiment_name'       : 'Resolution64/8x8patchesKernel2x2',
    'experiment_id'         : 'resolution64/8x8patchesKernel2x2',
    'variant'               : 'selformer',
    'B'                     : 256,
    'special_batch_for_data': False,
    'rewind_channels'       : False,
    'N1'                    : 100,
    'N2'                    : 125,
    'q_config'              : {'patchwise', 'none'},
    'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'second_at_a_time'      : False,
    'augmenting'            : p2['augmentation_prob']>0,
    'concatenate_original'  : False
}


all_iter = {  }
data_iter = {  }
m2_iter = {  } 
graph_columns = ['q_config', 'test_auc']

df_results = make_experiment_selformer2(exp_config, p1, data, p2, all_iter = all_iter, data_iter = data_iter, m2_iter= m2_iter, graph_columns = graph_columns)

# 4. Make plot and save it
table_dir = Path("Resolution64/8x8patchesKernel2x2" + "aggregated_tables")
table_dir.mkdir(parents=True, exist_ok=True)

# Aggregate: group by all graph_columns except the last (value column)
value_column = graph_columns[-1]
group_cols = graph_columns[:-1]

# Use the reusable aggregation/top-n function
agg, top_n = aggregate_and_save_top_configs(df = df_results, group_cols = group_cols, value_column = value_column, table_dir = table_dir, n=5)
    