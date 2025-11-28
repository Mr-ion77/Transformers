# --- This MUST be at the top ---
import sys, os
from pathlib import Path

# Path to the script itself
script_path = Path(__file__).resolve()

# Path to .../QTransformer
project_dir = script_path.parent.parent.parent
# Path to .../home/carlosR
base_dir = project_dir.parent

# Add BOTH directories to sys.path
sys.path.append(str(project_dir))  # Adds /home/carlosR/QTransformer
sys.path.append(str(base_dir))     # Adds /home/carlosR

import torch, itertools
import json
import pandas as pd
import mi_quantum.quantum as quantum  
import mi_quantum.data as data
from mi_quantum.data import preprocess_and_save, cut_extra_channels_from_latents
import mi_quantum.training as training
import torchvision.utils as vutils

# Hyperparameters:

# Hyperparams
p1 = {
    '1_learning_rate': 0.0025, '1_hidden_size': 48, '1_dropout': 0.3,
    '1_quantum' : False, '1_num_head': 4, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 5, '1_patch_size': 4, '1_weight_decay': 1e-7, '1_attention_selection': 'none', 
    '1_selection_amount': 49, '1_RD': 1, '1_connectivity' : 'king' ,'1_entangle_method' : 'CRX', '1_special_cls' : 'none', '1_parallel': 1, '1_patience': -1, 
    '1_scheduler_factor': 0.985, '1_q_stride': 1, '1_ancilla' : 0, '1_channels_out' : list(range(9)), '1_augmentation_prob' : 0, '1_val_train_pond' : 1,
    '1_flatten_extra_channels' : False, '1_quanv_kernel_size' : 3
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': 0.3,
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 5, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'selection_amount': 49, 'RD': 1, 'special_cls' : 'none', 'parallel': 2, 'patience': -1, 'scheduler_factor': 0.985, 'q_stride': 1, 'augmentation_prob' : 0,
    'val_train_pond' : 1, 'len_channels_scaler' : 2
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : True,
    'num_experiments'       : 1,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'pixels'                : 28,
    'experiment_name'       : 'Resolution224/16x16patches/kernel3x3 Selformer',
    'experiment_id'         : 'final_stand/3x3/dropout_channels/extra_patches/4x4patches/concatenate_original',
    'variant'               : 'selformer',
    'B'                     : 256,
    'special_batch_for_data': False,
    'rewind_channels'       : False,
    'N1'                    : 1,
    'N2'                    : 5,
    'q_config'              : {'patchwise','none'},
    'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'second_at_a_time'      : False,
    'augmenting'            : False,
    'concatenate_original'  : True
}

# Helper functions: 
def make_dropout(drop):
    return {'embedding_attn': drop, 'after_attn': drop, 'feedforward': drop, 'embedding_pos': drop}

def custom_update_dict(original, updates):
    for key, value in updates.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            custom_update_dict(original[key], value)
        else:
            original[key] = value

def make_directories_for_experiment(variant = 'selformer', exp_config = None, p1 = None, p2 = None , all_iter = None,
                                    m1_iter = None, data_iter = None, m2_iter = None, columns = []):
    # This function is kept as-is from your script, but 'exp_config' needs a default
    if exp_config is None:
        raise ValueError("exp_config cannot be None")
        
    base_dir = f"../QTransformer_Results_and_Datasets/{variant}_results"
    exp_dir = os.path.join(base_dir, exp_config['experiment_id'])
    
    os.makedirs(os.path.join(base_dir, 'current_results' if not exp_config['second_at_a_time'] else 'current_results2'), exist_ok=True)
    try:
        os.makedirs(exp_dir, exist_ok=False)
    except FileExistsError as e:
        print(f"Warning: {e}")
        print(f"Directory for experiment ID '{exp_config['experiment_id']}' already exists. Results may be overwritten.")

    # Save hyperparameters
    with open(os.path.join(exp_dir, 'hyperparameters.json'), 'w') as f:
        f.write(f"Experiment Name: {exp_config['experiment_name']}\nBatch Size: {exp_config['B']}\n")
        if p1:
             f.write(f"Number of Epochs Selector: {exp_config.get('N1', 'N/A')}\n")
             f.write('\nHyperparameters for Selector\n')
             json.dump(p1, f, indent=4)
        if p2:
             f.write(f"Number of Epochs Classifier: {exp_config.get('N2', 'N/A')}\n")
             f.write('\nHyperparameters for Classifier\n')
             json.dump(p2, f, indent=4)
        
        # Handle iter blocks
        iter_blocks = [
            ('All iter', all_iter), ('Model 1 iter', m1_iter),
            ('Data iter', data_iter), ('Model 2 iter', m2_iter)
        ]
        for header, block in iter_blocks:
            if block:
                f.write(f'\n{header}\n')
                json.dump(block, f, indent=4)

    csv_path = os.path.join(exp_dir, 'results_grid_search.csv')
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=columns).to_csv(csv_path, mode='a', header=True, index=False)

    return csv_path, exp_dir

def rank_patches_by_attention(attn: torch.Tensor) -> torch.Tensor:
            """
            Ranks image patches by the total attention they receive.

            """
            # Average over heads: (B, T, T)
            attn_mean = attn.mean(dim=1)

            # Total attention received by each token: sum over the source positions (axis=-2)
            # attention_received[b, j] = sum over i of attn[b, i, j]
            attention_received = attn_mean.sum(dim=1)  # shape: (B, T)

            # Sort patches by total attention received, descending
            sorted_indices = attention_received.argsort(dim=1, descending=True)  # shape: (B, T)

            return sorted_indices

#Load Data

csv_path, save_path = make_directories_for_experiment( variant='selformer', exp_config = exp_config, p1 = p1, p2 = p2, 
                                                        all_iter = {}, m1_iter = {}, data_iter = {}, m2_iter = {}, columns = [] )
# Load data
notrans_train_dl, train_dl, val_dl, test_dl, shape = data.get_medmnist_dataloaders(
    pixel = exp_config['pixels'], data_flag='dermamnist', extra_tr_without_trans = True, batch_size=exp_config['B'], num_workers=4, pin_memory=True
)

# Obtain general settings regarding dataset shape
num_channels = shape[-1] if exp_config['channels_last'] else shape[0]
# Train the selector first: 

 # Model
model1 = quantum.vit.VisionTransformer(
    img_size=shape[-1], num_channels=shape[0], num_classes=exp_config['num_classes'],
    patch_size=p1['1_patch_size'], hidden_size= shape[0]* p1['1_patch_size']**2, num_heads=p1['1_num_head'], Attention_N = p1['1_Attention_N'],
    num_transformer_blocks=p1['1_num_transf'], attention_selection= p1['1_attention_selection'], selection_amount = p1['1_selection_amount'], special_cls = p1['1_special_cls'], 
    mlp_hidden_size=p1['1_mlp_size'], quantum_mlp = False, dropout = make_dropout( p1['1_dropout']) , channels_last=exp_config['channels_last'], quantum_classification = False,
    parallel = p1['1_parallel'], RD = p1['1_RD'], q_stride = p1['1_q_stride'], connectivity = 'chain'
)

# Train first model
test_auc_sel, test_acc_sel, val_auc_sel, val_acc_sel, train_auc_sel, _, params_sel = training.train_and_evaluate(
    model1, train_dl, val_dl, test_dl, num_classes= exp_config['num_classes'],
    learning_rate=p1['1_learning_rate'], num_epochs=exp_config['N1'], device=exp_config['device'], mapping=False,
    res_folder=str(save_path), hidden_size=p1['1_hidden_size'], dropout= make_dropout( p1['1_dropout']),
    num_heads=p1['1_num_head'], patch_size=p1['1_patch_size'], num_transf=p1['1_num_transf'],
    mlp=p1['1_mlp_size'], wd=p1['1_weight_decay'], patience= p1['1_patience'], scheduler_factor=p1['1_scheduler_factor'], autoencoder=False,
    augmentation_prob = p1['1_augmentation_prob'], val_train_pond=p1['1_val_train_pond']
)

# Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
DataLoaders = [notrans_train_dl, val_dl, test_dl]
paddings = { 2 : { 'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0 }, 3 : { 'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1 } }

Kernels = { 
            'patchwise' :   quantum.quanvolution.QuantumConv2D(
                                kernel_size = p1['1_quanv_kernel_size'],
                                stride = 1,
                                padding = paddings[p1['1_quanv_kernel_size']],
                                channels_out = p1['1_channels_out'],
                                ancilla= p1['1_ancilla'],
                                graph = p1['1_connectivity'],
                                entangle_method = p1['1_entangle_method']
                            )
            }

Kernels2 = { 'none' : torch.nn.Identity() }

Latents = preprocess_and_save(
    B = exp_config['B'],
    DataLoaders = DataLoaders,
    kernels = Kernels2,
    save_path = f"../QTransformer_Results_and_Datasets/selformer_results/quantum_datasets",
    mode = 'by_selected_patches',
    model1 = model1,
    p1 = p1,
    num_channels = num_channels,
    flatten_extra_channels = p1['1_flatten_extra_channels'],
    device = exp_config['device'],
    flatten = not exp_config['augmenting'], 
    concatenate_original = exp_config.get('concatenate_original', False) # In this case true
)
config = 'none'
config_dataset =  Latents[config] # data.create_dataloaders(data_dir = None, batch_size= 256, channels_last= False, shuffle = False, tensors = [train_dataset, val_dataset, test_dataset], transforms={'train': None, 'val': None, 'test': None}) #
shape2 = config_dataset[-1] #

print("Shape2:", shape2, "1_channels_out:", p1['1_channels_out'])
hidden_size = shape2[-1] if not exp_config['augmenting'] else shape2[-1] * shape2[-2] * shape2[-3]


model2 = quantum.vit.VisionTransformer(
    img_size=shape[-1], num_channels= 3, num_classes=exp_config['num_classes'],
    patch_size=p2['patch_size'], hidden_size= hidden_size, num_heads=p2['num_head'], Attention_N = p2['Attention_N'],
    num_transformer_blocks=p2['num_transf'], attention_selection= p2['attention_selection'], special_cls = p2['special_cls'], 
    mlp_hidden_size=p2['mlp_size'], quantum_mlp = False, dropout = make_dropout(p2['dropout']), channels_last=exp_config['channels_last'], quantum_classification = False,
    parallel = p2['parallel'] , RD = p2['RD'], q_stride = p2['q_stride'], connectivity = 'chain', patch_embedding_required = 'flatten' if exp_config['augmenting'] else 'false'
)


latent_train_dl, latent_val_dl, latent_test_dl = config_dataset[:-1]

# Train second model
test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params = training.train_and_evaluate(
    model2, latent_train_dl, latent_val_dl, latent_test_dl, num_classes=exp_config['num_classes'],
    learning_rate=p2['learning_rate'], num_epochs=exp_config['N2'], device=exp_config['device'], mapping=False,
    res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=make_dropout(p2['dropout']),
    num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
    mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor'], 
    autoencoder=False, augmentation_prob = p2['augmentation_prob'], val_train_pond=p2['val_train_pond']
)

# Now let's what's happening with the attentions!

for i, dl in enumerate(config_dataset[:-1]):

    quantum_percentage = []
    first = True
     
    for imgs, labels, idxs in dl:
        _ , attention_for_images = model2.transformer_blocks[0][0].attn(imgs.to(exp_config['device']))
        indices_by_attention = rank_patches_by_attention(attention_for_images)
        mask = (indices_by_attention >= p1['1_selection_amount'])
        quantum_percentage.append( torch.sum(  mask ) / ( p1['1_selection_amount'] * (1+len(p1['1_channels_out'])) * imgs.shape[0]))

    print(f"Split {i} has an average selection of quantum patches of {sum(quantum_percentage)/len(quantum_percentage)}")








