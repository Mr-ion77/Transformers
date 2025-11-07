import torch
import mi_quantum as qpctorch
import pandas as pd
from pathlib import Path
import itertools
from mi_quantum.quantum.quanvolution import QuantumConv2D
import matplotlib.pyplot as plt
import os, sys, json
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TelegramBot import SendToTelegram

# Hyperparams
p1 = {
    '1_learning_rate': 0.0025, '1_hidden_size': 48, '1_dropout': 0.225,
    '1_quantum' : False, '1_num_head': 4, '1_Attention_N' : 2, '1_num_transf': 2, '1_mlp_size': 5, '1_patch_size': 4, '1_weight_decay': 1e-7, '1_attention_selection': 'none', 
    '1_selection_amount': 49, '1_RD': 1, '1_connectivity' : 'star' ,'1_entangle_method' : 'CRX', '1_special_cls' : False, '1_paralel': 1, '1_patience': -1, 
    '1_scheduler_factor': 0.985, '1_q_stride': 1, '1_ancilla' : 0, '1_channels_out' : [-1], '1_augmentation_prob' : 1, '1_val_train_pond' : 1,
    '1_flatten_extra_channels' : False
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': 0.3,
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 5, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'selection_amount': 20, 'RD': 1, 'special_cls' : False, 'paralel': 2, 'patience': -1, 'scheduler_factor': 0.975, 'q_stride': 1, 'augmentation_prob' : 0,
    'val_train_pond' : 0.5
}

exp_config = {
    'channels_last'         : False,         # True if last dimension of datasets tensors match channels dimension
    'repeat_selector'       : False,         # True to train autoencoder each time for more variability
    'send_telegram'         : True,
    'num_experiments'       : 20,
    'num_classes'           : 7,
    'trained_selector_once' : False,
    'experiment_name'       : 'Dropout and Channels Grid Search + Extra Channels as Extra Patches',
    'experiment_id'         : '/final_stand/dropout_channels_extra_patches',
    'variant'               : 'selformer',
    'B'                     : 256,
    'N1'                    : 125,
    'N2'                    : 105,
    'q_config'              : {'none', 'patchwise'},
    'device'                : torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

default_columns = [
        'test_auc_sel', 'test_acc_sel', '#params_sel' , 'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc', 'train_acc', '#params_class'
    ]

def make_dropout(drop):
    return {'embedding_attn': drop, 'after_attn': drop, 'feedforward': drop, 'embedding_pos': drop}


def make_directories_for_experiment(variant = 'selformer', exp_config = exp_config, p1 = p1, p2 = p2 , all_iter = None,
                                    sel_iter = None, data_iter = None, class_iter = None, columns = default_columns):
    os.makedirs('../QTransformer_Results_and_Datasets/' + variant + '_results/current_results', exist_ok = True)
    try:
        os.makedirs('../QTransformer_Results_and_Datasets/' + variant + '_results/'+ exp_config['experiment_id'], exist_ok = False)
    except FileExistsError:
        print(f"Directory for experiment ID '{exp_config['experiment_id']}' already exists. Results may be overwritten. Make sure to save this results elsewhere" 
                "or modify 'exp_config['experiment_id']' to a new value if you want to keep previous results.")

    with open('../QTransformer_Results_and_Datasets/' + variant + '_results/'+ exp_config['experiment_id'] +'/hyperparameters.json', 'w') as f:
        f.write(f"Experiment Name: {exp_config['experiment_name']}\nBatch Size: {exp_config['B']}\nNumber of Epochs Selector: {exp_config['N1']}\nNumber of Epochs Classifier: {exp_config['N2']}\n")
        f.write('\nHyperparameters for Selector\n')
        json.dump(p1, f, indent=4)
        f.write('\nHyperparameters for Classifier\n')  # Separator text between dictionaries
        json.dump(p2, f, indent=4)
        iter_blocks = [
            ('All iter', all_iter),
            ('Sel iter', sel_iter),
            ('Data iter', data_iter),
            ('Class iter', class_iter),
        ]
        for header, block in iter_blocks:
            if block:
                f.write(f'\n{header}\n')
                json.dump(block, f, indent=4)

    csv_path = '../QTransformer_Results_and_Datasets/' + variant + '_results/' + exp_config['experiment_id'] + '/results_grid_search.csv'
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, mode='a', header=True, index=False)

    return csv_path

def make_experiment_selformer(exp_config, p1_base, p2_base, all_iter={}, sel_iter={}, data_iter={}, class_iter={}, graph_columns = [ 'q_config', 'test_auc']):
    
    root_id = exp_config['experiment_id']
    all_iter_counter = 1
    columns = ['idx', 'q_config', 'channels_out', 'latent_shape'] + list(all_iter.keys()) + list(sel_iter.keys()) + list(data_iter.keys()) + list(class_iter.keys()) + default_columns
    # filepath: /home/carlosR/QTransformer/test_functions.py
    for pack_all in itertools.product(*all_iter.values()):

        experiment_id = root_id + '/all_iter_' + str(all_iter_counter)
        all_iter_counter += 1
        exp_config.update({'experiment_id' : experiment_id, 'trained_selector_once' : False})

        # Create fresh copies for each iteration
        p1 = p1_base.copy()
        p2 = p2_base.copy()
        
        # Modifications only affect the copies
        current_params = dict(zip(all_iter.keys(), pack_all))
        p1.update(current_params)
        p2.update(current_params)
        print("Current overall params:", current_params)


        if __name__ == "__main__":
            try:
                # Save dictionary with all the hyperparameters and results in a json file
                progress = 0
                csv_path = make_directories_for_experiment( variant='selformer', exp_config = exp_config, p1 = p1, p2 = p2, 
                                                            all_iter = all_iter, sel_iter = sel_iter, data_iter = data_iter, class_iter =class_iter, columns = columns )

                # Load data
                notrans_train_dl, train_dl, val_dl, test_dl, shape = qpctorch.data.get_medmnist_dataloaders(
                    pixel=28, data_flag='dermamnist', extra_tr_without_trans = True, batch_size=exp_config['B'], num_workers=4, pin_memory=True
                )

                # Obtain general settings regarding dataset shape
                num_channels = shape[-1] if exp_config['channels_last'] else shape[0]
                progress_levels = [0, 25, 50, 75, 100]

                # Grid search loop
                if exp_config['send_telegram']:
                        SendToTelegram(progress = 0)   

                for idx_ in range(exp_config['num_experiments']):
                    idx = idx_ + 1
                    progress = int( 100* (idx+1)//exp_config['num_experiments'] )
                    if exp_config['send_telegram'] and progress in progress_levels:
                        SendToTelegram(progress = progress)                

                    print(f"\n\nPoint {idx}")
                    save_path = Path(f"../QTransformer_Results_and_Datasets/selformer_results/current_results/grid_search{idx}")
                    save_path.mkdir(parents=True, exist_ok=True)
                    os.makedirs(save_path / 'autoencoder', exist_ok=True)

                    # Determine if quantum processing is enabled based on exp_config['q_config']
                    # exp_config['q_config'] can be a string or a list/tuple; handle both
                    NoneBool, PatchBool = 'none' in exp_config['q_config'], 'patchwise' in exp_config['q_config']
                    print(f"Current quantum configuration:\nNormal latent representations: {NoneBool}\nPatchwise Quantum latent representations: {PatchBool}")

                    for pack_sel in itertools.product(*sel_iter.values()):

                        current_params = dict(zip(sel_iter.keys(), pack_sel))
                        p1.update(current_params)
                        current_params.update({'selection_amount' : p1['1_selection_amount'] // 2})
                        p2.update(current_params)
                        print("Current params for selector:", current_params)

                        if (not exp_config['trained_selector_once']) or exp_config['repeat_selector']:
                            exp_config['trained_selector_once'] = True

                            print(f"\nTraining first model: Autoencoder\nOptiosn: Autoencoder with Quantum Layers: {list(exp_config['q_config'])} and Learning Rate: {p1['1_learning_rate']}\n")

                            # Model
                            model1 = qpctorch.quantum.vit.VisionTransformer(
                                img_size=shape[-1], num_channels=shape[0], num_classes=exp_config['num_classes'],
                                patch_size=p1['1_patch_size'], hidden_size= shape[0]* p1['1_patch_size']**2, num_heads=p1['1_num_head'], Attention_N = p1['1_Attention_N'],
                                num_transformer_blocks=p1['1_num_transf'], attention_selection= p1['1_attention_selection'], selection_amount = p1['1_selection_amount'], special_cls = p1['1_special_cls'], 
                                mlp_hidden_size=p1['1_mlp_size'], quantum_mlp = False, dropout = make_dropout( p1['1_dropout']) , channels_last=exp_config['channels_last'], quantum_classification = False,
                                paralel = p1['1_paralel'], RD = p1['1_RD'], q_stride = p1['1_q_stride'], connectivity = 'chain'
                            )

                            # Train first model
                            test_auc_sel, test_acc_sel, val_auc_sel, val_acc_sel, train_auc_sel, _, params_sel = qpctorch.training.train_and_evaluate(
                                model1, train_dl, val_dl, test_dl, num_classes=exp_config['num_classes'],
                                learning_rate=p1['1_learning_rate'], num_epochs=exp_config['N1'], device=exp_config['device'], mapping=False,
                                res_folder=str(save_path), hidden_size=p1['1_hidden_size'], dropout= make_dropout( p1['1_dropout']),
                                num_heads=p1['1_num_head'], patch_size=p1['1_patch_size'], num_transf=p1['1_num_transf'],
                                mlp=p1['1_mlp_size'], wd=p1['1_weight_decay'], patience= p1['1_patience'], scheduler_factor=p1['1_scheduler_factor'], autoencoder=False,
                                augmentation_prob = p1['1_augmentation_prob'], val_train_pond=p1['1_val_train_pond']
                            )

                            print(f"\nSelector training completed succesfully.\nTest, Val, Train AUC (first step): {test_auc_sel:.5f}, {val_auc_sel:.5f}, {train_auc_sel:.5f}\n")

                        trained_once_none_config = False

                        for pack_data in itertools.product(*data_iter.values()):
                        
                            current_params = dict(zip(data_iter.keys(), pack_data))
                            print("Current params for data:", current_params)
                            p1.update(current_params)
                            p2.update(current_params)
                                
                            # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
                            DataLoaders = [notrans_train_dl, val_dl, test_dl]
                            if NoneBool:
                                NorLatentDatasetsTensors = []
                            if PatchBool:
                                QuLatentDatasetsTensors = []
                                padding = {'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0}
                                print(f'Quantum Conv2D settings: Patch size: 2, Stride:1, \nPadding: {padding}, Connectivity: {p1["1_connectivity"]}, \nEntangle method: {p1["1_entangle_method"]}, Ancilla: {p1["1_ancilla"]}' )
                                Quanvolution = qpctorch.quantum.quanvolution.QuantumConv2D(
                                    patch_size=2, stride=1, padding=padding, channels_out = p1["1_channels_out"], graph = p1["1_connectivity"], entangle_method = p1['1_entangle_method'], ancilla = p1['1_ancilla'], pad_filler = 'median'
                                    ).to(exp_config['device'])


                            print(f"Quantum configuration is {exp_config['q_config']} ")
                            
                            for i, dl in enumerate(DataLoaders):

                                all_labels = []
                                if NoneBool:
                                    all_latents_normal = []
                                if PatchBool:
                                    all_latents_quantum = []

                                for images, labels, _ in tqdm(dl, desc= f"Processing { ['train', 'validation', 'test'][i]} split: \n"):
                                    images = images.to(exp_config['device'])
                                    with torch.no_grad():

                                        B_img = images.shape[0]
                                        selected_patches = model1.get_patches_by_attention(x = images, paralel_branch = 0)[0]

                                        if NoneBool:
                                            all_latents_normal.extend( selected_patches.view((B_img, p1['1_selection_amount'], num_channels* p1['1_patch_size']* p1['1_patch_size'])).cpu() )   
                                        if PatchBool:
                                            # Reshape patches to fit quanvolution input
                                            aux_patches = selected_patches.view(-1, num_channels, p1['1_patch_size'], p1['1_patch_size'])  # (B * num_patches, C, patch_size, patch_size)
                                            aux_patch_outs = Quanvolution(aux_patches)  # (B * num_patches, C, H_out, W_out)
                                            if p1['1_flatten_extra_channels']:
                                                shape_to_reshape_toQ = (B_img, p1['1_selection_amount'], len(p1['1_channels_out']) * num_channels * p1['1_patch_size']**2 )
                                            else:
                                                shape_to_reshape_toQ = (B_img, p1['1_selection_amount'] * len(p1['1_channels_out']), num_channels * p1['1_patch_size']**2 )
                                            latent_patch_aux = aux_patch_outs.reshape(shape_to_reshape_toQ)  # (B, num_patches, new_patch_dim) aux_patch_outs.view((B_img, p1['1_selection_amount'],num_channels, p1['1_patch_size'], p1['1_patch_size']))
                                            all_latents_quantum.extend( latent_patch_aux.cpu() )

                                        all_labels.extend( labels )
                                print("Reshape config: Flatten extra channels?",p1['1_flatten_extra_channels'] )
                                print("Shape out of the convolution:", aux_patch_outs.shape)
                                print("Shape after reshape:", latent_patch_aux.shape)

                                all_labels = torch.tensor(all_labels)
                                if NoneBool:
                                    all_latents_normal = torch.stack(all_latents_normal)
                                    NorLatentDatasetsTensors.append( list(zip(all_latents_normal ,all_labels))  )
                                if PatchBool:
                                    all_latents_quantum = torch.stack(all_latents_quantum)
                                    QuLatentDatasetsTensors.append ( list(zip(all_latents_quantum, all_labels)) )

                            
                            NorLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = exp_config['B'], channels_last = exp_config['channels_last'],
                                                            tensors = NorLatentDatasetsTensors, transforms = {'train': None, 'val': None, 'test': None}
                                                            ) if NoneBool else None
                            
                            PatchLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = exp_config['B'], channels_last = exp_config['channels_last'],
                                                        tensors = QuLatentDatasetsTensors, transforms = {'train': None, 'val': None, 'test': None}
                                                        ) if PatchBool else None
                    

                            Latents = {k: v for k, v in zip(['none', 'patchwise'], [NorLatents, PatchLatents]) if k in exp_config['q_config']}

                            if idx == 0:
                                save_path_latent = Path(f"../QTransformer_Results_and_Datasets/selformer_results/current_results/latent_datasets")
                                save_path_latent.mkdir(parents=True, exist_ok=True)
                                if PatchBool:
                                    torch.save(QuLatentDatasetsTensors[0], save_path_latent / 'qlatent_train_dataset.pt')
                                    torch.save(QuLatentDatasetsTensors[1], save_path_latent / 'qlatent_val_dataset.pt')
                                    torch.save(QuLatentDatasetsTensors[2], save_path_latent / 'qlatent_test_dataset.pt')
                                if NoneBool:
                                    torch.save(NorLatentDatasetsTensors[0], save_path_latent / 'nlatent_train_dataset.pt')
                                    torch.save(NorLatentDatasetsTensors[1], save_path_latent / 'nlatent_val_dataset.pt')
                                    torch.save(NorLatentDatasetsTensors[2], save_path_latent / 'nlatent_test_dataset.pt')

                            # Create second model for the second step)

                            for config in list(exp_config['q_config']):

                                if trained_once_none_config and config == 'none':
                                    continue                                

                                for pack_class in itertools.product(*class_iter.values()):
                        
                                    current_params = dict(zip(class_iter.keys(), pack_class))
                                    print("Current params for classification:", current_params)
                                    p1.update(current_params)
                                    p2.update(current_params)

                                    config_dataset = Latents[config]
                                    shape2 = config_dataset[-1]  # Shape of one sample in the test set
                                    print("Shape2:", shape2)

                                    model2 = qpctorch.quantum.vit.VisionTransformer(
                                        img_size=shape[-1], num_channels= 3, num_classes=exp_config['num_classes'],
                                        patch_size=p2['patch_size'], hidden_size= shape2[-1], num_heads=p2['num_head'], Attention_N = p2['Attention_N'],
                                        num_transformer_blocks=p2['num_transf'], attention_selection= p2['attention_selection'], special_cls = p2['special_cls'], 
                                        mlp_hidden_size=p2['mlp_size'], quantum_mlp = False, dropout = make_dropout(p2['dropout']), channels_last=exp_config['channels_last'], quantum_classification = False,
                                        paralel = p2['paralel'], RD = p2['RD'], q_stride = p2['q_stride'], connectivity = 'chain', patch_embedding_required = 'false'
                                    )

                                    print('\nTraining second model: classifier ViT on latent representations\n')
                                    print(f'QUANTUM SETTING IS: {config} and current lr is: {p2["learning_rate"]}')
                                    
                                    # Train second model
                                    test_auc, test_acc, val_auc, val_acc, train_auc, train_acc, params = qpctorch.training.train_and_evaluate(
                                        model2, config_dataset[0], config_dataset[1], config_dataset[2], num_classes=exp_config['num_classes'],
                                        learning_rate=p2['learning_rate'], num_epochs=exp_config['N2'], device=exp_config['device'], mapping=False,
                                        res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=make_dropout(p2['dropout']),
                                        num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
                                        mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor'], 
                                        autoencoder=False, augmentation_prob = p2['augmentation_prob'], val_train_pond=p2['val_train_pond']
                                    )
                                    
                                    # Save results
                                    row = {
                                        'idx': idx, 
                                            'q_config': config, 'channels_out' : len(p1['1_channels_out']) if config == 'patchwise' else 0, 'latent_shape' : shape2,
                                            'test_auc_sel': test_auc_sel, 'test_acc_sel': test_acc_sel, '#params_sel': params_sel, 
                                            'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc, 'train_auc': train_auc, 'train_acc' : train_acc,
                                            '#params_class': params,
                                            **p1,
                                            **p2
                                    }

                                    pd.DataFrame([row], columns=columns).to_csv(
                                        csv_path, mode='a', header=False, index=False
                                    )

                                if config == 'none':
                                    trained_once_none_config = True     


                if exp_config['send_telegram']:
                    SendToTelegram(csv_file = csv_path, columns = graph_columns,
                                title = exp_config['experiment_name'] )

            except Exception as e:
                SendToTelegram(progress = progress, error_message=str(e))
                print(str(e))


all_iter = {'1_selection_amount' : [49, 25, 20]}
data_iter = { '1_channels_out' : [[3], [2, 3], [1,2,3], [0,1,2,3] ], '1_flatten_extra_channels':[False, True] }
class_iter = {'dropout' : [ 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525 ] }
graph_columns = ['1_flatten_extra_channels', 'channels_out', 'dropout', 'test_auc']

make_experiment_selformer(exp_config, p1, p2, all_iter = all_iter, data_iter= data_iter, class_iter= class_iter, graph_columns= graph_columns)