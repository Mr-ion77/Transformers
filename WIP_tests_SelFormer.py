import torch
import mi_quantum as qpctorch
import pandas as pd
from pathlib import Path
import itertools
from mi_quantum.quantum.quanvolution import QuantumConv2D
import matplotlib.pyplot as plt
import os, sys, json
from tqdm import tqdm

# Config

sys.path.append(str(Path(__file__).resolve().parent.parent))
from TelegramBot import SendToTelegram

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

B = 256
N1 = 150  # Number of epochs Autoencoder
N2 = 150  # Number of epochs Classifier

# Hyperparams
p1 = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 6, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'RD': 1, 'special_cls' : False, 'paralel': 2, 'patience': -1, 'scheduler_factor': 0.9995, 'q_stride': 1  # No early stopping
}

p2 = {
    'learning_rate': [5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3], 'hidden_size': 25*6, 'dropout': 0.1, 'weight_decay': 1e-7, 'n_layers' : 10
}


if __name__ == "__main__":
    try:
        # Save dictionary with all the hyperparameters and results in a json file
        progress = 0
        os.makedirs('../QTransformer_Results_and_Datasets/selformer_results/current_results', exist_ok = True)

        with open('../QTransformer_Results_and_Datasets/selformer_results/current_results/hyperparameters.json', 'w') as f:
            f.write('\nHyperparameters for Autoencoder\n')
            json.dump(p1, f, indent=4)
            f.write('\nHyperparameters for Classifier\n')  # Separator text between dictionaries
            json.dump(p2, f, indent=4)

        columns = [
            # 'idx', 'learning_rate', 'hidden_size', 'dropout', 'num_head', 'num_transf', 'mlp_size', 'patch_size',
            # 'weight_decay', 'attention_selection', 'entangle', 'penny_or_kipu', 'RD', 'convolutional', 'paralel', 
            'idx', 'lr', 'q_config', 'test_mse', 'val_mse', '#params1' , 'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc',  '#params2'
        ]

        channels_last = False           # Set to True if last dimension of datasets tensors match channels dimension
        RepeatSelector = False       # Set to True if you want to train the autoencoder each time for more variability. For a better performance and faster training set to False.
        SendToTelegramBool = True
        NExperiments = 20
        Trained_Selector_Once = True
        
        csv_path = '../QTransformer_Results_and_Datasets/selformer_results/current_results/results_grid_search.csv'
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, mode='a', header=True, index=False)

        q_config = {'none', 'patchwise', 'quanvolution'}
        progress_levels = [0, 25, 50, 75, 100]
        # Grid search loop
        if SendToTelegramBool:
                SendToTelegram(progress = 0)   

        for idx in range(NExperiments):
            progress = int( 100* (idx+1)//NExperiments )
            if SendToTelegramBool and progress in progress_levels:
                SendToTelegram(progress = progress)                

            for lr in [5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3]:
                print(f"\n\nPoint {idx}")
                save_path = Path(f"../QTransformer_Results_and_Datasets/selformer_results/current_results/grid_search{idx}")
                save_path.mkdir(parents=True, exist_ok=True)
                os.makedirs(save_path / 'autoencoder', exist_ok=True)

                # Determine if quantum processing is enabled based on q_config
                # q_config can be a string or a list/tuple; handle both
                NoneBool, PatchBool, QuanvBool = 'none' in q_config, 'patchwise' in q_config, 'quanvolution' in q_config
                print(f"Current quantum configuration:\nNormal latent representations: {NoneBool}\nPatchwise Quantum latent representations: {PatchBool}\nQuanvolution latent representations: {QuanvBool}")

                if (not Trained_Selector_Once) or RepeatSelector:
                    Trained_Autoencoder_Once = True

                    print(f'\nTraining first model: Autoencoder\nOptiosn: Autoencoder with Quantum Layers: {list(q_config)} and Learning Rate: {p["learning_rate"]}\n')

                    # Load data
                    train_dl, val_dl, test_dl, shape = qpctorch.data.get_medmnist_dataloaders(
                        pixel=28, data_flag='dermamnist', batch_size=B, num_workers=4, pin_memory=True
                    )

                    # Obtain general settings regarding dataset shape
                    num_channels = shape[-1] if channels_last else shape[0]
                    img_size = shape[1]  # Assuming square images

                    # Model
                    model1 = qpctorch.quantum.vit.VisionTransformer(
                        img_size=shape[-1], num_channels=shape[0], num_classes=num_classes,
                        patch_size=p['patch_size'], hidden_size= shape[0]* p['patch_size']**2, num_heads=p['num_head'], Attention_N = p['Attention_N'],
                        num_transformer_blocks=p['num_transf'], attention_selection= p['attention_selection'], special_cls = special_cls, 
                        mlp_hidden_size=p['mlp_size'], quantum_mlp = False, dropout = p['dropout'], channels_last=False, entangle=False, quantum_classification = False,
                        paralel = p['paralel'], RD = p['RD'], train_q = False, q_stride = p['q_stride'], connectivity = 'chain'
                    )

                    # Train second model
                    test_auc_sel, test_acc_sel, val_auc_sel, val_acc_sel, train_auc_sel, params_sel = qpctorch.training.train_and_evaluate(
                        model, train_dl, val_dl, test_dl, num_classes=7,
                        learning_rate=p['learning_rate'], num_epochs=N, device=device, mapping=False,
                        res_folder=str(save_path), hidden_size=p['hidden_size'], dropout=p['dropout'],
                        num_heads=p['num_head'], patch_size=p['patch_size'], num_transf=p['num_transf'],
                        mlp=p['mlp_size'], wd=p['weight_decay'], patience= p['patience'], scheduler_factor=p['scheduler_factor'], autoencoder=False
                    )

                    print(f"\nSelector training completed succesfully.\nTest, Val, Train AUC (first step): {test_auc_sel:.5f}, {val_auc_sel:.5f}, {train_auc_sel:.5f}\n")

                    # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
                    DataLoaders = [train_dl, val_dl, test_dl]
                    if NoneBool:
                        NorLatentDatasetsTensors = []
                    if PatchBool:
                        QuLatentDatasetsTensors = []
                        Quanvolution = qpctorch.quantum.quanvolution.QuantumConv2D(patch_size=2, stride=1, padding=1, channels_out = [0], graph= 'chain')


                    print(f'Quantum configuration is {q_config} ')
                    
                    for i, dl in enumerate(DataLoaders):
                        if NoneBool:
                            all_latents_normal = []
                        if PatchBool:
                            all_latents_quantum = []

                        
                        all_labels = []
                        all_indices = []
                        for images, labels, indices in tqdm(dl, desc= f"Processing { ['train', 'validation', 'test'][i]} split: \n"):
                            images = images.to(device)
                            with torch.no_grad():

                                if NoneBool:
                                    normal_outs = []
                                if PatchBool:
                                    quantum_outs = []
                                                 
                                for i in range(p['paralel']):

                                    selected_patches = model1.get_patches_by_attention(x = images, paralel_branch = i)

                                    if NoneBool:
                                        normal_outs.append( selected_patches.cpu() )   
                                    if PatchBool:
                                        # Reshape patches to fit quanvolution input
                                        aux_patches = selected_patches.view(-1, num_channels, p['patch_size'], p['patch_size'])  # (B * num_patches, C, patch_size, patch_size)
                                        aux_patch_outs = Quanvolution(aux_patches)  # (B * num_patches, C, H_out, W_out)
                                        latent_patch_aux = aux_quantum_outs.view(selected_patches.size(0), -1, aux_patch_outs.size(1))  # (B, num_patches, new_patch_dim)

                                        quantum_outs.append( latent_patch_aux.cpu() )


                            all_labels.extend( labels )
                            if NoneBool:
                                normal_representations = torch.cat(normal_outs, dim = -1).cpu()  
                                all_latents_normal.extend(normal_representations) 
                            if PatchBool:
                                quantum_representations = torch.cat(quantum_outs, dim = -1).cpu()        
                                all_latents_quantum.extend( quantum_representations )


                        all_labels = torch.tensor(all_labels)
                        if NoneBool:
                            all_latents_normal = torch.stack(all_latents_normal)
                            NorLatentDatasetsTensors.append( list(zip(all_latents_normal ,all_labels))  )
                        if PatchBool:
                            all_latents_quantum = torch.stack(all_latents_quantum)
                            QuLatentDatasetsTensors.append ( list(zip(all_latents_quantum, all_labels)) )


                            
                    
                    NorLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                                    tensors = NorLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                                    ) if NoneBool else None
                    
                    PatchLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                                tensors = QuLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                                ) if PatchBool else None
            

                    Latents = {k: v for k, v in zip(['none', 'patchwise', 'quanvolution'], [NorLatents, PatchLatents, QuanvLatents]) if k in q_config}

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

                for config in list(q_config):
                    config_dataset = Latents[config]
                    shape2 = config_dataset[-1]
                    p2['learning_rate'] = lr
                    model = qpctorch.quantum.vit.VisionTransformer(
                        img_size=shape[-1], num_channels=shape[0], num_classes=num_classes,
                        patch_size=p['patch_size'], hidden_size= shape[0]* p['patch_size']**2, num_heads=p['num_head'], Attention_N = p['Attention_N'],
                        num_transformer_blocks=p['num_transf'], attention_selection= p['attention_selection'], special_cls = special_cls, 
                        mlp_hidden_size=p['mlp_size'], quantum_mlp = False, dropout = p['dropout'], channels_last=False, entangle=False, quantum_classification = False,
                        paralel = p['paralel'], RD = p['RD'], train_q = False, q_stride = p['q_stride'], connectivity = 'chain'
                    )

                    print('\nTraining second model: classifier ViT on latent representations\n')
                    print(f'QUANTUM SETTING IS: {config} and current lr is: {p2["learning_rate"]}')
                    
                    # Train second model
                    test_auc, test_acc, val_auc, val_acc, train_auc, params = qpctorch.training.train_and_evaluate(
                        model, train_dl, val_dl, test_dl, num_classes=7,
                        learning_rate=p['learning_rate'], num_epochs=N, device=device, mapping=False,
                        res_folder=str(save_path), hidden_size=p['hidden_size'], dropout=p['dropout'],
                        num_heads=p['num_head'], patch_size=p['patch_size'], num_transf=p['num_transf'],
                        mlp=p['mlp_size'], wd=p['weight_decay'], patience= p['patience'], scheduler_factor=p['scheduler_factor'], autoencoder=False
                    )
                    
                    # Save results
                    row = {
                        'idx': idx, 
                            'lr': lr, 'q_config' : config, 'test_auc_sel': test_auc_sel, 'test_acc_sel': val_auc_sel, '#params_sel': params_sel, 
                            'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc, 'train_auc': train_auc,'#params_class': params
                    }

                    pd.DataFrame([row], columns=columns).to_csv(
                        '../QTransformer_Results_and_Datasets/selformer_results/current_results/results_grid_search.csv', mode='a', header=False, index=False
                    )


        if SendToTelegramBool:
            SendToTelegram(csv_file = "../QTransformer_Results_and_Datasets/selformer_results/current_results/results_grid_search.csv", columns = ['lr', 'q_config', 'test_auc'])

    except Exception as e:
         SendToTelegram(progress = progress, error_message=str(e))
         print(str(e))

