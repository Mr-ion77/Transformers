import torch
import mi_quantum as qpctorch
import pandas as pd
from pathlib import Path
import itertools
from mi_quantum.quantum.quanvolution import QuantumConv2D, QuantumConv1D
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
N2 = 100  # Number of epochs Classifier

# Hyperparams
p1 = {
    'learning_rate': 5e-3, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.125, 'after_attn': 0.175, 'feedforward': 0.125, 'embedding_pos': 0.125},
    'num_head': 1, 'Attention_N' : 2, 'num_transf': 1, 'mlp_size': 18, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'none', 'entangle_method' : 'CRX',
    'paralel' : 1 ,'connectivity': 'chain', 'RD': 1, 'patience': -1, 'scheduler_factor': 0.999, 'q_stride': 1, 'ancilla' : 0
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 18, 'dropout': {'embedding_attn': 0.15, 'after_attn': 0.175, 'feedforward': 0.15, 'embedding_pos': 0.15},
    'quantum' : False, 'num_head': 1, 'Attention_N' : 2, 'num_transf': 4, 'mlp_size': 18, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'none',
    'RD': 1, 'special_cls' : False, 'paralel': 1, 'patience': -1, 'scheduler_factor': 0.9995, 'q_stride': 1
}

NameOfExperiment = 'AutoEnformer results for None vs Vertical vs Quanvolution with CRX'
ExpID = 'none_vs_vert_vs_quanv_no_ancilla/CRX'

if __name__ == "__main__":
    try:
        # Save dictionary with all the hyperparameters and results in a json file
        progress = 0
        os.makedirs('../QTransformer_Results_and_Datasets/autoenformer_results/current_results', exist_ok = True)
        try:
            os.makedirs('../QTransformer_Results_and_Datasets/autoenformer_results/'+ ExpID, exist_ok = False)
        except FileExistsError:
            raise ValueError(f"Directory for experiment ID '{ExpID}' already exists. Results may be overwritten. Make sure to save this results elsewhere" 
                    "or modify 'ExpID' to a new value if you want to keep previous results.")

        with open('../QTransformer_Results_and_Datasets/autoenformer_results/'+ ExpID +'/hyperparameters.json', 'w') as f:
            f.write('\nHyperparameters for Autoencoder\n')
            json.dump(p1, f, indent=4)
            f.write('\nHyperparameters for Classifier\n')  # Separator text between dictionaries
            json.dump(p2, f, indent=4)

        columns = [
            'idx', 'lr', 'q_config', 'test_mse', 'val_mse', '#params1' , 'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc',  '#params2'
        ]

        channels_last = False           # Set to True if last dimension of datasets tensors match channels dimension
        RepeatAutoencoder = False       # Set to True if you want to train the autoencoder each time for more variability. For a better performance and faster training set to False.
        SendToTelegramBool = True
        NExperiments = 20
        Trained_Autoencoder_Once = False # If RepeatAutoencoder is False, set this to True if you have already trained the autoencoder once and have the latent datasets saved. If False, it will train the autoencoder once and save the latent datasets for future use.


        
        csv_path = '../QTransformer_Results_and_Datasets/autoenformer_results/'+ ExpID +'/results_grid_search.csv'
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, mode='a', header=True, index=False)

        q_config = {'none', 'quanvolution', 'vertical'}  # Options: 'none', 'patchwise', 'quanvolution', 'vertical'
        progress_levels = [0, 25, 50, 75, 100]
        # Grid search loop
        if SendToTelegramBool:
                SendToTelegram(progress = 0)   

        for idx in range(NExperiments):
            progress = int( 100* (idx+1)//NExperiments )
            if SendToTelegramBool and progress in progress_levels:
                SendToTelegram(progress = progress)                

            for lr in [5e-4, 1e-3, 2.5e-3, 5e-3]:
                print(f"\n\nPoint {idx}")
                save_path = Path(f"../QTransformer_Results_and_Datasets/autoenformer_results/current_results/grid_search{idx}")
                save_path.mkdir(parents=True, exist_ok=True)
                os.makedirs(save_path / 'autoencoder', exist_ok=True)

                # Determine if quantum processing is enabled based on q_config
                # q_config can be a string or a list/tuple; handle both
                NoneBool, PatchBool, QuanvBool, VerticalBool = 'none' in q_config, 'patchwise' in q_config, 'quanvolution' in q_config, 'vertical' in q_config
                print(f"Current quantum configuration:\nNormal latent representations: {NoneBool}\nPatchwise Quantum latent representations: {PatchBool}\nQuanvolution latent representations: {QuanvBool}")

                if (not Trained_Autoencoder_Once) or RepeatAutoencoder:
                    Trained_Autoencoder_Once = True
                    # Train autoencoder only once if RepeatAutoencoder is False
                    print(f'\nTraining first model: Autoencoder\nOptiosn: Autoencoder with Quantum Layers: {list(q_config)} and Learning Rate: {p1["learning_rate"]}\n')

                    # Load data
                    train_dl, val_dl, test_dl, shape = qpctorch.data.get_medmnist_dataloaders(
                        pixel=28, data_flag='dermamnist', batch_size=B, num_workers=4, pin_memory=True
                    )

                    # Obtain general settings regarding dataset shape
                    num_channels = shape[-1] if channels_last else shape[0]
                    img_size = shape[1]  # Assuming square images

                    # Model
                    model1 = qpctorch.quantum.vit.AutoEnformer(
                        img_size=img_size, num_channels=num_channels,   # set num_classes as needed
                        patch_size=p1['patch_size'], hidden_size=p1['hidden_size'], num_heads=p1['num_head'],
                        num_transformer_blocks=p1['num_transf'], attention_selection=p1['attention_selection'],
                        mlp_hidden_size=p1['mlp_size'], Attention_N = p1['Attention_N'], dropout=p1['dropout'], 
                        paralel = p1['paralel'], channels_last=False, q_stride= p1['q_stride']
                    )

                    print(model1.q_stride)

                    # Train
                    test_mse, val_mse, params1 = qpctorch.training.train_and_evaluate(
                        model1, train_dl, val_dl, test_dl, num_classes=7,
                        learning_rate=p1['learning_rate'], num_epochs=N1, device=device, mapping=False,
                        res_folder=str(save_path) + '/autoencoder', hidden_size=p1['hidden_size'], dropout=p1['dropout'],
                        num_heads=p1['num_head'], patch_size=p1['patch_size'], num_transf=p1['num_transf'],
                        mlp=p1['mlp_size'], wd=p1['weight_decay'], patience= p1['patience'], scheduler_factor= p1['scheduler_factor'], autoencoder=True,
                        save_reconstructed_images = True if idx == 0 else False
                    ) # type: ignore

                    print(f"\nAutoencoder training completed succesfully.\nTest MSE (first step): {test_mse:.5f}")

                    # Prepare datasets for the second step: get latent representations for each dataset and transform them into a new dataloader
                    DataLoaders = [train_dl, val_dl, test_dl]
                    if NoneBool:
                        NorLatentDatasetsTensors = []
                    if PatchBool:
                        QuLatentDatasetsTensors = []
                        QuantumLayer = qpctorch.quantum.pennylane_backend.QuantumLayer(num_qubits = 9, graph = p1['connectivity'], entangle_method = p1['entangle_method'])
                    if QuanvBool:
                        MoLatentDatasetsTensors = []
                        Quanvolution = QuantumConv2D(patch_size=3, stride=1, padding=1, channels_out = [4], graph= p1['connectivity'], entangle_method= p1['entangle_method'], ancilla = p1['ancilla'])
                    if VerticalBool:
                        VoLatentDatasetsTensors = []
                        padding = {'Up': 1, 'Down': 1} 
                        VerticalQuanvolution = QuantumConv1D(window_size=3, stride=1, padding=padding, channels_out = [1], graph= p1['connectivity'], entangle_method= p1['entangle_method'], ancilla = p1['ancilla'])

                    print(f'Quantum configuration is {q_config} ')
                    
                    for i, dl in enumerate(DataLoaders):
                        if NoneBool:
                            all_latents_normal = []
                        if PatchBool:
                            all_latents_quantum = []
                        if QuanvBool:
                            all_latents_quanv = []
                        if VerticalBool:
                            all_latents_vertical = []
                        
                        all_labels = []
                        all_indices = []
                        for images, labels, indices in tqdm(dl, desc= f"Processing { ['train', 'validation', 'test'][i]} split: \t"):
                            images = images.to(device)
                            with torch.no_grad():

                                if NoneBool:
                                    normal_outs = []
                                if PatchBool:
                                    quantum_outs = []
                                if QuanvBool:
                                    quanv_outs = []  
                                if VerticalBool:
                                    vertical_outs = []                 
                                
                                latent_aux = model1.get_latent_representation(images)
                                
                                for i in range(p1['paralel']):
                                    if NoneBool:
                                        normal_outs.append( latent_aux[i] )   # FOR NEXT TEST ONLY WITH NORMAL
                                    if PatchBool:
                                        quantum_outs.append( QuantumLayer( latent_aux[i] ) )
                                    if QuanvBool:
                                        quanv_outs.append( Quanvolution( latent_aux[i].unsqueeze(dim = 1) ) ) # Add channel dimension = 1 channel
                                    if VerticalBool:
                                        vertical_outs.append( VerticalQuanvolution( latent_aux[i].unsqueeze(dim = 1) ) ) # Add channel dimension = 1 channel

                            all_labels.extend( labels )
                            if NoneBool:
                                normal_representations = torch.cat(normal_outs, dim = -1).cpu()  
                                all_latents_normal.extend(normal_representations) # all_latents_normal.extend( latent_aux.view( (latent_aux.shape[1],latent_aux.shape[2], -1) ) )
                            if PatchBool:
                                quantum_representations = torch.cat(quantum_outs, dim = -1).cpu()        
                                all_latents_quantum.extend( quantum_representations )
                            if QuanvBool:
                                quanv_representations = torch.cat(quanv_outs, dim = -1).cpu()
                                all_latents_quanv.extend( quanv_representations )
                            if VerticalBool:
                                vertical_representations = torch.cat(vertical_outs, dim = -1).cpu()
                                all_latents_vertical.extend( vertical_representations )

                        all_labels = torch.tensor(all_labels)
                        if NoneBool:
                            all_latents_normal = torch.stack(all_latents_normal)
                            NorLatentDatasetsTensors.append( list(zip(all_latents_normal ,all_labels))  )
                        if PatchBool:
                            all_latents_quantum = torch.stack(all_latents_quantum)
                            QuLatentDatasetsTensors.append ( list(zip(all_latents_quantum, all_labels)) )
                        if QuanvBool:
                            all_latents_quanv = torch.stack(all_latents_quanv)
                            MoLatentDatasetsTensors.append ( list(zip(all_latents_quanv, all_labels)) )
                        if VerticalBool:
                            all_latents_vertical = torch.stack(all_latents_vertical)
                            VoLatentDatasetsTensors.append ( list(zip(all_latents_vertical, all_labels)) )

                            
                    
                    NorLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                                    tensors = NorLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                                    ) if NoneBool else None
                    
                    PatchLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                                tensors = QuLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                                ) if PatchBool else None
            
                    QuanvLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                                tensors = MoLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                                ) if QuanvBool else None

                    VerticalLatents = qpctorch.data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                                tensors = VoLatentDatasetsTensors, transforms={'train': None, 'val': None, 'test': None}
                                                ) if VerticalBool else None

                    Latents = {k: v for k, v in zip(['none', 'patchwise', 'quanvolution', 'vertical'], [NorLatents, PatchLatents, QuanvLatents, VerticalLatents]) if k in q_config}

                    if idx == 0:
                        save_path_latent = Path(f"../QTransformer_Results_and_Datasets/autoenformer_results/current_results/latent_datasets")
                        save_path_latent.mkdir(parents=True, exist_ok=True)
                        if PatchBool:
                            torch.save(QuLatentDatasetsTensors[0], save_path_latent / 'qlatent_train_dataset.pt')
                            torch.save(QuLatentDatasetsTensors[1], save_path_latent / 'qlatent_val_dataset.pt')
                            torch.save(QuLatentDatasetsTensors[2], save_path_latent / 'qlatent_test_dataset.pt')
                        if NoneBool:
                            torch.save(NorLatentDatasetsTensors[0], save_path_latent / 'nlatent_train_dataset.pt')
                            torch.save(NorLatentDatasetsTensors[1], save_path_latent / 'nlatent_val_dataset.pt')
                            torch.save(NorLatentDatasetsTensors[2], save_path_latent / 'nlatent_test_dataset.pt')
                        if QuanvBool:
                            torch.save(MoLatentDatasetsTensors[0], save_path_latent / 'mlatent_train_dataset.pt')
                            torch.save(MoLatentDatasetsTensors[1], save_path_latent / 'mlatent_val_dataset.pt')
                            torch.save(MoLatentDatasetsTensors[2], save_path_latent / 'mlatent_test_dataset.pt')
                        if VerticalBool:
                            torch.save(VoLatentDatasetsTensors[0], save_path_latent / 'vlatent_train_dataset.pt')
                            torch.save(VoLatentDatasetsTensors[1], save_path_latent / 'vlatent_val_dataset.pt')
                            torch.save(VoLatentDatasetsTensors[2], save_path_latent / 'vlatent_test_dataset.pt')
                # Create second model for the second step)

                for config in list(q_config):
                    config_dataset = Latents[config]
                    shape2 = config_dataset[-1]
                    p2['learning_rate'] = lr
                    model2 = qpctorch.quantum.vit.DeViT(num_classes=7, p = p2, shape = shape, dim_latent = shape2[-1]) # The shape needed is that of the original images, in this case [3, 28, 28]

                    print('\nTraining second model: classifier ViT on latent representations\n')
                    print(f'QUANTUM SETTING IS: {config} and current lr is: {p2["learning_rate"]}')
                    # Train second model
                    test_auc, test_acc, val_auc, val_acc, train_auc, params2 = qpctorch.training.train_and_evaluate(
                        model2, config_dataset[0], config_dataset[1], config_dataset[2], num_classes=7,
                        learning_rate=p2['learning_rate'], num_epochs=N2, device=device, mapping=False,
                        res_folder=str(save_path), hidden_size=p2['hidden_size'], dropout=p2['dropout'],
                        num_heads=p2['num_head'], patch_size=p2['patch_size'], num_transf=p2['num_transf'],
                        mlp=p2['mlp_size'], wd=p2['weight_decay'], patience= p2['patience'], scheduler_factor=p2['scheduler_factor'], autoencoder=False
                    )
                    
                    # Save results
                    row = {
                        'idx': idx, 
                            'lr': lr, 'q_config' : config, 'test_mse': test_mse, 'val_mse': val_mse, '#params1': params1, 
                            'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 'val_acc': val_acc, 'train_auc': train_auc,'#params2': params2,
                            **p1, **p2
                    }

                    pd.DataFrame([row], columns=columns).to_csv(csv_path, mode='a', header=False, index=False)


        if SendToTelegramBool:
            SendToTelegram(csv_file = csv_path, columns = ['lr', 'q_config', 'test_auc'], title = NameOfExperiment)

    except Exception as e:
         SendToTelegram(progress = progress, error_message=str(e))
         print(str(e))

