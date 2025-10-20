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
N = 150 # Number of epochs

# Hyperparams

p = {
    'learning_rate': 0.0025, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
    'quantum' : False, 'num_head': 4, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 6, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'RD': 1, 'special_cls' : False, 'paralel': 2, 'patience': -1, 'scheduler_factor': 0.998, 'q_stride': 4, 'connectivity': 'david_star', 'selection_amount': 25  # No early stopping
}

NameOfExperiment = 'Attention Filtering with selection amount 25'
ExpID = 'attn_filter_sel_amount_25'

num_classes = 7

if __name__ == "__main__":
    try:
        # Save dictionary with all the hyperparameters and results in a json file
        progress = 0
        save_path = Path(f"../QTransformer_Results_and_Datasets/transformer_results/mods/{ExpID}")
        os.makedirs('../QTransformer_Results_and_Datasets/transformer_results/current_results', exist_ok = True)
        try:
            save_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"Directory for experiment ID '{ExpID}' already exists. Results may be overwritten. Make sure to save this results elsewhere" 
                    "or modify 'ExpID' to a new value if you want to keep previous results.")

        with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
            f.write('\nHyperparameters for Transformer\n')
            json.dump(p, f, indent=4)
            

        columns = [
            'idx', 'filter_config', 'test_auc', 'test_acc', 'val_auc', 'val_acc', 'train_auc',  '#params'
        ]

        channels_last = False           # Set to True if last dimension of datasets tensors match channels dimension
        SendToTelegramBool = True
        NExperiments = 20

        
        csv_path = os.path.join(save_path, 'results_grid_search.csv')
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, mode='a', header=True, index=False)


        # Load data
        train_dl, val_dl, test_dl, shape = qpctorch.data.get_medmnist_dataloaders(
            pixel=28, data_flag='dermamnist', batch_size=B, num_workers=4, pin_memory=True
        )

        # Obtain general settings regarding dataset shape
        num_channels = shape[-1] if channels_last else shape[0]
        img_size = shape[1]  # Assuming square images

        progress_levels = [0, 25, 50, 75, 100]
        # Grid search loop
        if SendToTelegramBool:
                SendToTelegram(progress = 0)   

        for idx in range(NExperiments):
            progress = int( 100* (idx+1)//NExperiments )
            if SendToTelegramBool and progress in progress_levels:
                SendToTelegram(progress = progress)                

            for filter_config in ['none', 'filter']:
                aux_save_path = Path(f"../QTransformer_Results_and_Datasets/transformer_results/current_results/grid_search{idx}")
                aux_save_path.mkdir(parents=True, exist_ok=True)

                print(f"\nPoint {idx} Training model with q_config set to: {filter_config}\n")

                model = qpctorch.quantum.vit.VisionTransformer(
                    img_size=shape[-1], num_channels=shape[0], num_classes=num_classes,
                    patch_size=p['patch_size'], hidden_size= shape[0]* p['patch_size']**2, num_heads=p['num_head'], Attention_N = p['Attention_N'],
                    num_transformer_blocks=p['num_transf'], attention_selection= p['attention_selection'], selection_amount= p['selection_amount'], special_cls = p['special_cls'], 
                    mlp_hidden_size=p['mlp_size'], quantum_mlp = False, dropout = p['dropout'], channels_last=False, quantum_classification = False,
                    paralel = p['paralel'], RD = p['RD'], q_stride = p['q_stride'], connectivity = p['connectivity']
                )

                # Train model
                test_auc, test_acc, val_auc, val_acc, train_auc, params = qpctorch.training.train_and_evaluate(
                    model, train_dl, val_dl, test_dl, num_classes=7,
                    learning_rate=p['learning_rate'], num_epochs=N, device=device, mapping=False,
                    res_folder=str(aux_save_path), hidden_size=p['hidden_size'], dropout=p['dropout'],
                    num_heads=p['num_head'], patch_size=p['patch_size'], num_transf=p['num_transf'],
                    mlp=p['mlp_size'], wd=p['weight_decay'], patience= p['patience'], scheduler_factor=p['scheduler_factor'], autoencoder=False
                )

                print(f"\nPoint {idx} finished training model with q_config set to: {filter_config}\n")

                # Save results
                row = {
                    'idx': idx, 
                        'filter_config': filter_config, 'test_auc': test_auc, 'test_acc': test_acc, 'val_auc': val_auc, 
                        'val_acc': val_acc, 'train_auc': train_auc,'#params': params
                        
                }

                pd.DataFrame([row], columns=columns).to_csv(
                    csv_path, mode='a', header=False, index=False
                )


        if SendToTelegramBool:
            SendToTelegram(csv_file = csv_path, columns = ['filter_config', 'test_auc'], title=NameOfExperiment)

    except Exception as e:
         SendToTelegram(progress = progress, error_message=str(e))
         print(str(e))

