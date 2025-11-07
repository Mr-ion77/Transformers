import torch
from mi_quantum.quantum.pennylane_backend import QuantumLayer
from mi_quantum import data as Data
from mi_quantum.quantum.quanvolution import QuantumKernel, QuantumConv2D
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

graphs = {'king_2_ancilla': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                     [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4],
                     [4, 9], [0, 9], [2, 9], [8, 9], [6, 9],
                     [1, 10], [5, 10], [7, 10], [3, 10], [4, 10]],
          'king_1_ancilla': [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                     [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4],
                     [4, 9], [0, 9], [2, 9], [8, 9], [6, 9]],
          'king'          : [[0, 2], [2, 8], [8, 6], [6, 0], [1, 5], [5, 7], [7, 3], [3, 1], [0, 4],
                            [1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [8, 4]]
}

p = {
    'num_qubits': 9,
    'ancilla' : 1,
    'entangle': True,
    'trainBool': False,
    'connectivity': graphs['king'],
    'patch_size': 3,
    'stride' : 1,
    'padding': 1,
    'channels_last' : False,
    'batch_size' : 32,
    'pad_filler' : 'median'
}


# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Quanvolution = QuantumConv2D(patch_size=p['patch_size'], stride=p['stride'], padding=p['padding'], channels_last = p['channels_last'],
                             graph = p['connectivity'], channels_out= [4], ancilla = p['ancilla'], pad_filler= p['pad_filler']).to(device)

train, val, test, shape = Data.get_medmnist_dataloaders(pixel = 28, data_flag = 'dermamnist', batch_size=p['batch_size'], num_workers=4, pin_memory=True)

def preprocess_and_save(B = p['batch_size'],DataLoaders = [train, val, test], quanv = Quanvolution, save_path = f"../QTransformer_Results_and_Datasets/transformer_results/current_results/quantum_datasets"):

    QuantumDatasetsTensors = [] 

    for i, dl in enumerate(DataLoaders):
        all_quantums = []
        all_labels = []
        all_indices = []

        for images, labels, indices in tqdm(dl, desc= f"Processing { ['train', 'validation', 'test'][i]} split: \n"):
            images = images.to(device)
            with torch.no_grad():

                B_img = images.shape[0]
                quantum_imgs = quanv(images).cpu()
                all_quantums.extend( quantum_imgs )          
                all_labels.extend( labels )


        all_labels = torch.tensor(all_labels)
        all_quantums = torch.stack(all_quantums)
        QuantumDatasetsTensors.append( list(zip(all_quantums ,all_labels))  )
        
        Quantums = Data.create_dataloaders(data_dir = None, batch_size = B, channels_last = channels_last,
                                        tensors = QuantumDatasetsTensors, transforms = {'train': None, 'val': None, 'test': None}
                                        )

        # Use the provided save_path (default points to ../QTransformer_Results_and_Datasets/...)
        save_path_quantum = Path(save_path)
        save_path_quantum.mkdir(parents=True, exist_ok=True)

        # Save hyperparameters dictionary `p` alongside the quantum dataset tensors
        try:
            with open(save_path_quantum / 'hyperparameters.json', 'w') as hf:
                json.dump(p, hf, indent=2)
        except Exception as e:
            print(f"Warning: failed to save hyperparameters.json: {e}")

        torch.save(QuantumDatasetsTensors[0], save_path_quantum / 'quantum_train_dataset.pt')
        torch.save(QuantumDatasetsTensors[1], save_path_quantum / 'quantum_val_dataset.pt')
        torch.save(QuantumDatasetsTensors[2], save_path_quantum / 'quantum_test_dataset.pt')

        Quantums.append( quantum_imgs.shape[1:] )


    return Quantums
        

if __name__ == 'main':
    preprocess_and_save([train,val,test], Quanvolution)



















