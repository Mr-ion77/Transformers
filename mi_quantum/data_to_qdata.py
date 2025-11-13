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
    'ancilla' : 0,
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

# --- Helper function to ensure 'kernels' is always a list ---
def _normalize_to_list(item):
    """Ensures the input item is a list."""
    if isinstance(item, (list, tuple)):
        return item
    return [item]

# --- Helper function to save datasets ---
def _save_dataset(dataset_tensors, save_path, suffix, split_name):
    """Saves a single dataset tensor list to a file."""
    try:
        torch.save(dataset_tensors, save_path / f'quantum_{split_name}_dataset{suffix}.pt')
    except Exception as e:
        print(f"Warning: failed to save quantum_{split_name}_dataset{suffix}.pt: {e}")

# --- Main optimized function ---
def preprocess_and_save(
    B = p['batch_size'],
    DataLoaders = [None, None, None], # [train, val, test]
    kernels = {'none' : torch.nn.Identity()}, 
    save_path = f"../QTransformer_Results_and_Datasets/quantum_datasets",
    mode = 'standard',  # 'standard' or 'by_selected_patches'
    model1 = None,      # The model with .get_patches_by_attention
    p1 = None,          # Dictionary with patch parameters (e.g., p['p1'])
    num_channels = None, 
    flatten_extra_channels = False,
    p = None, 
    device = device
):
    """
    Preprocess datasets using one or more Quanvolution modules efficiently.
    ... [rest of docstring] ...

    Args:
        ... [original args] ...
        kernels (dict): Dictionary mapping names (str) to Quanvolution modules.
                      Example: {'patchwise': Quanvolution(...)}
        ...
    """

    # --- 1. Initialization ---
    if not isinstance(kernels, dict):
        raise TypeError(f"Expected 'kernels' to be a dictionary, but got {type(kernels)}")
        
    # NEW: Extract names and layers from the dictionary
    kernels_names = list(kernels.keys())
    kernels_list = list(kernels.values())
    
    num_kernels = len(kernels_list)
    
    # MODIFICATION: Initialize results as a dictionary
    results = {}
    
    dl_names = ['train', 'validation', 'test']

    # Validation for new mode
    if mode == 'by_selected_patches':
        if not all([model1, p1, num_channels is not None]):
            raise ValueError(
                "For 'by_selected_patches' mode, you must provide "
                "'model1', 'p1', and 'num_channels'."
            )
        print("Running in 'by_selected_patches' mode.")
        # Print reshape config once for clarity
        print(f"Reshape config: Flatten extra channels? {p1.get('1_flatten_extra_channels', 'N/A')}")
    else:
        print("Running in 'standard' mode.")

    all_quantum_datasets_tensors = [[] for _ in range(num_kernels)]
    last_processed_shapes = [None] * num_kernels

    # --- 2. Setup Save Directory & Hyperparameters ---
    save_path_quantum = Path(save_path)
    save_path_quantum.mkdir(parents=True, exist_ok=True)

    try:
        if p: # Only save if 'p' is provided
            with open(save_path_quantum / 'hyperparameters.json', 'w') as hf:
                json.dump(p, hf, indent=2) 
    except Exception as e:
        print(f"Warning: failed to save hyperparameters.json: {e}")

    # --- 3. Single Pass Data Processing ---
    for i, dl in enumerate(DataLoaders):
        dl_name = dl_names[i] if i < len(dl_names) else f"split_{i}"
        temp_data_batches = [[] for _ in range(num_kernels)]
        
        if dl is None: # Handle mock dataloaders being None
            print(f"Skipping {dl_name} as dataloader is None.")
            # Still need to append empty lists for shape consistency
            for q_idx in range(num_kernels):
                all_quantum_datasets_tensors[q_idx].append([])
            continue

        for images, labels, indices in tqdm(dl, desc=f"Processing {dl_name} split"):
            images = images.to(device)
            B_img = images.shape[0] # Batch size for this iteration
            
            with torch.no_grad():
                
                # Loop over each kernelsolution layer
                for q_idx, qlayer in enumerate(kernels_list):
                    
                    # --- THIS IS THE MODIFIED BLOCK ---
                    if mode == 'standard':
                        # Original behavior
                        processed_data = qlayer(images).cpu()

                    elif mode == 'by_selected_patches':
                        # Ensure num_channels is set for this mode
                        if num_channels is None:
                             raise ValueError("num_channels must be set for 'by_selected_patches' mode")

                        # 1. Get patches from model1
                        selected_patches = model1.get_patches_by_attention(x=images, paralel_branch=0)[0]
                        # selected_patches shape: (B_img, 1_selection_amount, C, patch_size, patch_size)

                        # 2. Reshape patches to fit kernelsolution input
                        aux_patches = selected_patches.view(
                            -1, 
                            num_channels, 
                            p1['1_patch_size'], 
                            p1['1_patch_size']
                        )  # Shape: (B * num_patches, C, patch_size, patch_size)

                        # 3. Apply the current kernelsolution layer
                        aux_patch_outs = qlayer(aux_patches)
                        # Shape: (B * num_patches, C * q, H_out, W_out) # C_out = C * (number of qubits measured = q)
                        
                        # Handle case where qlayer output might be flat
                        if aux_patch_outs.dim() == 2: # (B*num_patches, features)
                            # Assuming H_out and W_out are 1, and C*q is the feature dim
                             measured_qubits = aux_patch_outs.shape[1] // num_channels
                             aux_patch_outs = aux_patch_outs.view(
                                 aux_patch_outs.shape[0], # B * num_patches
                                 -1, # C * q
                                 1,  # H_out
                                 1   # W_out
                             )
                        else:
                            measured_qubits = aux_patch_outs.shape[-3] // num_channels # q = (C * q) /c, in theory haha

                        if hasattr(qlayer, 'channels_out'):
                            # Check that the number of output channels matches the number of measured qubits
                            assert len(qlayer.channels_out) == measured_qubits, \
                                f"The number of output channels ({len(qlayer.channels_out)}) must equal the number of measured qubits ({measured_qubits})."
                        else:
                            # Assume 1 measured qubit if channels_out attribute doesn't exist
                            assert 1 == measured_qubits, f"The number of output channels (1) must equal the number of measured qubits ({measured_qubits})."

                        # 4. Determine reshape dimensions
                        assert aux_patch_outs.shape[2] * aux_patch_outs.shape[3] == p1['1_patch_size']**2, \
                                f"An internal shape mismatch error happened during the layer: {qlayer}:\n( { aux_patch_outs.shape[2] * aux_patch_outs.shape[3]}) must equal the number the square of the patchsize ({p1['1_patch_size']**2})."
                        
                        if flatten_extra_channels:

                            shape_to_reshape_toQ = (
                                B_img, 
                                p1['1_selection_amount'], 
                                num_channels * measured_qubits * p1['1_patch_size']**2
                            )
                        else:
                            shape_to_reshape_toQ = (
                                B_img,
                                p1['1_selection_amount'] * measured_qubits,
                                num_channels * p1['1_patch_size']**2
                            )

                        processed_data = aux_patch_outs.view(shape_to_reshape_toQ).cpu()

                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    # Store the shape for later (shape of a single item, post-batch)
                    if processed_data.nelement() > 0:
                        last_processed_shapes[q_idx] = processed_data.shape[1:]

                    # Append (data_batch, label_batch)
                    temp_data_batches[q_idx].append((processed_data, labels.cpu()))
                    
                    # Print shapes for the *first batch* of the *first split*
                    if i == 0 and all(len(b) == 1 for b in temp_data_batches): # if first batch
                        if mode == 'by_selected_patches':
                            print(f"\n--- Debug Shapes (q_idx: {q_idx}, name: {kernels_names[q_idx]}, split: {dl_name}, batch 0) ---")
                            print(f"Shape out of q-convolution (aux_patch_outs): {aux_patch_outs.shape}")
                            print(f"Shape after reshape (processed_data batch): {processed_data.shape}")
                            print(f"Stored item shape (last_processed_shapes): {processed_data.shape[1:]}")
                            print(f"--------------------------------------------------")
                        else: # Standard mode
                            print(f"\n--- Debug Shapes (q_idx: {q_idx}, name: {kernels_names[q_idx]}, split: {dl_name}, batch 0) ---")
                            print(f"Shape after q-convolution (processed_data batch): {processed_data.shape}")
                            print(f"Stored item shape (last_processed_shapes): {processed_data.shape[1:]}")
                            print(f"--------------------------------------------------")
                            
        # --- 4. Consolidate Batches ---

        for q_idx in range(num_kernels):
            if not temp_data_batches[q_idx]:
                print(f"Warning: No data processed for {dl_name}, kernels {kernels_names[q_idx]} (idx {q_idx}).")
                all_quantum_datasets_tensors[q_idx].append([])
                continue

            all_quantums_split = torch.cat([data[0] for data in temp_data_batches[q_idx]], dim=0)
            all_labels_split = torch.cat([data[1] for data in temp_data_batches[q_idx]], dim=0)

            dataset_tensor_list = list(zip(all_quantums_split, all_labels_split))
            all_quantum_datasets_tensors[q_idx].append(dataset_tensor_list)

    # --- 5. Create DataLoaders and Save Datasets ---

    for q_idx in range(num_kernels):

        dataset_name_suffix = kernels_names[q_idx] 
        current_kernels_tensors = all_quantum_datasets_tensors[q_idx]

        if not current_kernels_tensors or not any(split for split in current_kernels_tensors):
            print(f"Warning: No data found for kernels '{dataset_name_suffix}' (idx {q_idx}). Skipping.")
            
            results[dataset_name_suffix] = None
            continue

        # Build dataloaders
        Quantums = Data.create_dataloaders(
            data_dir=None,
            batch_size=B,
            channels_last=p.get('channels_last', False) if p else False,
            tensors=current_kernels_tensors,
            transforms={'train': None, 'val': None, 'test': None}
        )

        # Save the datasets using the new suffix
        # Ensure that the number of splits matches the save calls
        num_splits = len(current_kernels_tensors)
        _save_dataset(current_kernels_tensors[0], save_path_quantum, dataset_name_suffix, 'train')
        if num_splits > 1:
            _save_dataset(current_kernels_tensors[1], save_path_quantum, dataset_name_suffix, 'val')
        if num_splits > 2:
            _save_dataset(current_kernels_tensors[2], save_path_quantum, dataset_name_suffix, 'test')
        # Add more if you have more splits
        if num_splits > 3: 
            for i in range(3, num_splits):
                 _save_dataset(current_kernels_tensors[i], save_path_quantum, dataset_name_suffix, f'split_{i}')

        
        # MODIFICATION: Assign dataloaders to dictionary using key
        results[dataset_name_suffix] = [ *Quantums, last_processed_shapes[q_idx]]

    return results
        

if __name__ == 'main':
    preprocess_and_save([train,val,test], Quanvolution)



















