import os
import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import json
import medmnist
from tqdm import tqdm
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
num_channels = 3

class MinMaxScalePerChannel:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        # Assume tensor shape: (C, H, W)
        c, h, w = tensor.shape
        scaled = torch.empty_like(tensor)
        
        for ch in range(c):
            ch_min = tensor[ch].min()
            ch_max = tensor[ch].max()

            if ch_max == ch_min:
                # Avoid division by zero; fill with min_val
                scaled[ch] = torch.full((h, w), self.min_val, dtype=tensor.dtype, device=tensor.device)
            else:
                ch_scaled = (tensor[ch] - ch_min) / (ch_max - ch_min)
                scaled[ch] = ch_scaled * (self.max_val - self.min_val) + self.min_val

        return scaled

train_transforms = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

valid_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

class SubsetAsDataset(Dataset):
    def __init__(self, subset: Subset):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def split_dataset_by_label(dataset, majority_label=5):
    majority_indices = []
    others_indices = []

    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == majority_label:
            majority_indices.append(i)
        else:
            others_indices.append(i)

    majority_subset = SubsetAsDataset( Subset(dataset, majority_indices) )
    others_subset = SubsetAsDataset( Subset(dataset, others_indices) )

    return majority_subset, others_subset

def relabel_dataset(dataset, majority_label, majority_or_others='majority'):
    """Relabels the dataset and returns a new IndexedTensorDataset."""
    new_images = []
    new_labels = []
    new_indices = []

    for i in range(len(dataset)):
        image, old_label, index = dataset[i]
        if isinstance(old_label, torch.Tensor):
            old_label = old_label.item()
        if majority_or_others == 'majority':
            new_label = 1 if old_label == majority_label else 0
        elif majority_or_others == 'others':
            new_label = old_label if old_label < majority_label else old_label - 1
            new_label = new_label[0]
        else:
            raise ValueError("majority_or_others must be 'majority' or 'others'")
        
        new_images.append(image)
        new_labels.append(new_label)
        new_indices.append(index)

    # Convertir a tensores
    image_tensor = torch.stack(new_images)
    label_tensor = torch.tensor(new_labels)
    index_tensor = torch.tensor(new_indices)

    print(f"Relabeled dataset: {len(new_images)} images, {len(new_labels)} labels, {len(new_indices)} indices. Also, {set(new_labels)} unique labels.")

    return TensorDataset(image_tensor, label_tensor, index_tensor)


def datasets_to_dataloaders( datasets, **dataloader_kwargs):
    """Returns dataloaders for the given datasets"""
    shape = datasets[0]['data'][0][0].shape  # Assuming all datasets have the same shape

    dataloader_list = []
    for dataset in datasets:
        dataloader_list.append( DataLoader(dataset['data'], shuffle = dataset['split'] == 'train', **dataloader_kwargs)     )

    return *dataloader_list, shape


def get_medmnist_dataloaders(pixel: int = 28, data_flag: str = 'breastmnist', extra_tr_without_trans = False, **dataloader_kwargs) -> tuple:
    """Returns dataloaders for the MedMNIST dataset"""
    # Transformaciones
    n_channels = 3

    valid_transform = valid_transforms
    
    train_transform = train_transforms
    
    info = medmnist.INFO[data_flag]                            # Estas dos líneas permiten reutilizar el código más fácilmente
    DataClass = getattr(medmnist, info['python_class'])        # para distintos datasets

    class IndexedMedMNIST(DataClass):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            return img, label, idx

        def clone(self):
            # Reconstruct the dataset with the same parameters
            new_instance = IndexedMedMNIST(
                split=self.split,
                transform=self.transform,
                target_transform=self.target_transform,
                download=False  # prevent re-downloading
            )

            # Manually copy relevant attributes
            if hasattr(self, 'data'):
                try:
                    new_instance.data = self.data.clone()
                except AttributeError:
                    new_instance.data = self.data.copy()  # For NumPy arrays

            if hasattr(self, 'labels'):
                try:
                    new_instance.labels = self.labels.clone()
                except AttributeError:
                    new_instance.labels = self.labels.copy()  # For NumPy arrays

            return new_instance

    train_dataset = { 'data': IndexedMedMNIST(split='train', transform=train_transform, download=True, size = pixel), 'split':'train'}    # Construye el dataset a partir de la clase obtenida
    valid_dataset = { 'data': IndexedMedMNIST(split='val', transform=valid_transform, download=True, size = pixel), 'split':'val'} 
    test_dataset = { 'data': IndexedMedMNIST(split='test', transform=valid_transform, download=True, size = pixel), 'split':'test'} 
    print(f"Loaded MedMNIST dataset '{data_flag}' with image size {pixel}x{pixel} and {n_channels} channels.")
    if extra_tr_without_trans:
        no_trans_train_dataset = { 'data': IndexedMedMNIST(split='train', transform=valid_transform, download=True, size = pixel),'split':'train'} 
        return datasets_to_dataloaders( [no_trans_train_dataset, train_dataset, valid_dataset, test_dataset], **dataloader_kwargs)

    return datasets_to_dataloaders([train_dataset, valid_dataset, test_dataset], **dataloader_kwargs)

# Function for: Quantum preprocessed datasets

q_train_transforms = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

q_valid_transforms = transforms.Compose([
    ])

class TransformedTensorDataset(torch.utils.data.Dataset):
    """Custom dataset that applies a transform to each item."""
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        image, label, *thing =  self.tensors[index] 
        if self.transform:
            image = self.transform(image)
        if thing:
            print(f"image shape: {image.shape}, label: {label.shape}, thing: {len(thing)}")
            raise ValueError("Something unexpected in tensors shape, consider revising the code.")
        return image, label, index

    def __len__(self):
        return len(self.tensors)

def load_numpy_data(data_dir, channels_last = True):
    # Load the .npy files
    train_images = torch.from_numpy( np.load(os.path.join(data_dir, 'q_train_images.npy')) ).float()
    val_images = torch.from_numpy( np.load(os.path.join(data_dir, 'q_val_images.npy')) ).float()
    test_images = torch.from_numpy( np.load(os.path.join(data_dir, 'q_test_images.npy')) ).float()

    if channels_last: # Change shape to [N, C, H, W]
        train_images = train_images.permute(0, 3, 1, 2)
        val_images = val_images.permute(0, 3, 1, 2)
        test_images = test_images.permute(0, 3, 1, 2)

    train_labels = np.load(os.path.join(data_dir, 'q_train_labels.npy'))
    val_labels = np.load(os.path.join(data_dir, 'q_val_labels.npy'))
    test_labels = np.load(os.path.join(data_dir, 'q_test_labels.npy'))

    # Create list of (image, label) pairs
    train_tensor = list(zip(train_images, torch.from_numpy(train_labels).long()))
    val_tensor = list(zip(val_images, torch.from_numpy(val_labels).long()))
    test_tensor = list(zip(test_images, torch.from_numpy(test_labels).long()))

    return train_tensor, val_tensor, test_tensor


def create_dataloaders(data_dir, batch_size=32, channels_last = True, shuffle=True, tensors = None, transforms={'train': q_train_transforms, 'val': q_valid_transforms, 'test': q_valid_transforms}):
    if tensors is not None:
        train_tensor, val_tensor, test_tensor = tensors
    else:
        train_tensor, val_tensor, test_tensor = load_numpy_data(data_dir, channels_last= channels_last)

    # Wrap with transformed datasets
    train_dataset = TransformedTensorDataset(train_tensor, transform=transforms['train'])
    val_dataset = TransformedTensorDataset(val_tensor, transform=transforms['val'])
    test_dataset = TransformedTensorDataset(test_tensor, transform=transforms['test'])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_tensor[0][0].shape  # Return the shape of the images


# Main functions to use while making experiments


def _save_dataset(dataset_tensors, save_path, suffix, split_name):
    """Saves a single dataset tensor list to a file."""
    try:
        torch.save(dataset_tensors, save_path / f'quantum_{split_name}_dataset{suffix}.pt')
    except Exception as e:
        print(f"Warning: failed to save quantum_{split_name}_dataset{suffix}.pt: {e}")

# --- Main optimized function ---
def preprocess_and_save(
    B = 256,
    DataLoaders = [None, None, None], # [train, val, test]
    kernels = {'none' : torch.nn.Identity()}, 
    save_path = f"../QTransformer_Results_and_Datasets/quantum_datasets",
    mode = 'standard',  # 'standard' or 'by_selected_patches'
    model1 = None,      # The model with .get_patches_by_attention
    p1 = None,          # Dictionary with patch parameters (e.g., p['p1'])
    num_channels = None, 
    channels_last = False,
    flatten_extra_channels = False,
    device = 'cpu',
    flatten = True,
    concatenate_original = False
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

    # --- 3. Single Pass Data Processing ---
    for i, dl in enumerate(DataLoaders):
        dl_name = dl_names[i] if i < len(dl_names) else f"split_{i}"
        temp_data_batches = [[] for _ in range(num_kernels)]
        
        if dl is None: # Handle mock dataloaders being None
            print(f"Skipping {dl_name} as dataloader is None.")
            # Still need to append empty lists for shape consistency
            for q_idx in range(num_kernels):
                all_quantum_datasets_tensors[q_idx].append( ([], []) )
            continue

        for images, labels, indices in tqdm(dl, desc=f"Processing {dl_name} split"):
            images = images.to(device)
            B_img = images.shape[0] # Batch size for this iteration
            
            with torch.no_grad():
                
                # Loop over each kernelsolution layer
                for q_idx, qlayer in enumerate(kernels_list):

                    qlayer.to(device)
                    images.to(device)
                    not_none_bool = kernels_names[q_idx] != 'none' 

                    # --- THIS IS THE MODIFIED BLOCK ---
                    if mode == 'standard':
                        # Original behavior
                        processed_data = qlayer(images).cpu()
                        if concatenate_original and not_none_bool:
                            processed_data = torch.cat([images, processed_data], dim = 1)  #Concatenate original channels to quantum processed ones

                    elif mode == 'by_selected_patches':
                        # Ensure num_channels is set for this mode
                        if num_channels is None:
                             raise ValueError("num_channels must be set for 'by_selected_patches' mode")

                        C, P = num_channels, p1['1_patch_size']

                        # 1. Get patches from model1
                        aux_patches, selected_indices = model1.get_patches_by_attention(x=images, parallel_branch=0) #Until here everything okay

                        # aux_patches = model1.get_selected_pixel_patches(images, selected_indices, quantum_channels = 0, originals = concatenate_original)
                        # selected_patches shape: (B_img, 1_selection_amount, C, patch_size, patch_size)
                        # 2. Reshape patches to fit kernelsolution input
                        aux_patches = aux_patches.view(-1, C, P, P)  # Shape: (B * num_patches, C, patch_size, patch_size)

                        aux_shape = (B_img, p1['1_selection_amount'], -1, C * P * P)
                        # 3. Apply the current kernelsolution layer and undo the change
                        aux_patch_outs = qlayer(aux_patches).reshape( aux_shape )
                        measured_qubits = aux_patch_outs.shape[2] # q = (C * q) /c, in theory haha

                        # Shape: (B , num_patches, C * q, H_out, W_out) # C_out = C * (number of qubits measured = q) .permute(0, 2, 1, 3, 4).contiguous().view(B, q * C, H_out, W_out)
                        if concatenate_original and not_none_bool:
                            aux_patch_outs = torch.cat([aux_patches.reshape(aux_shape), aux_patch_outs.to(aux_patches.device)], dim = 2) # Concatenate original patches to quantum processed ones
                            # Shape: (B, num_patches, (q+1) , C * H_out * W_out)

                        Q = measured_qubits + concatenate_original * not_none_bool

                        if hasattr(qlayer, 'channels_out'):
                            # Check that the number of output channels matches the number of measured qubits
                            assert (len(qlayer.channels_out) ) == (measured_qubits), \
                                f"The number of output channels ({len(qlayer.channels_out)}) must equal the number of measured qubits ({measured_qubits})."
                        else:
                            # Assume 1 measured qubit if channels_out attribute doesn't exist
                            assert 1 == measured_qubits, f"The number of output channels (1) must equal the number of measured qubits ({measured_qubits})."

                        # 4. Determine reshape dimensions
                        assert aux_patch_outs.shape[-1] // C == P**2, \
                                f"An internal shape mismatch error happened during the layer: {qlayer}:\n( { aux_patch_outs.shape[-1] // C}) must equal the square of the patchsize ({P**2})."
                        
                        if flatten:
                            # Flatten spatial dimensions
                            if flatten_extra_channels:
                                shape_to_reshape_toQ = (
                                    B_img, p1['1_selection_amount'], 
                                    num_channels * Q * P**2
                                )
                            else:
                                aux_patch_outs = aux_patch_outs.transpose(1, 2)
                                shape_to_reshape_toQ = (
                                    B_img, p1['1_selection_amount'] * Q,
                                    num_channels * P**2
                                )
                        
                        else:
                            # Keep spatial dimensions
                            if flatten_extra_channels:
                                shape_to_reshape_toQ = (
                                    B_img, p1['1_selection_amount'], 
                                    num_channels * Q, 
                                    P, P
                                )
                            else:
                                aux_patch_outs = aux_patch_outs.transpose(1, 2)
                                shape_to_reshape_toQ = (
                                    B_img, p1['1_selection_amount'] * Q,
                                    num_channels, P, P
                                )

                        processed_data = aux_patch_outs.reshape(shape_to_reshape_toQ).cpu()

                        if not flatten:
                            assert list(processed_data.shape[-3:]) == [num_channels,P,P], \
                                f"Reshape error: got shape {list(processed_data.shape[-3:])}, expected {[num_channels,P,P]}"
                        else:
                            assert processed_data.shape[-1] == num_channels*P*P, \
                                f"Reshape error: got shape {processed_data.shape[-1]}, expected {num_channels*P*P}"

                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    # Store the shape for later (shape of a single item, post-batch)
                    if processed_data.nelement() > 0:
                        last_processed_shapes[q_idx] = processed_data.shape[1:]

                    # Append (data_batch, label_batch)
                    temp_data_batches[q_idx].append( (processed_data, labels.cpu()) )
                    
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
                all_quantum_datasets_tensors[q_idx].append( ([], []) )
                continue

            all_quantums_split = torch.cat([data[0] for data in temp_data_batches[q_idx]], axis = 0)
            all_labels_split = torch.cat([data[1] for data in temp_data_batches[q_idx]], axis = 0)

            print(f"Hypershape for {dl_name}, kernels {kernels_names[q_idx]} (idx {q_idx}): {all_quantums_split.shape}, Labels shape: {all_labels_split.shape}")

            all_quantum_datasets_tensors[q_idx].append( list(zip(all_quantums_split, all_labels_split) ) )
        

    # --- 5. Create DataLoaders and Save Datasets ---

    for q_idx in range(num_kernels):

        dataset_name_suffix = kernels_names[q_idx] 
        current_kernels_tensors = all_quantum_datasets_tensors[q_idx]

        if not current_kernels_tensors or not any(split for split in current_kernels_tensors):
            print(f"Warning: No data found for kernels '{dataset_name_suffix}' (idx {q_idx}). Skipping.")
            
            results[dataset_name_suffix] = None
            continue

        # Build dataloaders
        Quantums = create_dataloaders(
            data_dir = None,
            batch_size = B,
            channels_last = channels_last,
            tensors = current_kernels_tensors,
            transforms = {'train': None, 'val': None, 'test': None},
            shuffle = False
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
        results[dataset_name_suffix] = Quantums
        print(f"Saved quantum datasets for kernels '{dataset_name_suffix}' at {save_path_quantum}")

    for k in results.keys():
        print(f"Kernel {k} got a final shape of {results[k][-1]}")

    return results


def cut_extra_channels_from_latents( Latents, i , channels_out):    
    """
    Latents: List of dataloaders [train, val, test]
    i: int, number of channels to keep
    channels_out: int, total channels
    """
    
    none_latent = Latents.get('none', None)
    patchwise_latent = Latents.get('patchwise', None)

    if patchwise_latent is None:
        print("No 'patchwise' latent found. Returning original Latents.")
        return Latents
    
    else:

        getbatchsize = next(iter(patchwise_latent[0]))
        B = getbatchsize[0].shape[0]  # Assuming shape is (B, C, H, W) or (B, C)
        new_patchwise_latent = []

        for split_idx, split in enumerate(patchwise_latent):

            if not isinstance(split, torch.utils.data.DataLoader):
                print(f"Split {split_idx} is not a DataLoader. Skipping.")
                continue

            new_data = []
            new_labels = []
            for samples, labels, idx in split:

                shape = samples.shape  # (B, total_channels, H, W) or (B, total_channels)
                new_samples = samples[:, : ( i*shape[1]// channels_out), ... ]  # Keep only the first i channels

                new_data.extend(new_samples)
                new_labels.extend(labels)
                shape_after = new_samples.shape

            new_patchwise_latent.append( list( zip(torch.stack(new_data), torch.tensor(new_labels)) ))

        new_patchwise_latent_dataloaders = create_dataloaders(
            data_dir= None,
            batch_size= B,
            channels_last= False,
            tensors = new_patchwise_latent,
            transforms = {'train': None, 'val': None, 'test': None}
        ) 

        NewLatents = {
            'none' : none_latent,
            'patchwise' : new_patchwise_latent_dataloaders
        }

        print("Cut extra channels from 'patchwise' latent.", f"\nDifference in shapes: {shape} -> {shape_after}")

        if NewLatents is None:
            print("Something went wrong")
            raise ValueError("Something went wrong")
        
    return NewLatents

        



















