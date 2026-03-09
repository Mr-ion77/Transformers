import os
import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import json
import medmnist
import random
import torchvision.transforms.functional as F
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


class RandomRotationChannelWiseMedian:
    def __init__(self, degrees, channels_last = True):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.channels_last = channels_last

    def __call__(self, img):
        # Convert to numpy to calculate medians
        img_np = np.array(img)
        # Use float for float-valued tensors (range [0,1]), int for uint8 images (range [0,255])
        cast = float if img_np.dtype.kind == 'f' else int
        
        if img_np.ndim == 2:
            # Grayscale: shape is (H, W)
            fill_values = cast(np.median(img_np))
        elif img_np.ndim == 3:
            # RGB/Multichannel: shape is (H, W, C) or (C, H, W)
            # Calculate median for each channel independently
            if self.channels_last:
                fill_values = tuple(cast(np.median(img_np[:, :, i])) for i in range(img_np.shape[2]))
            else:
                fill_values = tuple(cast(np.median(img_np[i, :, :])) for i in range(img_np.shape[0]))
        else:
            fill_values = 0 # Fallback

        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # F.rotate handles a tuple for the fill parameter to fill per-channel
        return F.rotate(img, angle, fill=fill_values)

# --- Updated Pipeline ---
train_transforms = transforms.Compose([
    RandomRotationChannelWiseMedian(90),
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
        RandomRotationChannelWiseMedian(90, channels_last=False),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

q_valid_transforms = transforms.Compose([
    ])

class TransformedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        # tensors is now a tuple: (images_tensor, labels_tensor)
        self.images, self.labels = tensors
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, index

    def __len__(self):
        return len(self.images)

def load_numpy_data(data_dir, channels_last=True):
    # Load the .npy files directly into PyTorch tensors
    train_images = torch.from_numpy(np.load(os.path.join(data_dir, 'q_train_images.npy'))).float()
    val_images = torch.from_numpy(np.load(os.path.join(data_dir, 'q_val_images.npy'))).float()
    test_images = torch.from_numpy(np.load(os.path.join(data_dir, 'q_test_images.npy'))).float()

    if channels_last: # Change shape from [N, H, W, C] to [N, C, H, W]
        train_images = train_images.permute(0, 3, 1, 2)
        val_images = val_images.permute(0, 3, 1, 2)
        test_images = test_images.permute(0, 3, 1, 2)

    train_labels = torch.from_numpy(np.load(os.path.join(data_dir, 'q_train_labels.npy'))).long()
    val_labels = torch.from_numpy(np.load(os.path.join(data_dir, 'q_val_labels.npy'))).long()
    test_labels = torch.from_numpy(np.load(os.path.join(data_dir, 'q_test_labels.npy'))).long()

    # RETURN TUPLES OF MASSIVE TENSORS, NOT LISTS OF TUPLES
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def create_dataloaders(data_dir, batch_size=32, channels_last=True, shuffle=True, tensors=None, transforms={'train': q_train_transforms, 'val': q_valid_transforms, 'test': q_valid_transforms}, num_workers=0, pin_memory=False):
    if tensors is not None:
        train_tensor, val_tensor, test_tensor = tensors
    else:
        # This will now safely return (images_tensor, labels_tensor) format
        train_tensor, val_tensor, test_tensor = load_numpy_data(data_dir, channels_last=channels_last)

    # Wrap with transformed datasets
    train_dataset = TransformedTensorDataset(train_tensor, transform=transforms['train'])
    val_dataset = TransformedTensorDataset(val_tensor, transform=transforms['val'])
    test_dataset = TransformedTensorDataset(test_tensor, transform=transforms['test'])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # train_tensor[0] is the entire images block of shape [N, C, H, W]. 
    # .shape[1:] returns [C, H, W] safely.
    image_shape = train_tensor[0].shape[1:] 

    return train_loader, val_loader, test_loader, image_shape

# Main functions to use while making experiments

def _save_dataset(dataset_tensors, save_path, suffix, split_name):
    """Saves a single dataset tensor list to a file."""
    try:
        torch.save(dataset_tensors, save_path / f'quantum_{split_name}_dataset{suffix}.pt')
    except Exception as e:
        print(f"Warning: failed to save quantum_{split_name}_dataset{suffix}.pt: {e}")

# --- Main optimized function ---
def preprocess_and_save(
    B=256,
    DataLoaders=[None, None, None],  # [train, val, test]
    kernels={'none': torch.nn.Identity()},
    save_path="../QTransformer_Results_and_Datasets/quantum_datasets",
    mode='standard',  # 'standard' or 'by_selected_patches'
    model1=None,      # The model with .get_patches_by_attention
    p1=None,          # Dictionary with patch parameters (e.g., p['p1'])
    num_channels=None,
    channels_last=False,
    flatten_extra_channels=False,
    device='cpu',
    flatten=True,
    concatenate_original=False
):
    if not isinstance(kernels, dict):
        raise TypeError(f"Expected 'kernels' to be a dictionary, but got {type(kernels)}")

    kernels_names = list(kernels.keys())
    kernels_list = list(kernels.values())
    num_kernels = len(kernels_list)
    results = {}
    dl_names = ['train', 'validation', 'test']

    if mode == 'by_selected_patches':
        if not all([model1, p1, num_channels is not None]):
            raise ValueError("For 'by_selected_patches' mode, you must provide 'model1', 'p1', and 'num_channels'.")
        print("Running in 'by_selected_patches' mode.")
        print(f"Reshape config: Flatten extra channels? {p1.get('1_flatten_extra_channels', 'N/A')}")
    else:
        print("Running in 'standard' mode.")

    all_quantum_datasets_tensors = [[] for _ in range(num_kernels)]
    save_path_quantum = Path(save_path)
    save_path_quantum.mkdir(parents=True, exist_ok=True)

    for i, dl in enumerate(DataLoaders):
        dl_name = dl_names[i] if i < len(dl_names) else f"split_{i}"
        temp_data_batches = [([], []) for _ in range(num_kernels)]  # Store (images_list, labels_list) separately

        if dl is None:
            print(f"Skipping {dl_name} as dataloader is None.")
            for q_idx in range(num_kernels):
                all_quantum_datasets_tensors[q_idx].append((None, None))
            continue

        for images, labels, indices in tqdm(dl, desc=f"Processing {dl_name} split"):
            images = images.to(device)
            B_img = images.shape[0]

            with torch.no_grad():
                for q_idx, qlayer in enumerate(kernels_list):
                    qlayer.to(device)
                    not_none_bool = kernels_names[q_idx] != 'none'

                    if mode == 'standard':
                        processed_data = qlayer(images).cpu()
                        if concatenate_original and not_none_bool:
                            processed_data = torch.cat([images.cpu(), processed_data], dim=1)

                    elif mode == 'by_selected_patches':
                        C, P = num_channels, p1['1_patch_size']
                        aux_patches, *selected_indices = model1.get_patches_by_attention(x=images, parallel_branch=0)
                        aux_patches = aux_patches.view(-1, C, P, P)

                        aux_shape = (B_img, p1['1_selection_amount'], -1, C * P * P)
                        aux_patch_outs = qlayer(aux_patches).reshape(aux_shape)
                        measured_qubits = aux_patch_outs.shape[2]

                        if concatenate_original and not_none_bool:
                            aux_patch_outs = torch.cat([aux_patches.reshape(aux_shape).cpu(), aux_patch_outs.cpu()], dim=2)

                        Q = measured_qubits + (1 if concatenate_original and not_none_bool else 0)

                        if flatten:
                            if flatten_extra_channels:
                                shape_to_reshape_toQ = (B_img, p1['1_selection_amount'], num_channels * Q * P**2)
                            else:
                                aux_patch_outs = aux_patch_outs.transpose(1, 2)
                                shape_to_reshape_toQ = (B_img, p1['1_selection_amount'] * Q, num_channels * P**2)
                        else:
                            if flatten_extra_channels:
                                shape_to_reshape_toQ = (B_img, p1['1_selection_amount'], num_channels * Q, P, P)
                            else:
                                aux_patch_outs = aux_patch_outs.transpose(1, 2)
                                shape_to_reshape_toQ = (B_img, p1['1_selection_amount'] * Q, num_channels, P, P)

                        processed_data = aux_patch_outs.reshape(shape_to_reshape_toQ).cpu()

                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    # Append to our separated lists
                    temp_data_batches[q_idx][0].append(processed_data)
                    temp_data_batches[q_idx][1].append(labels.cpu())

        # Consolidate Batches safely
        for q_idx in range(num_kernels):
            if not temp_data_batches[q_idx][0]:
                print(f"Warning: No data processed for {dl_name}, kernel {kernels_names[q_idx]}.")
                all_quantum_datasets_tensors[q_idx].append((None, None))
                continue
            
            try:
                all_quantums_split = torch.cat(temp_data_batches[q_idx][0], dim=0)
                all_labels_split = torch.cat(temp_data_batches[q_idx][1], dim=0)
            except RuntimeError as e:
                print(f"\nFATAL: Tensor concatenation failed for {dl_name}, kernel {kernels_names[q_idx]}.")
                shapes = [t.shape for t in temp_data_batches[q_idx][0][-3:]] # Show last 3 shapes to spot the odd one out
                print(f"Shapes of last few batches: {shapes}")
                raise e # Kill the script. Do not proceed with corrupted data.

            # Store as a raw tuple of (Images_Tensor, Labels_Tensor) instead of a massive list of individual tuples
            all_quantum_datasets_tensors[q_idx].append((all_quantums_split, all_labels_split))

    # Create DataLoaders and Save Datasets
    for q_idx in range(num_kernels):
        dataset_name_suffix = kernels_names[q_idx]
        current_kernels_tensors = all_quantum_datasets_tensors[q_idx]

        if not current_kernels_tensors or current_kernels_tensors[0] == (None, None):
            results[dataset_name_suffix] = None
            continue

        # Pass the raw tuple of tensors directly
        Quantums = create_dataloaders(
            data_dir=None,
            batch_size=B,
            channels_last=channels_last,
            tensors=current_kernels_tensors,
            transforms={'train': q_train_transforms, 'val': None, 'test': None},
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        print(f"Created dataloaders for kernel '{dataset_name_suffix}' with {len(Quantums[0].dataset)} training samples, {len(Quantums[1].dataset)} validation samples, and {len(Quantums[2].dataset)} test samples.")
        print(f"Sample shapes for kernel '{dataset_name_suffix}':\n - Train: {Quantums[-1]}")

        splits = ['train', 'val', 'test']
        for split_idx, split_name in enumerate(splits):
            if split_idx < len(current_kernels_tensors) and current_kernels_tensors[split_idx][0] is not None:
                _save_dataset(current_kernels_tensors[split_idx], save_path_quantum, dataset_name_suffix, split_name)

        results[dataset_name_suffix] = Quantums
        print(f"Saved quantum datasets for kernel '{dataset_name_suffix}' at {save_path_quantum}")

    return results


def cut_extra_channels_from_latents(Latents, i, channels_out):    
    """
    Latents: Dict of dataloaders {'none': [...], 'patchwise': [train, val, test, shape]}
    i: int, number of target channels multiplier
    channels_out: int, total channels in current representation
    """
    none_latent = Latents.get('none', None)
    patchwise_latent = Latents.get('patchwise', None)

    if patchwise_latent is None:
        print("No 'patchwise' latent found. Returning original Latents.")
        return Latents

    # Safely get batch size without exhausting an iterator blindly
    try:
        getbatchsize = next(iter(patchwise_latent[0]))
        B = getbatchsize[0].shape[0]
    except StopIteration:
        print("DataLoader is empty. Returning original Latents.")
        return Latents

    new_patchwise_tensors = []
    original_shape = None
    final_shape = None

    # We only want to iterate over the DataLoaders (the first 3 elements), 
    # ignoring the shape element at the end of the list.
    for split_idx, split in enumerate(patchwise_latent[:3]):
        
        if not isinstance(split, torch.utils.data.DataLoader):
            print(f"Split {split_idx} is not a DataLoader. Appending (None, None).")
            new_patchwise_tensors.append((None, None))
            continue

        new_data_list = []
        new_labels_list = []

        for samples, labels, idx in split:
            if original_shape is None:
                original_shape = tuple(samples.shape)
            
            # Perform the cut directly on the batched tensor
            target_channels = (i * samples.shape[1]) // channels_out
            new_samples = samples[:, :target_channels, ...]
            
            if final_shape is None:
                final_shape = tuple(new_samples.shape)

            # Keep as tensors, do not unpack
            new_data_list.append(new_samples)
            new_labels_list.append(labels)

        # Safely handle empty splits
        if not new_data_list:
            new_patchwise_tensors.append((None, None))
            continue

        # Concatenate efficiently on the batch dimension
        all_samples = torch.cat(new_data_list, dim=0)
        all_labels = torch.cat(new_labels_list, dim=0)

        # Append as a raw tuple (images_tensor, labels_tensor)
        new_patchwise_tensors.append((all_samples, all_labels))

    new_patchwise_latent_dataloaders = create_dataloaders(
        data_dir=None,
        batch_size=B,
        channels_last=False,
        tensors=new_patchwise_tensors, # Pass the raw tensors
        transforms={'train': train_transforms, 'val': None, 'test': None},
        shuffle=True,
        num_workers=4,
        pin_memory=True
    ) 

    NewLatents = {
        'none': none_latent,
        'patchwise': new_patchwise_latent_dataloaders
    }

    print("Cut extra channels from 'patchwise' latent.")
    if original_shape and final_shape:
        print(f"Difference in batch shapes: {original_shape} -> {final_shape}")

    return NewLatents

        



















