import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_quantum.quantum.pennylane_backend import QuantumLayer


def custom_pad_2d(x, padding, pad_filler='median'):
    """
    Pad a 4-D tensor x=(B, C, H, W) with the spatial median per (B,C) channel.
    padding: int number of pixels to pad on all sides.
    Returns padded tensor of shape (B, C, H+2*padding, W+2*padding).
    """
    
    if isinstance(padding, (int)):
        if padding == 0:
            return x
        else:
            padding = {'Up': padding, 'Down': padding, 'Left': padding, 'Right': padding}
        
    elif not isinstance(padding, dict):
        raise ValueError("Padding must be an integer or a dict of 4 integers {'Up': int, 'Down': int, 'Left': int, Right': int}.")

    B, C, H, W = x.shape
    if pad_filler == 'zero':
        padded_x = F.pad(x, (padding['Left'], padding['Right'], padding['Up'], padding['Down']), mode='constant', value=0)
        return padded_x
    elif pad_filler == 'median':
        flat = x.reshape(B, C, -1)
        med = flat.median(dim=-1).values  # (B, C)

        Hn = H + padding['Up'] + padding['Down']
        Wn = W + padding['Left'] + padding['Right']
        med_exp = med.view(B, C, 1, 1).expand(B, C, Hn, Wn).contiguous()
        med_exp[:, :, padding['Down']:(H+padding['Down']), padding['Left']:(W+padding['Left']) ] = x
        
        return med_exp


class QuantumKernel(nn.Module):
    def __init__(self, circuit, channels_out = [-1], ancilla = 0):
        super().__init__()
        self.circuit = circuit
        self.channels_out = channels_out
        self.ancilla = ancilla        

    def forward(self,x):
        assert x.shape[-1] + self.ancilla == self.circuit.num_qubits, f"kernel_size**2 should match number of qubits in the kernel"
        
        if self.ancilla > 0:
            ancilla_tensor = torch.zeros(*x.shape[:-1], self.ancilla, device=x.device, dtype=x.dtype)
            x = torch.cat([ x, ancilla_tensor], dim=-1)

        circuit_out = (1+self.circuit(x))/2 # Normalize between 0 and 1

        return circuit_out[...,  self.channels_out]

        
    

class QuantumConv2D(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, channels_out = [4], channels_last = False, graphs= 'chain', 
                 entangle_method ='CNOT', ancilla = 1, pad_filler = 'median', invert_embedding = True, train_q = False,
                 U3_layers = 0, entangling_layers = 1, input_channels = 1):
        super().__init__()

        if ancilla and channels_out != [-1]:
            print(f'Please be ware that when ancilla is set to True channels_out must be [-1], but got {channels_out}.')
        
        self.train_q = train_q
        self.U3_layers = U3_layers
        self.entangling_layers = entangling_layers
        self.invert_embedding = invert_embedding
        self.channels_out = channels_out 
        self.kernel = QuantumKernel(
            circuit = QuantumLayer(num_qubits = kernel_size**2 + ancilla, graphs = graphs, entangle_method=entangle_method, 
                                   invert = self.invert_embedding, train_q = train_q, U3_layers=self.U3_layers, entangling_layers=self.entangling_layers),
            channels_out = channels_out, ancilla = ancilla
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pad_filler = pad_filler
        if isinstance(padding, int):
            self.padding = {'Up': padding, 'Down': padding, 'Left': padding, 'Right': padding}
        elif not isinstance(padding, dict):
            raise ValueError("Padding must be an integer or a dict of 4 integers {'Up': int, 'Down': int, 'Left': int, Right': int}.")
        self.ancilla = ancilla
        # We'll perform median padding manually so set unfold padding to 0
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=0) # Unfold the input to get sliding local blocks
        self.channels_last = channels_last
        self.input_channels = input_channels

    

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        x = x if self.channels_last == False else x.permute(0, 3, 1, 2)
        if x.ndim == 4:
            B, C, H, W = x.shape 
        elif x.ndim == 3:
            assert self.input_channels != None, f"When input.ndim == 3 the number of input_channels is necessary to avoid shape mismatch errors. "
            B, C, H, W = x.shape[0], self.input_channels, x.shape[1], x.shape[2]
            x = x.unsqueeze(1)
            
        q = len(self.channels_out) 

        H_out = (H + self.padding['Up'] + self.padding['Down'] - self.kernel_size) // self.stride + 1
        W_out = (W + self.padding['Left'] + self.padding['Right'] - self.kernel_size) // self.stride + 1

        outputs_by_channel = []

        for channel in range(C):

            y = x[:, channel, :, :].unsqueeze(1)  # shape (B, 1, H, W)
            # Apply median padding manually so unfold uses padded tensor
            y = custom_pad_2d(y, self.padding, self.pad_filler)

            patches = self.unfold(y)  # (B, kernel_size^2, L), L is number of patches
            patches = patches.transpose(1, 2)  # (B, L, kernel_size^2)

            # Flatten to (B * L, patch_vector)
            patch_vectors = patches.reshape(-1, patches.shape[-1])  # (B*L, patch_dim)

            if patch_vectors.shape[1] + self.ancilla != self.kernel.circuit.num_qubits:
                raise ValueError(f"Expected input dim {self.kernel.circuit.num_qubits - self.ancilla}, got {patch_vectors.shape[1]}")

            # Process with the Kernel — ideally batched
            kernel_out = self.kernel(patch_vectors)  # expected output: (B*L, q)

            # Reshape back to (B, L, out_dim)
            kernel_out = kernel_out.view(B, -1, q)  # (B, L, q)

            # Reshape to (B, q, H_out, W_out)
    
            output = kernel_out.transpose(1, 2).view(B, -1, H_out, W_out)
            outputs_by_channel.append(output)
        
        if C > 1:
            full_out_by_channel = torch.cat(outputs_by_channel, dim=1)  # (B, C * q, H_out, W_out)
            full_out_by_channel = full_out_by_channel.view(B, C, q, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous().view(B, q * C, H_out, W_out)
            return full_out_by_channel
        else:
            return output[:,0,:,:] # (B, H_out, W_out)
        

class Convolution2D(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, channels_out=[4], channels_last=False, 
                 pad_filler='median', bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_filler = pad_filler
        self.channels_last = channels_last
        self.channels_out_list = channels_out
        
        # q represents the number of output features per input channel
        # In your Quantum version, this is len(channels_out)
        self.q = len(channels_out)

        # The classical equivalent of the QuantumKernel per channel.
        # We use a Linear layer because Unfold turns the convolution into a matrix multiplication.
        # Input dim: kernel_size^2 | Output dim: q
        self.classical_kernel = nn.Linear(kernel_size**2, self.q, bias=bias)

        if isinstance(padding, int):
            self.padding = {'Up': padding, 'Down': padding, 'Left': padding, 'Right': padding}
        elif not isinstance(padding, dict):
            raise ValueError("Padding must be an integer or a dict of 4 integers.")
        else:
            self.padding = padding

        # Manual padding means we set unfold padding to 0
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=0)

    def forward(self, x):
        """
        x: (B, C, H, W) or (B, H, W, C) if channels_last is True
        """
        print(f"x.shape: {x.shape}")
        # Handle channel ordering
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)

        if x.ndim == 4:
            B, C, H, W = x.shape
        elif x.ndim == 3:
            B, C, H, W = 1, *x.shape
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D")

        # Calculate output dimensions
        H_out = (H + self.padding['Up'] + self.padding['Down'] - self.kernel_size) // self.stride + 1
        W_out = (W + self.padding['Left'] + self.padding['Right'] - self.kernel_size) // self.stride + 1

        outputs_by_channel = []

        for channel in range(C):
            # Isolate channel: (B, 1, H, W)
            y = x[:, channel, :, :].unsqueeze(1)
            
            # Apply your custom padding logic
            y = custom_pad_2d(y, self.padding, self.pad_filler)

            # Extract patches: (B, kernel_size^2, L) where L = H_out * W_out
            patches = self.unfold(y)
            patches = patches.transpose(1, 2)  # (B, L, kernel_size^2)

            # Flatten for the linear layer (classical kernel)
            patch_vectors = patches.reshape(-1, self.kernel_size**2) # (B*L, patch_dim)

            # Apply classical kernel
            kernel_out = self.classical_kernel(patch_vectors)  # (B*L, q)

            # Reshape back to (B, q, H_out, W_out)
            output = kernel_out.view(B, -1, self.q).transpose(1, 2).view(B, self.q, H_out, W_out)
            outputs_by_channel.append(output)

        if C > 1:
            # Concatenate and permute to match your specific quantum channel grouping logic
            full_out = torch.cat(outputs_by_channel, dim=1)  # (B, C * q, H_out, W_out)
            full_out = full_out.view(B, C, self.q, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()
            return full_out.view(B, self.q * C, H_out, W_out)
        else:
            print(f"C<=1")
            # Matches your return output[:,0,:,:] for single channel
            return outputs_by_channel[0][:, 0, :, :]
    
class QuantumConv1D(nn.Module):
    def __init__(self, window_size=3, stride=1, padding=0, channels_out = [4], graphs= 'chain', entangle_method = 'CNOT', ancilla = 1, pad_filler = 'median', input_channels = None, transposeB = False):
        super().__init__()

        if ancilla and channels_out != [-1]:
            print(f'Please be ware that when ancilla is set to True channels_out must be [-1], but got {channels_out}. Automatically setting channels_out to [-1]')

        self.channels_out = channels_out if not ancilla else [-1]
        self.entangle_method = entangle_method
        self.pad_filler = pad_filler
        self.kernel = QuantumKernel(
            circuit = QuantumLayer(num_qubits = window_size + ancilla, graphs = graphs, entangle_method=entangle_method),
            channels_out = channels_out, ancilla = ancilla
        )

        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        if isinstance(padding, int):
            self.padding = {'Up': padding, 'Down': padding, 'Left': 0, 'Right': 0}
        elif not isinstance(padding, dict):
            raise ValueError("Padding must be an integer or a dict of 2 integers {'Up': int, 'Down': int}.")
        else:
            self.padding['Left'] = 0
            self.padding['Right'] = 0

        self.ancilla = ancilla
        # We'll perform median padding manually so set unfold padding to 0
        self.unfold = nn.Unfold(kernel_size=(window_size, 1), stride=(stride, 1), padding=(0, 0)) # Unfold the input to get sliding local blocks
        self.transposeB = transposeB

    def forward(self, x):
        orig_dim = x.dim()
        
        # 1. Standardize to 4D: (B, C, H, W)
        if orig_dim == 2:
            B, L = x.shape
            C, W = 1, 1
            x = x.unsqueeze(1).unsqueeze(3) # (B, 1, L, 1)
        elif orig_dim == 3:
            B, C, L = x.shape
            W = 1
            x = x.unsqueeze(3) # (B, C, L, 1)
        elif orig_dim == 4:
            B, C, L, W = x.shape
        else:
            raise ValueError(f"Input must be 2D, 3D, or 4D. Got {orig_dim}D")

        # 2. Handle Transpose
        if self.transposeB:
            # Swap L and W dimensions
            x = x.transpose(-1, -2).contiguous()
        
        # Current shapes after potential transpose
        B, C, L_curr, W_curr = x.shape

        # 3. Calculate Output Length based on current 'L' (Height)
        L_out = (L_curr + self.padding['Up'] + self.padding['Down'] - self.window_size) // self.stride + 1
        outputs_by_channel = []

        for channel in range(C):
            y = x[:, channel:channel+1, :, :] 
            y = custom_pad_2d(y, self.padding, self.pad_filler)

            # Unfold extracts patches along the Height (L_curr)
            patches = self.unfold(y) # (B, window_size, L_out * W_curr)
            
            # Reshape to (B, window_size, L_out, W_curr)
            patches = patches.view(B, self.window_size, L_out, W_curr)
            
            # Permute to (B, L_out, W_curr, window_size) for the kernel
            patches = patches.permute(0, 2, 3, 1).contiguous()
            
            patch_vectors = patches.view(-1, self.window_size)
            kernel_out = self.kernel(patch_vectors) # (Total_Patches, D)
            D = kernel_out.shape[-1]

            # Reshape to (B, L_out, W_curr, D) then permute to (B, D, L_out, W_curr)
            output = kernel_out.view(B, L_out, W_curr, D).permute(0, 3, 1, 2).contiguous()
            outputs_by_channel.append(output)

        results = torch.cat(outputs_by_channel, dim=1) # (B, C*D, L_out, W_curr)

        # 4. Consistent Dimensionality Reduction
        # If we started with 3D (B, C, L), we expect (B, C*D, L_out)
        if orig_dim == 3:
            # If transposeB was True, L_out is now in the H position, W_curr is 1
            # If transposeB was False, W_curr is 1
            # We want to remove the '1' dimension without accidentally squeezing B or C*D
            if self.transposeB:
                # Result is (B, C*D, L_out, original_L) -> this doesn't happen in 1D conv 
                # usually, but let's be safe:
                return results.squeeze(-2) if results.shape[-2] == 1 else results
            else:
                return results.squeeze(-1) # Removes the W=1 dimension

        # If we started with 2D (B, L), we likely want (B, L_out) or (B, D, L_out)
        if orig_dim == 2:
            results = results.squeeze(-1) # Remove W
            if results.shape[1] == 1: # If only one output channel
                return results.squeeze(1)
            return results

        return results






