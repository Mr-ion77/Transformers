import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_quantum.quantum.pennylane_backend import QuantumLayer


class QuantumKernel(nn.Module):
    def __init__(self, circuit, channels_out = [-1], ancilla = 1):
        super().__init__()
        self.circuit = circuit
        self.channels_out = channels_out if ancilla == 0 else list(range(-1, -ancilla - 1, -1))
        self.ancilla = ancilla        

    def forward(self,x):
        assert x.shape[-1] + self.ancilla == self.circuit.num_qubits, f"patch_size**2 should match number of qubits in the kernel"
        
        if self.ancilla > 0:
            ancilla_tensor = torch.zeros(*x.shape[:-1], self.ancilla, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ancilla_tensor], dim=-1)


        circuit_out = self.circuit(x)

        return circuit_out[...,  self.channels_out]
    

class QuantumConv2D(nn.Module):
    def __init__(self, patch_size=3, stride=1, padding=0, channels_out = [4], channels_last = False, graph= 'chain', ancilla = 1, trainBool = False):
        super().__init__()

        if ancilla and channels_out != [-1]:
            print(f'Please be ware that when ancilla is set to True channels_out must be [-1], but got {channels_out}. Automatically setting channels_out to [-1]')

        self.channels_out = channels_out if not ancilla else [-1]
        self.kernel = QuantumKernel(
            circuit = QuantumLayer(num_qubits = patch_size**2 + ancilla, entangle = True, graph = graph, trainBool = False),
            channels_out = channels_out, ancilla = ancilla
        )
        self.trainBool = trainBool
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.ancilla = ancilla
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.stride, padding=self.padding) # Unfold the input to get sliding local blocks
        self.channels_last = channels_last

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        x = x if self.channels_last == False else x.permute(0, 3, 1, 2)

        B, C, H, W = x.shape 
        H_out = (H + 2*self.padding - self.patch_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.patch_size) // self.stride + 1

        outputs_by_channel = []

        for channel in range(C):

            y = x[:,channel,:,:].unsqueeze(1) # shape (B, 1, H, W)

            patches = self.unfold(y)  # (B, patch_size^2, L), L is number of patches
            patches = patches.transpose(1, 2)  # (B, L, patch_size^2)

            # Flatten to (B * L, patch_vector)
            patch_vectors = patches.reshape(-1, patches.shape[-1])  # (B*L, patch_dim)

            if patch_vectors.shape[1] + self.ancilla != self.kernel.circuit.num_qubits:
                raise ValueError(f"Expected input dim {self.kernel.circuit.num_qubits - self.ancilla}, got {patch_vectors.shape[1]}")

            # Process with the Kernel — ideally batched
            with torch.set_grad_enabled(self.kernel.circuit.trainBool):
                kernel_out = self.kernel(patch_vectors)  # expected output: (B*L, out_dim)

            # Reshape back to (B, L, out_dim)
            kernel_out = kernel_out.view(B, -1, kernel_out.shape[-1])  # (B, L, D)

            # Reshape to (B, D, H_out, W_out)
    
            output = kernel_out.transpose(1, 2).view(B, -1, H_out, W_out)
            outputs_by_channel.append(output)

        return torch.cat(outputs_by_channel, dim=1) if C > 1 else output[:,0,:,:] # (B, C×D, H_out, W_out)
    
class QuantumConv1D(nn.Module):
    def __init__(self, window_size=3, stride=1, padding=0, channels_out = [4], graph= 'chain', ancilla = 1, trainBool = False):
        super().__init__()

        if ancilla and channels_out != [-1]:
            print(f'Please be ware that when ancilla is set to True channels_out must be [-1], but got {channels_out}. Automatically setting channels_out to [-1]')

        self.channels_out = channels_out if not ancilla else [-1]
        self.kernel = QuantumKernel(
            circuit = QuantumLayer(num_qubits = window_size + ancilla, entangle = True, graph = graph, trainBool = False),
            channels_out = channels_out, ancilla = ancilla
        )
        self.trainBool = trainBool
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.ancilla = ancilla
        self.unfold = nn.Unfold(kernel_size=(window_size, 1), stride=(stride, 1), padding=(padding, 0)) # Unfold the input to get sliding local blocks

    def forward(self, x):
        """
        x: (B, C, L)
        """
        if x.dim() == 3:
            B, C, L = x.shape 
        elif x.dim() == 2:
            B, L = x.shape 
            C = 1
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            B, C, H, W = x.shape 
            if H != 1:
                print(f'Warning: expected H=1 for 1D input, got H={H}. Applying 1-D convolution over H dimension for each W separately.')
            L = H

        L_out = (L + 2*self.padding - self.window_size) // self.stride + 1

        outputs_by_channel = []

        for channel in range(C):

            # Prepare y as a 4-D tensor (B, 1, L, W) where W==1 for simple 1-D input
            if x.dim() == 3:
                # x: (B, C, L) -> y: (B, 1, L, 1)
                y = x[:, channel, :].unsqueeze(1).unsqueeze(3)
                B_loc = B
                W = 1
            else:
                # x: (B, C, H, W) -> treat H as length L and process each column (W) separately
                y = x[:, channel, :, :].unsqueeze(1)  # (B, 1, H, W)
                B_loc = B
                W = x.shape[3]

            # Extract patches: (B, window_size * 1, L_out * W)
            patches = self.unfold(y)  # (B, window_size, L_out * W)

            # Reshape to separate column positions: (B, window_size, L_out, W)
            patches = patches.view(B_loc, self.window_size, L_out, W)

            # Move window dim to last so we can get (B, L_out, W, window_size)
            patches = patches.permute(0, 2, 3, 1)  # (B, L_out, W, window_size)

            # Flatten to (B * L_out * W, window_size) to batch through quantum kernel
            patch_vectors =  patches.reshape(-1, patches.shape[-1])  # (B*L_out*W, window_size)

            if patch_vectors.shape[1] + self.ancilla != self.kernel.circuit.num_qubits:
                raise ValueError(f"Expected input dim {self.kernel.circuit.num_qubits - self.ancilla}, got {patch_vectors.shape[1]}")

            # Process with the Kernel — batched over all patches and columns
            with torch.set_grad_enabled(self.kernel.circuit.trainBool):
                kernel_out = self.kernel(patch_vectors)  # (B*L_out*W, D)

            D = kernel_out.shape[-1]

            # Reshape back to (B, L_out, W, D)
            kernel_out = kernel_out.view(B_loc, L_out, W, D)

            # Move output channels to second dim: (B, D, L_out, W)
            output = kernel_out.permute(0, 3, 1, 2).contiguous()

            outputs_by_channel.append(output)

        # If original input was 3-D, collapse W dim (which will be 1) to return (B, C*D, L_out)
        if x.dim() == 3:
            # outputs_by_channel entries are (B, D, L_out, 1) -> squeeze last dim
            outputs_by_channel = [o.squeeze(-1) for o in outputs_by_channel]  # (B, D, L_out)
            return torch.cat(outputs_by_channel, dim=1)  # (B, C*D, L_out)

        # For 4-D input return (B, C*D, L_out, W)
        return torch.cat(outputs_by_channel, dim=1)






