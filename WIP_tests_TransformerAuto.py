from mi_quantum.mkexperiments import make_experiment_autoenformer
import mi_quantum.quantum as quantum
import torch

# 1. Define Base Configs
exp_config_base = {
    'experiment_id': 'autoenformer_results/first',
    'experiment_name': 'autoenformer_results/first',
    'B': 256,
    'N1': 100, # Num epochs
    'N2' : 125,
    'num_experiments': 10,
    'num_classes': 7,
    'square' : True,
    'repeat_autoencoder' : False,
    'pixels' : 28,
    'q_config' : { 'none', '2x2', '3x3', '1xn', 'nx1' },
    'channels_last': False,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'second_at_a_time' : False,
    'send_telegram': False
}

# Hyperparams
p1 = {
    'learning_rate': 5e-3, 'hidden_size': 48, 'dropout': {'embedding_attn': 0.025, 'after_attn': 0.025, 'feedforward': 0.025, 'embedding_pos': 0.025},
    'num_head': 48, 'Attention_N' : 2, 'num_transf': 1, 'mlp_size': 30, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'none', 'entangle_method' : 'CRX',
    'parallel' : 1 ,'connectivity': 'king', 'RD': 1, 'patience': -1, 'scheduler_factor': 0.999, 'q_stride': 1, 'ancilla' : 0
}

p2 = {
    'learning_rate': 0.0025, 'hidden_size': 30, 'dropout': {'embedding_attn': 0.125, 'after_attn': 0.125, 'feedforward': 0.125, 'embedding_pos': 0.125},
    'quantum' : False, 'num_head': 5, 'Attention_N' : 2, 'num_transf': 2, 'mlp_size': 30, 'patch_size': 4, 'weight_decay': 1e-7, 'attention_selection': 'filter',
    'RD': 1, 'special_cls' : False, 'parallel': 1, 'patience': -1, 'scheduler_factor': 0.9995, 'q_stride': 1, 'selection_amount' : 49
}

# Kernels to pass as in-between-steps processors
# Ensure CosineEncoding is defined or imported

class CosineEncoding(torch.nn.Module):
    def __init__(self, nesting = 1):
        super(CosineEncoding, self).__init__()
        self.nesting = nesting
    def forward(self, x):
        out = x
        for _ in range(self.nesting):
            out = torch.cos(out * torch.pi/2)
        return out

# --- Configuration Constants (extracted from your previous logic) ---
# You can adjust these values here to change the static kernels

ENTANGLE = 'CRX'
KERNEL_SIZE = 3
PADDING_2D = {'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1} # For kernel_size 3
n = 3
PADDING_1D = {'Up': (n-1)//2, 'Down': (n-1)//2} # For window_size 9
INPUT_CHANNELS = 1

data_kernels = {
    # 1. Identity (No processing)
    'none': torch.nn.Identity(),

    # 2. Standard 2D Quantum Convolution (Patchwise)
    '3x3': quantum.quanvolution.QuantumConv2D(
        kernel_size=3,
        stride=1,
        padding={'Up': 1, 'Down': 1, 'Left': 1, 'Right': 1},
        channels_out=[4],
        ancilla=0,
        graphs='king',
        entangle_method=ENTANGLE,
        invert_embedding=True, # Defaulted to True in previous snippets
        pad_filler='zero',
        input_channels = 1
    ),
    '2x2': quantum.quanvolution.QuantumConv2D(
        kernel_size=2,
        stride=1,
        padding={'Up': 1, 'Down': 0, 'Left': 1, 'Right': 0},
        channels_out= [1],
        ancilla=0,
        graphs='star',
        entangle_method=ENTANGLE,
        invert_embedding=True, # Defaulted to True in previous snippets
        pad_filler='zero',
        input_channels= 1
    ),

    # 3. Vertical 1D Quantum Convolution
    'nx1': quantum.quanvolution.QuantumConv1D(
        window_size=n,
        stride=1,
        padding={'Up': (n-1)//2, 'Down': (n-1)//2, 'Left': 0, 'Right': 0},
        channels_out=[(n+1)//2],
        graphs='chain',
        entangle_method=ENTANGLE,
        ancilla=0,
        pad_filler='zero',
        transposeB = False
    ),

    # 3. Vertical 1D Quantum Convolution
    '1xn': quantum.quanvolution.QuantumConv1D(
        window_size=n,
        stride=1,
        padding={'Up': (n-1)//2, 'Down': (n-1)//2, 'Left': 0, 'Right': 0},
        channels_out=[(n+1)//2],
        graphs='chain',
        entangle_method=ENTANGLE,
        ancilla=0,
        pad_filler='zero', 
        transposeB= True
    ),

    # 4. Cosine Encoding
    'cosine': CosineEncoding(nesting=1)
}

# 2. Define Iterables
all_iter = {

}

m1_iter = {

}

data_iter = {
    
}

m2_iter = {

}

graph_columns = ['q_config', 'test_auc']

if __name__ == "__main__":
    # 3. Run Experiment
    print("--- Starting Refactored AutoEnformer Experiment ---")
    make_experiment_autoenformer(
        exp_config_base, 
        p1_base=p1,
        p2_base = p2,
        data_kernels= data_kernels,
        all_iter=all_iter, 
        data_iter= data_iter,
        m1_iter=m1_iter,
        m2_iter=m2_iter,
        graph_columns=graph_columns
    )

    print("--- Refactored Experiment Finished ---")