import torch
import pennylane as qml
import numpy as np
from .graphs import graph_builder

class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        num_qubits,
        graph='chain',
        entangle_method='CNOT',
        invert=True, 
        train_q = False,
        train_r = False
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.entangle_method = entangle_method
        self.train_q = train_q
        self.train_r = train_r
        print(f"Quantum layer trainable? : {self.train_q or self.train_r}")
        
        # --- 1. Robust Graph Loading ---
        if isinstance(graph, str):
            # Build graph from string
            self.graph_data = graph_builder(graph, num_qubits)
        elif isinstance(graph, dict):
            # Validate dict
            if 'edges' not in graph or 'weights' not in graph:
                raise ValueError("Graph dict must contain 'edges' and 'weights'.")
            self.graph_data = graph
        else:
            raise ValueError(f"Invalid graph input: {graph}")

        # --- 2. Extract Data (and ensure valid lists/arrays) ---
        self.edges = self.graph_data['edges']
        self.weights_data = self.graph_data['weights'] 

        # --- 3. Device & Circuit ---
        dev = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit_(inputs, weights):
            
            # --- Input Embedding ---
            # Pre-processing input
            inputs = np.pi * (1 - invert + (2 * invert - 1) * torch.clamp(inputs, min=0, max=1))
            
            if self.entangle_method == 'edges':
                # WARNING: This logic is hardcoded for 4 qubits. 
                # Ensure num_qubits == 4 if using this method.
                exaggeration = 3
                qml.RY(exaggeration * (inputs[..., 0] - inputs[..., 1] + inputs[..., 2] - inputs[..., 3]), wires=0)
                qml.RY(exaggeration * (inputs[..., 0] - inputs[..., 2] + inputs[..., 1] - inputs[..., 3]), wires=2)
                qml.RY(inputs[..., 1], wires=1)
                qml.RY(inputs[..., 3], wires=3)
                qml.CRY((np.pi - inputs[..., 3]), wires=[0, 3])
                qml.CRY((np.pi - inputs[..., 3]), wires=[2, 3])
            
            else:
                # Standard Angle Embedding
                qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')

                # --- Entanglement Layers ---
                if self.entangle_method == 'SEL':
                    # Apply ONCE, not per edge
                    qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits), ranges=[1])
                
                else:
                    # If using SEL, pass the provided weight tensor directly
                    if self.entangle_method == 'SEL':
                        qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits), ranges=[1])
                    else:
                        # If layer is trainable, apply a per-qubit unitary rotation (U3/Rot)
                        # Expect `weights` to have shape (1, num_qubits, 3) when trainable.
                        if self.train_q:
                            # Normalize shapes: accept (1, N, 3) or (N, 3)
                            try:
                                if hasattr(weights, 'ndim') and weights.ndim == 3 and weights.shape[0] == 1:
                                    weights = weights[0]
                            except Exception:
                                pass

                            for q in range(self.num_qubits):
                                theta = weights[3*q]
                                phi = weights[3*q+1]
                                lam = weights[3*q+2]
                                qml.Rot(theta, phi, lam, wires=q)

                        # Iterate over EDGES, not the dict keys, for entangling gates
                        for i, edge in enumerate(self.edges):
                            u, v = edge[0], edge[1]

                            # Use the specific weight for this edge if available
                            if not self.train_r:
                                w = self.weights_data[i] if i < len(self.weights_data) else (np.pi / 3)
                            else:
                                w = weights[3*self.num_qubits*self.train_q + i]

                            if self.entangle_method == 'CNOT':
                                qml.CNOT(wires=[u, v])
                            elif self.entangle_method == 'CRX':
                                qml.CRX(w, wires=[u, v])
                            elif self.entangle_method == 'CRY':
                                qml.CRY(w, wires=[u, v])

            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # --- 4. Setup Weights ---
        # If using SEL or per-qubit trainable rotations, expose a weight tensor
        # with one layer of shape (1, num_qubits, 3). Otherwise provide a dummy scalar shape.
        shapes = {'SEL': (1, num_qubits, 3), 'train': (num_qubits * 3 * self.train_q + len(self.edges) * self.train_r)}
        weight_shape = shapes['SEL'] if self.entangle_method == 'SEL' else shapes['train']
        
        # Create the QNode
        self.magic = qml.qnn.TorchLayer(circuit_, {"weights": weight_shape})

        # Freeze weights (Reservoir Computing style)
        # for param in self.magic.parameters():
        #     param.data.zero_()
        #     param.requires_grad = False
        print(f"Quantum Layer Info: train_q={self.train_q} | Weight Shape={weight_shape} | Layer Params={sum(p.numel() for p in self.magic.parameters())} | Entangle Method : {self.entangle_method}")

    def forward(self, inputs):
        if inputs.ndim == 3:
            return torch.vmap(torch.vmap(self.magic))(inputs)
        elif inputs.ndim == 2:
            return torch.vmap(self.magic)(inputs)
        elif inputs.ndim == 1:
            return self.magic(inputs)
        else:
            raise ValueError("Input tensor must be 1D, 2D or 3D")