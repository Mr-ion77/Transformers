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
        invert=True
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.entangle_method = entangle_method
        
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
        def circuit_(inputs, qnode_weights):
            
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
                    qml.StronglyEntanglingLayers(qnode_weights, wires=range(self.num_qubits), ranges=[1])
                
                else:
                    # Iterate over EDGES, not the dict keys
                    for i, edge in enumerate(self.edges):
                        u, v = edge[0], edge[1]
                        
                        # Use the specific weight for this edge if available
                        # (Fallback to pi/3 if you prefer, but using the weight map is usually better)
                        w = self.weights_data[i] if i < len(self.weights_data) else (np.pi/3)

                        if self.entangle_method == 'CNOT':    
                            qml.CNOT(wires=[u, v])
                        elif self.entangle_method == 'CRX':
                            qml.CRX(w, wires=[u, v]) 
                        elif self.entangle_method == 'CRY':
                            qml.CRY(w, wires=[u, v])

            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # --- 4. Setup Weights ---
        weight_shape = (1, num_qubits, 3) if self.entangle_method == 'SEL' else (1,)
        
        # Create the QNode
        self.magic = qml.qnn.TorchLayer(circuit_, {"qnode_weights": weight_shape})

        # Freeze weights (Reservoir Computing style)
        for param in self.magic.qnode_weights.values():
            param.data.zero_()
            param.requires_grad = False

    def forward(self, inputs):
        if inputs.ndim == 3:
            return torch.vmap(torch.vmap(self.magic))(inputs)
        elif inputs.ndim == 2:
            return torch.vmap(self.magic)(inputs)
        elif inputs.ndim == 1:
            return self.magic(inputs)
        else:
            raise ValueError("Input tensor must be 1D, 2D or 3D")