import torch
import pennylane as qml
import numpy as np
from .graphs import graph_builder

class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        num_qubits,
        graphs, 
        entangle_method='CNOT',
        invert=True, 
        U3_layers = 0,
        entangling_layers = 0, 
        train_q = False # Now controls ONLY entangling params
    ):
        super().__init__()

        self.num_qubits = num_qubits
        self.entangle_method = entangle_method
        self.U3_layers = int(U3_layers)
        self.entangling_layers = int(entangling_layers)
        self.invert = invert
        self.train_q = train_q

        # 1. Graph Processing (Logic remains same)
        if not isinstance(graphs, list): graphs = [graphs]
        if len(graphs) < self.entangling_layers:
            new_graphs = [graphs[i % len(graphs)] for i in range(self.entangling_layers)]
            graphs = new_graphs

        self.layers_edges = []
        self.layers_weights = []
        for g in graphs[:self.entangling_layers]:
            data = graph_builder(g, num_qubits) if isinstance(g, str) else g
            self.layers_edges.append(data['edges'])
            self.layers_weights.append(data['weights'])

        # 2. Parameter Indexing
        # We split weights into two distinct groups for TorchLayer to manage
        self.u3_param_count = 3 * self.num_qubits * self.U3_layers
        
        self.entangle_param_count = 0
        if self.entangle_method in ['CRX', 'CRY']:
            for edges in self.layers_edges:
                self.entangle_param_count += len(edges)
        
        # 3. Device & Circuit
        dev = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit_(inputs, u3_weights, ent_weights):
            # Input Encoding
            inputs = np.pi * (1 - self.invert + (2 * self.invert - 1) * torch.clamp(inputs, min=0, max=1))
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            
            u3_idx = 0
            ent_idx = 0
            max_layers = max(self.U3_layers, self.entangling_layers)
            
            for L in range(max_layers):
                # Rotation Layer (U3) - Always trainable if present
                if L < self.U3_layers:
                    for q in range(self.num_qubits):
                        qml.Rot(*u3_weights[u3_idx:u3_idx+3], wires=q)
                        u3_idx += 3
                
                # Entanglement Layer - Frozen based on train_q
                if L < self.entangling_layers:
                    current_edges = self.layers_edges[L]
                    current_fixed_weights = self.layers_weights[L]

                    for i, (u, v) in enumerate(current_edges):
                        if self.entangle_method == 'CNOT':
                            qml.CNOT(wires=[u, v])
                        else:
                            w = ent_weights[ent_idx]
                            ent_idx += 1
                            if self.entangle_method == 'CRX':
                                qml.CRX(w + current_fixed_weights[i], wires=[u, v])
                            elif self.entangle_method == 'CRY':
                                qml.CRY(w + current_fixed_weights[i], wires=[u, v])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # 4. TorchLayer Setup with weight separation
        weight_shapes = {
            "u3_weights": (self.u3_param_count,) if self.u3_param_count > 0 else (0,),
            "ent_weights": (self.entangle_param_count,) if self.entangle_param_count > 0 else (0,)
        }
        
        self.magic = qml.qnn.TorchLayer(circuit_, weight_shapes)
        
        # 5. Selective Freezing
        if not self.train_q:
            if hasattr(self.magic, 'ent_weights'):
                self.magic.ent_weights.requires_grad = False
                # Optionally zero them out if that was your goal
                with torch.no_grad():
                    self.magic.ent_weights.fill_(0)

    def forward(self, inputs):
        # ... (vmap logic remains same)
        if inputs.ndim == 3:
            return torch.vmap(torch.vmap(self.magic))(inputs)
        elif inputs.ndim == 2:
            return torch.vmap(self.magic)(inputs)
        return self.magic(inputs)