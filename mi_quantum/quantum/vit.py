import torch
from torch import nn
from mi_quantum.quantum.pennylane_backend import QuantumLayer
import numbers
import torch.nn.functional as F

# See:
# - https://nlp.seas.harvard.edu/annotated-transformer/
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

def identity_tensor(d: int, n: int) -> torch.Tensor:
        """
        Creates an n-dimensional identity tensor of shape (d, d, ..., d)
        with ones where all indices are equal, zeros elsewhere.
        """
        # Create an n-dimensional grid of indices
        indices = torch.arange(d)
        # Generate n copies of indices for broadcasting
        grids = torch.meshgrid(*([indices] * n), indexing='ij')
        # Stack to get shape (n, d, d, ..., d)
        stacked = torch.stack(grids)  # shape: (n, d, d, ..., d)

        # Check where all indices along the first dimension are equal
        # That is, all equal along axis=0
        equal_mask = torch.all(stacked == stacked[0], dim=0)

        return equal_mask.to(dtype=torch.float32)

def rank_patches_by_attention(attn: torch.Tensor) -> torch.Tensor:
            """
            Ranks image patches by the total attention they receive.

            """
            # Average over heads: (B, T, T)
            attn_mean = attn.mean(dim=1)

            # Total attention received by each token: sum over the source positions (axis=-2)
            # attention_received[b, j] = sum over i of attn[b, i, j]
            attention_received = attn_mean.sum(dim=1)  # shape: (B, T)

            # Sort patches by total attention received, descending
            sorted_indices = attention_received.argsort(dim=1, descending=True)  # shape: (B, T)

            return sorted_indices


class NMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        N=2,  # Order of multilinear form
        dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.N = N


        if self.N < 2:
            raise ValueError("N (order of multilinear form) must be at least 2.")

        # One projection per tensor dimension
        self.projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(N)])

        self.v_proj = nn.Linear(embed_dim, embed_dim)  # Dedicated value projection
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout['embedding_attn'])

        # Learnable N-way tensor for multilinear attention
        if N != 2:
            self.A = nn.Parameter(torch.randn(*(self.N * (self.head_dim,))))
        else:
            self.register_buffer('A_identity', identity_tensor(d=self.head_dim, n=2))
    

    def forward(self, x):
        B, S, E = x.shape
        assert E == self.embed_dim

        # compute the N projections: each proj -> (B, S, H, D) then transpose to (B, H, S, D)
        proj_x = [
            proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, H, S, D)
            for proj in self.projections
        ]

        # build einsum string:
        # A has indexes a0 a1 ... a_{N-1}
        # projection n has subscript "b h token_n letter_n" where token_n is:
        #   - 'i' for mode 0 (query axis, kept)
        #   - 'j' for mode 1 (key axis, kept)
        #   - 's' for modes 2..N-1 (context axes, summed/marginalized)
        #
        # Result should be 'b h i j'
        # pools of letters
        tokens = list("ijk")  # token positions
        embeds = list("acdefguvwxyzlmnopqrst")  # embedding dims (avoid b,h,i,j,k,... collisions)

        assert self.N <= len(embeds), f"N too large, max {len(embeds)}"
        
        # subscripts for A (the N-way tensor)
        A_sub = "".join(embeds[:self.N])  # e.g. "abc" for N=3
        
        proj_subs = []
        for n in range(self.N):
            token = tokens[n] if n < 2 else tokens[2]  # first two → i,j ; rest → s (context)
            dim = embeds[n]                            # unique embedding letter
            proj_subs.append(f"bh{token}{dim}")
        
        # result: always bhij (standard 2D attention map)
        einsum_str = f"{A_sub}," + ",".join(proj_subs) + "->bhij"

        # Example N=3: 'acd,bhia,bhjc,bhkd->bhij'

        # execute einsum, resulting shape -> (B, H, S, S)
        # (this implicitly sums over the 's' token index for context modes)
        A = self.A if self.N != 2 else self.A_identity.to(x.device)
        attn_logits = torch.einsum(einsum_str, A, *proj_x)  # (B, H, S, S)

        # scale (similar to standard attention)
        attn_logits = attn_logits / (self.head_dim ** 0.5)

        # softmax over keys (j) to get attention weights per query i
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)


        # compute values using dedicated v_proj (so values are independent of mode projections)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,D)

        # weighted sum: (B,H,S,D) = sum_j attn[b,h,i,j] * v[b,h,j,d]
        values = torch.einsum("bhij,bhjd->bhid", attn, v)

        # reshape back to (B, S, E)
        values = values.transpose(1, 2).reshape(B, S, E)

        out = self.o_proj(values)

        return out, attn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, special_cls = 'none'):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.special_cls = special_cls

        print('Started a MutliheadSelfAttention layer with embed_dim:', embed_dim, 'num_heads:', num_heads, 'head_dim:', self.head_dim)

        if self.special_cls == 'full_projection':
            self.cls_proj = nn.Linear(embed_dim, embed_dim)
        if self.special_cls == 'part_projection':
            self.cls_ponderator = nn.Parameter( torch.tensor([0.5]) )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout['embedding_attn'])
        self.o_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # x.shape = (batch_size, seq_len, embed_dim)
        assert embed_dim == self.embed_dim, f"Input embedding dimension ({embed_dim}) should match layer embedding dimension ({self.embed_dim})"

        full_cls_bool = self.special_cls == 'full_projection'                           # For ramification purposes depending on special_cls config
        part_cls_bool = ( self.special_cls == 'partial_projection' or full_cls_bool )   # For ramification purposes depending on special_cls config

        if part_cls_bool:
            cls_token = x[:,0,:]
            x = x[:,1:,:]

            if full_cls_bool:
                cls_proj = self.cls_proj(x).reshape(batch_size, seq_len - part_cls_bool, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
            else:
                cls_proj = cls_token.expand(-1, seq_len - part_cls_bool, -1).reshape(batch_size, seq_len - part_cls_bool, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, D)
                cls_proj = self.cls_ponderator*cls_proj + x*(1-cls_proj)

        q, k, v, = [
            proj(x).reshape(batch_size, seq_len - part_cls_bool, self.num_heads, self.head_dim).transpose(1, 2)                                                          # (B, H, S, D)  
            for proj, x in zip([self.q_proj, self.k_proj, self.v_proj], [x, x, x, x])
        ]

        # q, k, v.shape = (batch_size, num_heads, seq_len, head_dim)
        qk_dot = q @ k.transpose(-2, -1)
        # promote for stability, compute norms and avoid NaNs
        q_norm2 = (q.float()**2).sum(dim = -1, keepdim = True).clamp(min=1e-5)
        # Compute scaled dot-product attention logits
        attn_logits_standard = ( qk_dot / ( (self.head_dim * q_norm2 )** 0.5)) 
        # Compute softmax and dropout to get weights
        attn_standard = self.dropout( attn_logits_standard.softmax(dim=-1) )    # (B, H, S, S)
        # Compute output
        values = attn_standard @ v  # (B, H, S, D)

        if part_cls_bool:
            # Compute special attention logits involving the cls token: 
            # compute the projection over the cls_token of the projection of key over query.
            # First get query for cls_token
            c = self.q_proj(cls_token).reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)                        # (B, H, 1, D)

            # Compute dot products and norms
            cq_dot = c @ q.transpose(-2, -1)                                                                                        # (B, H, 1, D) @ (B, H, D, S) -> (B, H, 1, S)
            c_norm2 =  ((c.float()**2).sum(dim = -1, keepdim = True).clamp(min=1e-5))                                               # (B, H, 1, D) (keepdim = True)

            # Compute scaled dot-product attention
            attn_logits_cls_to_others = attn_logits_standard * cq_dot / (  (q_norm2 * c_norm2 )** 0.5  )                            # (B, H, S, S)
            attn_cls_to_others = self.dropout( attn_logits_cls_to_others.softmax(dim=-1) )                                          # (B, H, S, S)

            # Now compute attention from others to cls_token in a standard way
            ck_dot = c @ k.transpose(-2, -1)                                                                                        # (B, H, 1, D) @ (B, H, D, S) -> (B, H, 1, S)
            attn_logits_others_to_cls = ( ck_dot / ( (self.head_dim * c_norm2 )** 0.5))                                             # (B, H, 1, S)

            # Compute softmax and dropout to get weights
            attn_others_to_cls = self.dropout( attn_logits_others_to_cls.softmax(dim=-1) )                                          # (B, H, 1, S)
            cls_out = (attn_others_to_cls @ v).reshape(batch_size, 1, self.num_heads * self.head_dim)                               # (B, H, 1, S) @ (B, H, S, D) -> (B, H, 1, D) -> (B, 1, E)

            # Compute output
            values += attn_cls_to_others @ cls_proj
            # values.shape = (batch_size, num_heads, seq_len - 1, head_dim)
            values = torch.cat( (cls_out ,values.transpose(1, 2).reshape(batch_size, seq_len - part_cls_bool, embed_dim)), dim = 1)
            
        else:
            values = values.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        # x.shape = (batch_size, seq_len, embed_dim)
        x = self.o_proj(values)     

        return x, attn_standard

class FeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size, hidden_size_out , quantum = True, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225},
                    q_stride = 4, graph = 'chain'):
        super().__init__()

        self.quantum = quantum
        print(f"Started a FeedForward Module with Quantum-setting: {self.quantum}")
        self.q_stride = q_stride
        self.mlp_hidden_size = mlp_hidden_size

        self.fc1 = nn.Linear(hidden_size, q_stride * mlp_hidden_size)
        self.fc2 = nn.Linear(q_stride * mlp_hidden_size, hidden_size_out)

        if self.quantum:
            self.vqc = QuantumLayer(mlp_hidden_size, graph = graph)
        else:
            self.vqc = nn.Linear(mlp_hidden_size, mlp_hidden_size)

        self.dropout = nn.Dropout(dropout['feedforward'])
        self.gelu = nn.GELU()
        self.q_stride = q_stride
        

    def forward(self, x):
        device = x.device

        if self.q_stride == 1:
            x = self.fc1(x)
            x = self.vqc(x)
            x = x.to(device)  # Ensure the output is on the same device as the input
            x = self.dropout(x)
            x = self.gelu(x)
            x = self.fc2(x)
        else:
            
            x = self.fc1(x)  # x shape: [B, C, L]

            # Extract q_stride slices of size mlp_hidden_size
            slices = [x[:, :, i : i + self.mlp_hidden_size] for i in range(self.q_stride)]

            # Stack into a single batch: shape [q_stride, B, C, mlp_hidden_size]
            x_slices = torch.stack(slices, dim=0)

            # Merge batch for parallel processing: [q_stride * B * C, mlp_hidden_size]
            q, B, C, H = x_slices.shape
            x_slices = x_slices.permute(1, 2, 0, 3).contiguous().view(-1, H)

            # Apply vqc in batch
            x_vqc_output = self.vqc(x_slices)  # returns [self.q_stride * B * C, D]

            # Reshape back: [B, C, q_stride, D]
            D = x_vqc_output.shape[-1]
            x_vqc_output = x_vqc_output.view(B, C, self.q_stride, D)

            # Concatenate outputs along last dimension
            x = x_vqc_output.permute(0, 1, 3, 2).contiguous().view(B, C, -1)

            # Continue forward
            x = self.dropout(x)
            x = self.gelu(x)
            x = self.fc2(x)

        return x

class TransformerBlock_Attention_Chosen_QMLP(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, hidden_size_out, Attention_N = 2, quantum_mlp = True, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, 
                    attention_selection="filter", q_lr = 49, special_cls = False , q_stride = 4, connectivity = 'chain', RD = 1, img_size = 28, patch_size = 4):
        super().__init__()

        self.attention_selection = attention_selection
        self.quantum_mlp = quantum_mlp
        self.dropout = dropout
        self.Attention_N = Attention_N
        self.special_cls = special_cls
        self.q_stride = q_stride
        self.q_lr = q_lr
        # Attention components
        self.attn_norm = nn.LayerNorm(hidden_size)
        if self.Attention_N == 2:
            self.attn = MultiheadSelfAttention(embed_dim = hidden_size, num_heads = num_heads, dropout = dropout, special_cls = self.special_cls)
        else:
            self.attn = NMultiheadSelfAttention(embed_dim = hidden_size, num_heads = num_heads, N=Attention_N, dropout = dropout)
        self.attn_dropout = nn.Dropout(dropout['after_attn'])
        self.hidden_size_out = hidden_size_out
        self.RD = RD

        # MLP components
        self.mlp_norm = nn.LayerNorm(hidden_size)

        self.mlp_sel = FeedForward(hidden_size, mlp_hidden_size, hidden_size_out, quantum = self.quantum_mlp,
                                    dropout = self.dropout, q_stride = self.q_stride,
                                    graph = connectivity)  # Quantum MLP

        if attention_selection not in ["filter", "none"] or RD > 1:
            self.mlp = nn.Linear(hidden_size, hidden_size_out) if attention_selection != "ID" else nn.Identity()

        self.mlp_dropout = nn.Dropout(dropout['feedforward'])

        if attention_selection == "ID" and hidden_size != hidden_size_out:
            raise ValueError("When attention_selection is 'ID', hidden_size must equal hidden_size_out.")


    def forward(self, x):
        # Attention block
        attn_input = self.attn_norm(x)
        attn_output, attn_map = self.attn(attn_input)
        attn_output = self.attn_dropout(attn_output)
        x = x + attn_output
        y = self.mlp_norm(x)

        # MLP input
        if self.attention_selection != "none":

            # Rank patches by attention
            attn_indices = rank_patches_by_attention(attn_map)
            sel_indices = attn_indices[:, :self.q_lr]       # High-attention patches
            normal_indices = attn_indices[:, self.q_lr:]      # Remaining patches

            # Ensure CLS token is always included
            cls_index = torch.zeros(sel_indices.size(0), 1, dtype=torch.long, device=sel_indices.device)
            sel_indices = torch.cat([cls_index, sel_indices[:, :-1]], dim=1)

            # Feedforward on selected patches
            y_sel_in = y.gather(1, sel_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            y_sel_out = self.mlp_sel(y_sel_in)

            # Classical MLP on the rest. Note that if quantum is False, then this is sort of redundant.
            if self.attention_selection == "MLP":
                y_normal_in = y.gather(1, normal_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
                y_normal_out = self.mlp(y_normal_in)
                y_normal_out = self.mlp_dropout(y_normal_out)

            elif self.attention_selection == "ID":
                y_normal_out = y.gather(1, normal_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))

            elif self.attention_selection == "filter":
                # If attention_selection is 'filter', we do not apply a classical MLP
                y_out = y_sel_out
                x = x.gather(1, sel_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))) if self.RD == 1 else self.mlp(x.gather(1, sel_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))))
                return x + y_out, attn_map

            else:
                raise ValueError(f"Unknown attention_selection: {self.attention_selection}")

            # Combine and return, preserving original order
            batch_size, num_tokens, dim = x.size()
            device = x.device

            # Create empty tensor to hold ordered outputs
            y_out = torch.zeros((batch_size, num_tokens, self.hidden_size_out), device=device)

            # Place outputs back in their original positions
            y_out.scatter_(1, sel_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size_out), y_sel_out)
            y_out.scatter_(1, normal_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size_out), y_normal_out)

            return x + y_out, attn_map

        else:
            # If no attention selection, use standard MLP
            y = self.mlp_sel(y)
            y = self.mlp_dropout(y)
            return x + y, attn_map


class VisionTransformer(nn.Module):
    def __init__(self, img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size, Attention_N = 2,
                    quantum_mlp = False, quantum_classification = False, dropout= {'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, 
                    channels_last=False, RD = 1, attention_selection = 'filter', selection_amount = None, special_cls = 'none',
                    parallel = 1, parallel_mode = 'copy', q_stride = 1, connectivity = 'chain', patch_embedding_required = 'true'
                    ):
        super().__init__()

        self.trainlosslist = []
        self.trauclist = []
        self.tracclist = []
        self.vallosslist = []
        self.auclist = []
        self.acclist = []
        self.attention_maps = []

        self.channels_last = channels_last
        self.RD = RD
        self.parallel = parallel
        self.parallel_mode = parallel_mode
        self.num_transformer_blocks = num_transformer_blocks
        self.Attention_N = Attention_N
        self.attention_selection = attention_selection
        self.starting_dim = num_channels * patch_size ** 2
        self.dropout_values = dropout
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        img_h, img_w = img_size

        # 2. Calculate patches along height and width separately
        num_patches_h = img_h // patch_size
        num_patches_w = img_w // patch_size

        # 3. Calculate total number of patches
        self.num_patches = num_patches_h * num_patches_w

        # 4. Determine the default selection amount (q_lr)
        # We use max(img_h, img_w) to maintain the original heuristic's linear scaling
        default_selection = max(img_h, img_w) // (2 * patch_size)

        self.q_lr = selection_amount if selection_amount is not None else default_selection
        self.quantum_mlp = quantum_mlp
        self.quantum_classification = quantum_classification
        self.special_cls = special_cls
        self.q_stride = q_stride
        self.connectivity = connectivity
        self.patch_embedding_required = patch_embedding_required
        self.patch_size = patch_size

        # Splitting an image into patches and linearly projecting these flattened patches can be
        # simplified as a single convolution operation, where both the kernel size and the stride size
        # are set to the patch size.
        self.patch_embedding = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_size
        )
        self.linear_after_patch_embedding = nn.Linear(hidden_size, hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.num_steps = 1 + self.num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_steps, hidden_size) * 0.02)
        self.dropout = nn.Dropout(self.dropout_values['embedding_pos'])

        
        # Transformer blocks with attention selection
        self.transformer_blocks = nn.ModuleList( [nn.ModuleList([TransformerBlock_Attention_Chosen_QMLP(hidden_size // self.RD**i, num_heads, mlp_hidden_size, hidden_size // self.RD**(i + 1) , 
                                                                                        Attention_N = self.Attention_N, quantum_mlp = self.quantum_mlp,
                                                                                        dropout = self.dropout_values, RD = self.RD, q_lr = self.q_lr,
                                                                                        attention_selection = self.attention_selection, special_cls = self.special_cls,
                                                                                        q_stride = self.q_stride, connectivity = self.connectivity)
                                            for i in range(num_transformer_blocks)]) for j in range(parallel) ] )

        self.layer_norm = nn.LayerNorm(hidden_size // (RD**(num_transformer_blocks)))  # Normalization after the last transformer block

        self.linear = nn.Linear( (hidden_size // (RD**(num_transformer_blocks)) ) * parallel, num_classes)
        self.linear2 = nn.Linear(num_classes,num_classes) if not self.quantum_classification else QuantumLayer(num_qubits=num_classes, graph=self.connectivity)

    def patch_embed_sample(self, x):
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        # x.shape = (batch_size, num_channels, img_size, img_size)
        assert x.shape[2] * x.shape[3] % (self.patch_embedding.kernel_size ** 2) == 0, "Image dimensions must be divisible by the patch size."
        x = self.patch_embedding(x)
        # x.shape = (batch_size, hidden_size, sqrt(num_patches), sqrt(num_patches))
        x = x.flatten(start_dim=2)
        # x.shape = (batch_size, hidden_size, num_patches)
        x = x.transpose(1, 2)
        # x.shape = (batch_size, num_patches, hidden_size)
        return x

    def get_patches_by_attention(self, x, parallel_branch = 0):
        """ 
        x: (batch_size, num_channels, img_size, img_size)
        ...
        returns: 
            gathered_patches: (batch_size, q_lr, hidden_size)
            sel_patch_indices_0_based: (batch_size, q_lr)
        """
        # x.shape = (batch_size, num_patches, hidden_size)
        x_embedded = self.patch_embed_sample(x) 
        
        # Positional embedding
        if  x_embedded.shape[1:] == self.pos_embedding.shape[1:]:
            x_with_pos = x_embedded + self.pos_embedding[:,1:,:].to(x_embedded.device) 

        elif x_embedded.shape[1] % (self.pos_embedding.shape[1] - 1)== 0:
            x_with_pos = x_embedded + self.pos_embedding[:,1:,:].repeat(1, x_embedded.shape[1] // (self.pos_embedding.shape[1]-1), 1) # [B, S, D]

        else:
            print(f"Warning, skipping positional_embedding as shapes of [input tensor,cls_token]: {x_embedded.shape} and pos_embedding {self.pos_embedding.shape} don't match")
            x_with_pos = x_embedded

        # CLS token
        # We want to add the cls token so that it takes part in the attention calculation
        cls_token = (self.cls_token + self.pos_embedding[:,0,:]).expand(x_with_pos.shape[0], -1, -1).to(x_with_pos.device)
        x_with_cls_and_pos = torch.cat((cls_token, x_with_pos), dim=1) 
        # x.shape = (batch_size, num_steps, hidden_size)
   
        # Attention block
        attn_input = self.transformer_blocks[parallel_branch][0].attn_norm(x_with_cls_and_pos)
        # Note: Your original code assumes attn returns (output, map)
        _, attn_map = self.transformer_blocks[parallel_branch][0].attn(attn_input)

        # Rank patches by attention
        attn_indices = rank_patches_by_attention(attn_map)


        # Remove CLS token (index 0) from selection and select top q_lr
        sel_indices_with_cls_offset = torch.stack( [ attn_indices[i][ attn_indices[i] != 0 ][:self.q_lr] for i in range(attn_indices.size(0)) ])

 
        # Convert to 0-based patch indices (by subtracting 1 for the CLS token)
        # This is the index relative to the *original* patch list (0 to num_patches-1)
        sel_patch_indices_0_based = sel_indices_with_cls_offset - 1

        # Gather the embedded patches
        gathered_patches = x_embedded.gather(1, sel_patch_indices_0_based.unsqueeze(-1).expand(-1, -1, x_embedded.size(-1)) ) # Shape: (batch_size, q_lr, hidden_size)

        
        return gathered_patches, sel_patch_indices_0_based


    def reconstruct_image_from_patches(self, patches, sel_patch_indices_0_based, original_image_shape):
        """
        Reconstructs images from patches matching the (B, Q, S, C, P, P) pipeline structure.
        
        Args:
            patches (torch.Tensor): The patch data. 
                                    Expected Shapes:
                                    - 6D: (B, Q, S, C, P, P) -> (Spatial)
                                    - 4D: (B, Q, S, C*P*P)   -> (Flattened)
            sel_patch_indices_0_based (torch.Tensor): Indices of selected patches.
                                    Shape: (B, S)
            original_image_shape (tuple): (C, H, W) or (B, C, H, W).
            
        Returns:
            torch.Tensor: Reconstructed images of shape (B, Q, C, H, W).
        """
        # --- 1. Setup Device and Dimensions ---
        # Ensure we are working on the device where the heavy patch data is
        device = patches.device
        
        # Move indices to the correct device immediately
        sel_patch_indices_0_based = sel_patch_indices_0_based.to(device)

        # Handle Original Image Shape
        if len(original_image_shape) == 4:
            C_img, H, W = original_image_shape[1], original_image_shape[2], original_image_shape[3]
        else:
            C_img, H, W = original_image_shape

        # Get Patch dimensions from class
        try:
            P_h, P_w = self.patch_size, self.patch_size
        except AttributeError:
            # Fallback if self.patch_size isn't set, try to infer from input if 6D
            if patches.dim() == 6:
                P_h, P_w = patches.shape[-2], patches.shape[-1]
            else:
                raise AttributeError("self.patch_size must be defined or input must be 6D.")

        # --- 2. Parse Input Tensor Dimensions ---
        # We expect the structure: (Batch, Quantum_Channels, Selected_Patches, ...)
        B = patches.shape[0]
        Q = patches.shape[1]
        S = patches.shape[2] # This is q_lr (selection amount)
        
        # Calculate grid specs
        grid_h = H // P_h
        grid_w = W // P_w
        num_patches_total = grid_h * grid_w
        pixel_dim = C_img * P_h * P_w

        # --- 3. Normalize Input to (B*Q, S, C, P, P) ---
        # We merge B and Q to treat them as a "large batch" for parallel reconstruction
        
        if patches.dim() == 6:
            # Input is (B, Q, S, C, P, P)
            # Reshape to (B*Q, S, C, P, P)
            pixel_patches = patches.view(B * Q, S, C_img, P_h, P_w)
            
        elif patches.dim() == 4:
            # Input is (B, Q, S, flat_pixels)
            # Verify dimensions
            if patches.shape[-1] != pixel_dim:
                raise ValueError(f"Flattened dim {patches.shape[-1]} != Expected {pixel_dim}")
            
            # Unflatten and merge B*Q
            pixel_patches = patches.reshape(B * Q, S, C_img, P_h, P_w)
            
        else:
            raise ValueError(f"Unexpected patch tensor dimension: {patches.dim()}. Expected 4 or 6.")

        # --- 4. Prepare Indices ---
        # Indices are currently (B, S). We need them to be (B*Q, S).
        # Logic: The same indices apply to all Q variations of a specific image.
        
        # (B, S) -> (B, 1, S) -> (B, Q, S) -> (B*Q, S)
        indices_expanded = sel_patch_indices_0_based.unsqueeze(1).expand(-1, Q, -1)
        indices_merged = indices_expanded.reshape(B * Q, S)
        
        # Expand for scatter: (B*Q, S) -> (B*Q, S, 1, 1, 1) -> (B*Q, S, C, P, P)
        idx_scatter = indices_merged.view(B * Q, S, 1, 1, 1).expand(-1, -1, C_img, P_h, P_w)

        # --- 5. Scatter to Canvas ---
        # Create empty canvas: (B*Q, Total_Patches, C, P, P)
        canvas_patches = torch.zeros(
            (B * Q, num_patches_total, C_img, P_h, P_w),
            device=device,
            dtype=patches.dtype
        )
        
        # Scatter the pixel data onto the canvas at the correct patch positions
        canvas_patches.scatter_(dim=1, index=idx_scatter, src=pixel_patches)

        # --- 6. Fold (Reconstruct Spatial Image) ---
        # Reshape for Fold: (B*Q, N, C, P, P) -> (B*Q, C*P*P, N)
        # Note: Fold expects (Batch, Channels * Kernel * Kernel, L)
        canvas_flat = canvas_patches.view(B * Q, num_patches_total, pixel_dim).transpose(1, 2)
        
        fold = nn.Fold(output_size=(H, W), kernel_size=(P_h, P_w), stride=(P_h, P_w))
        
        # Result: (B * Q, C, H, W)
        reconstructed_combined = fold(canvas_flat)

        # --- 7. Final Reshape ---
        # Separate Batch and Quantum dimensions: (B, Q, C, H, W)
        reconstructed_final = reconstructed_combined.view(B, Q, C_img, H, W)
        
        return reconstructed_final
    
    def get_selected_pixel_patches_indices(self, images, patch_indices, quantum_channels=0, originals=True):
        """
        Helper to extract raw pixel patches from original images corresponding to selected indices.
        Useful for visualizing which parts of the image were selected.

        Args:
            images (torch.Tensor): Original images (B, C, H, W)
            patch_indices (torch.Tensor): Indices to select (B, q_lr)
            quantum_channels (int): Number of quantum variations (Q)
            originals (bool): Include originals?

        Returns:
            torch.Tensor: Selected pixel patches formatted for reconstruction 
                          Shape: (B, q_lr * Q, C * patch_size**2)
        """
        # Ensure indices are on the same device as images to prevent indexing errors
        patch_indices = patch_indices.to(images.device)

        B, C, H, W = images.shape
        P = self.patch_size
        
        # 1. Unfold image into all patches: (B, C*P*P, N_patches)
        # We transpose to (B, N_patches, C*P*P) to make gathering easier
        all_patches_flat = F.unfold(images, kernel_size=P, stride=P).transpose(1, 2)
        
        # 2. Gather the specific patches using the indices
        # patch_indices is (B, q_lr). We gather along dim 1.
        # We expand indices to (B, q_lr, flat_dim)
        flat_dim = all_patches_flat.shape[2]
        
        # Use torch.gather or advanced indexing. 
        # Advanced indexing is cleaner here:
        batch_indices = torch.arange(B, device=images.device).unsqueeze(1) # (B, 1)
        selected_pixels = all_patches_flat[batch_indices, patch_indices]   # (B, q_lr, flat_dim)
        
        # 3. Expand for Quantum Channels (Q)
        # Since these are original pixels, we just repeat them for each 'channel' visualization
        assert quantum_channels + originals > 0
        Q = int(quantum_channels + originals)
        
        # (B, q_lr, flat_dim) -> (B, Q, q_lr, flat_dim)
        selected_pixels_expanded = selected_pixels.unsqueeze(1).expand(-1, Q, -1, -1)
        
        # Flatten Q and q_lr to match expected input: (B, Q * q_lr, flat_dim)
        selected_patches_flat = selected_pixels_expanded.reshape(B * Q * patch_indices.size(1), C, P, P)
        
        return selected_patches_flat


    def forward(self, x):

        shape = x.shape

        if self.patch_embedding_required == 'true':
            x = self.patch_embed_sample(x)

        elif self.patch_embedding_required == 'flatten':
            x = x.view((x.shape[0], x.shape[1], -1))        
        
        x = self.linear_after_patch_embedding(x)

        # Positional embedding
        if  x.shape[1:] == self.pos_embedding.shape[1:]:
            x = x + self.pos_embedding.to(x.device) 

        elif x.shape[1] % (self.num_patches) == 0:
            x = x + self.pos_embedding[:, 1:, :].repeat(1, x.shape[1] // (self.num_patches), 1) # [B, S, D]

        # Repeat x for each parallel branch
        if self.parallel_mode == 'copy':
             # CLS token
            cls_token = (self.cls_token + self.pos_embedding[:, 0, :]).expand( x.shape[0], -1, -1 )
            x = torch.cat(( cls_token, x), dim=1)
            # x.shape = (batch_size, num_steps, hidden_size)
            x_parallel = x.unsqueeze(0).repeat(self.parallel, 1, 1, 1)  # [P, B, S, D]

        elif self.parallel_mode == 'quantum':
            Q = self.parallel
            assert shape[-2] % self.parallel == 0, \
                f"When parameter 'paralel_mode' = 'quantum', parameter 'paralel' is expected to be the amount of stacked versions of the original image (Q + concatenate_original)"

            x_aux = x.reshape( shape[0], Q , -1, shape[-1] ).permute(1, 0, 2, 3).contiguous() # [Q, B, S, D] , Q = Quantum versions of the image
            cls_token = (self.cls_token + self.pos_embedding[:, 0, :]).unsqueeze(0).expand( Q, x.shape[0], -1, -1 )
            x_parallel = torch.cat([cls_token, x_aux], dim = -2)
        else:   
            raise ValueError(f"Expected values 'quantum' or 'copy' for input argument 'parallel_mode' but instead got {self.parallel_mode}")

        x_parallel = self.dropout(x_parallel)

        attn_maps = []
        outputs = []

        for i in range(self.parallel):
            out = x_parallel[i]  # [B, S, D]
            for j in range(self.num_transformer_blocks):
                out, attn = self.transformer_blocks[i][j](out)  # [B, S, D], attn: [B, H, S, S] or similar #type: ignore
                attn_maps.append(attn)

            out = self.layer_norm(out)         # [B, S, D]
            out = out[:, 0]                    # [B, D]
            outputs.append(out)                # Collect [B, D]

        # Concatenate along hidden dimension
        x = torch.cat(outputs, dim=1)  # [B, D * P]

        # Classification logits
        x = self.linear(x)
        x = self.linear2(x)

        # x.shape = (batch_size, num_classes)
        return x, attn_maps

    def save_reconstructed_after_selection(self , notrans_train_dl : torch.utils.data.DataLoader , save_path = "prov/selected_dataset",n_batches : int = 1 ) -> None:
        from pathlib import Path
        from PIL import Image
        import numpy as np

        save_path_rec = Path(save_path + "/reconstructed")
        save_path_ori = Path(save_path + "/ori")
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path_rec.mkdir(parents=True, exist_ok=True)
        save_path_ori.mkdir(parents=True, exist_ok=True)
        
        param = next(self.transformer_blocks[0][0].parameters(), None)
        if param is not None:
            device = param.device
        else:
            raise ValueError("Could not acces model device")

        count = 0

        for img, lbl, idx in notrans_train_dl:
            count += 1
            # Move batch to the same device as the model (use non_blocking if dataloader has pin_memory=True)
            img = img.to(device, non_blocking=True)
            shape = img.shape if img.ndim == 3 else img.shape[1:] if img.ndim == 4 else (1, *img.shape)
            _, indices_sel = self.get_patches_by_attention(img)
            imgs_sel = self.get_selected_pixel_patches(img, indices_sel)
            reconstructed_imgs = self.reconstruct_image_from_patches(imgs_sel,indices_sel, shape)
            # reconstructed_imgs is expected as a torch.Tensor with shape (B, C, H, W) or (B, H, W, C)
            for b_i in range(reconstructed_imgs.shape[0]):
                recon = reconstructed_imgs[b_i].detach().cpu().numpy()
                # If channel-first (C,H,W) -> convert to H,W,C
                if recon.ndim == 3 and recon.shape[0] in (1, 3):
                    recon = np.transpose(recon, (1, 2, 0))
                # If single-channel with last dim == 1 -> squeeze
                if recon.ndim == 3 and recon.shape[2] == 1:
                    recon = recon[:, :, 0]
                # Normalize to 0..255 uint8
                minv, maxv = float(recon.min()), float(recon.max())
                if maxv <= 1.0 and minv >= 0.0:
                    img_uint8 = (recon * 255.0).astype(np.uint8)
                else:
                    rng = maxv - minv + 1e-8
                    img_uint8 = ((recon - minv) / rng * 255.0).astype(np.uint8)
                # Create PIL image (grayscale or RGB)
                if img_uint8.ndim == 2:
                    im = Image.fromarray(img_uint8, mode='L')
                else:
                    if img_uint8.shape[2] > 3:
                        img_uint8 = img_uint8[:, :, :3]
                    im = Image.fromarray(img_uint8)
                # Try to resolve a dataset index from the dataloader 'idx' (tensor or list), fallback to batch-local index
                try:
                    sample_idx = int(idx[b_i].item())
                except Exception:
                    try:
                        sample_idx = int(idx[b_i])
                    except Exception:
                        sample_idx = b_i
                # Save reconstructed image
                fname = save_path_rec / f"recon_{sample_idx}_{b_i}.png"
                im.save(str(fname))
                # Also save the original input image in save_path_ori using the same index
                try:
                    orig = img[b_i].detach().cpu().numpy()
                except Exception:
                    # fallback if img is already numpy or other format
                    orig = np.array(img[b_i])
                if orig.ndim == 3 and orig.shape[0] in (1, 3):
                    orig = np.transpose(orig, (1, 2, 0))
                if orig.ndim == 3 and orig.shape[2] == 1:
                    orig = orig[:, :, 0]
                minv_o, maxv_o = float(orig.min()), float(orig.max())
                if maxv_o <= 1.0 and minv_o >= 0.0:
                    orig_uint8 = (orig * 255.0).astype(np.uint8)
                else:
                    rng_o = maxv_o - minv_o + 1e-8
                    orig_uint8 = ((orig - minv_o) / rng_o * 255.0).astype(np.uint8)
                if orig_uint8.ndim == 2:
                    im_o = Image.fromarray(orig_uint8, mode='L')
                else:
                    if orig_uint8.shape[2] > 3:
                        orig_uint8 = orig_uint8[:, :, :3]
                    im_o = Image.fromarray(orig_uint8)
                fname_o = save_path_ori / f"origin_{sample_idx}_{b_i}.png"
                im_o.save(str(fname_o))
            # Optional: break after first batch when testing to avoid saving whole dataset
            if count >= n_batches:
                break
    
class Encoder(nn.Module):
    def __init__(self, encoder_layers, dropout_pos):
        super(Encoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.dropout_pos = dropout_pos
        self.parallel = len(self.encoder_layers)
        self.num_transformer_blocks = len(self.encoder_layers[0])


    def forward(self, x, pos_embedding):
        # Apply patch and position embeddings, including the class token
        
        x += pos_embedding[:, :(x.shape[1])]
        out = self.dropout_pos(x)
        # Repeat x for each parallel branch
        x_parallel = x.unsqueeze(0).repeat(self.parallel, 1, 1, 1)  # [P, B, S, D]

        last_layers_outputs = []
        outputs = []

        for i in range(self.parallel):
            out = x_parallel[i]  # [B, S, D]

            for j in range(self.num_transformer_blocks):
                out, _ = self.encoder_layers[i][j](out)  # [B, S, D], attn: [B, H, S, S] or similar #type: ignore

            outputs.append(out)
            if type(out) != torch.Tensor:
                raise ValueError("The output is not a tensor.")
            
        if type(out) != torch.Tensor:
            raise ValueError("The output is not a tensor.")

        return torch.stack(outputs, dim = 0)  # Shape: (parallel, batch_size, num_patches + 1, hidden_size)

class Decoder(nn.Module):
    def __init__(self, decoder_layers, mlp_hidden_size, hidden_size):
        super(Decoder, self).__init__()
        self.decoder_layers = decoder_layers
        self.parallel = len(self.decoder_layers)
        self.num_transformer_blocks = len(self.decoder_layers[0])
        self.mix_parallels_info = nn.Linear(self.parallel * mlp_hidden_size, hidden_size)


    def forward(self, z):
        outputs = []
        # Pass the latent representation through each decoder block
        last_layers_outputs = []
        outputs = []

        for i in range(self.parallel):
            out = z[i]  # [B, S, D]

            for j in range(self.num_transformer_blocks - 1):
                out, _ = self.decoder_layers[i][j](out)  # [B, S, D], attn: [B, H, S, S] or similar #type: ignore

            last_layer = self.decoder_layers[i][-1]
            attn_input = last_layer.attn_norm(out)
            attn_output, attn_map = last_layer.attn(attn_input)
            attn_output = last_layer.attn_dropout(attn_output)
            out = out + attn_output

            y = last_layer.mlp_norm(out)
            mlp_out = last_layer.mlp_sel.fc1(y)
            mlp_out = last_layer.mlp_sel.vqc(mlp_out)
            mlp_out = mlp_out.to(y.device)  # Ensure the output is on the same device as the input
            mlp_out = last_layer.mlp_sel.dropout(mlp_out)
            outputs.append( last_layer.mlp_sel.gelu(mlp_out) ) # Here shape is [B, S, mlp_hidden_size]
        
        pre_mixing_parallel_info = torch.cat(outputs, dim = -1) # Here shape is [B, S, mlp_hidden_size * parallel]
        mixed_out = self.mix_parallels_info(pre_mixing_parallel_info)

        # Concatenate outputs from all parallel branches
        return mixed_out
    
class AutoEnformer(nn.Module):
            def __init__(self, img_size, num_channels, patch_size, hidden_size, num_heads, num_transformer_blocks ,mlp_hidden_size,
                              Attention_N = 2, dropout={'embedding_attn': 0.225, 'after_attn': 0.225, 'feedforward': 0.225, 'embedding_pos': 0.225}, channels_last=False, attention_selection='none', RD=1,
                              q_stride = 1, parallel = 1):
                super(AutoEnformer, self).__init__()

                self.channels_last = channels_last
                self.RD = RD
                self.trainlosslist = []
                self.vallosslist = []
                self.parallel = parallel
                self.num_transformer_blocks = num_transformer_blocks
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.mlp_hidden_size = mlp_hidden_size
                self.Attention_N = Attention_N
                self.attention_selection = attention_selection
                self.starting_dim = num_channels * patch_size ** 2
                self.dropout_values = dropout
                self.num_channels = num_channels
                self.q_stride = q_stride

                self.patch_embedding = nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size
                )
                num_patches = (img_size // patch_size)**2

                num_steps = 1 + num_patches

                self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, hidden_size) * 0.02)
                self.dropout = nn.Dropout(self.dropout_values['embedding_pos'])

                self.encoder_layers = nn.ModuleList( [nn.ModuleList ([TransformerBlock_Attention_Chosen_QMLP(self.hidden_size // self.RD**i, self.num_heads, self.mlp_hidden_size, self.hidden_size // self.RD**(i + 1) , 
                                                                                                    Attention_N = self.Attention_N , quantum_mlp = False,
                                                                                                    dropout = self.dropout_values,
                                                                                                    attention_selection = 'none',
                                                                                                    q_stride = self.q_stride )
                                                        for i in range(self.num_transformer_blocks)])  for j in range(self.parallel) ] )
                
                self.decoder_layers = nn.ModuleList( [nn.ModuleList ([TransformerBlock_Attention_Chosen_QMLP(self.hidden_size // self.RD**i, self.num_heads, self.mlp_hidden_size, self.hidden_size // self.RD**(i + 1) ,
                                                                                            Attention_N = self.Attention_N , quantum_mlp = False,
                                                                                            dropout = self.dropout_values,
                                                                                            attention_selection = 'none',
                                                                                            q_stride = self.q_stride )
                                                        for i in range(self.num_transformer_blocks, 0, -1) ]) for j in range(self.parallel) ] )
                                        
                self.Encoder = Encoder(self.encoder_layers, self.dropout)
                self.Decoder = Decoder(self.decoder_layers, self.mlp_hidden_size, self.hidden_size)

            def get_latent_representation(self, img):

                # Check and handle channel-last format if needed
                if self.channels_last:
                    img = img.permute(0, 3, 1, 2)
                
                # Create image patches and project them to the hidden size
                x = self.patch_embedding(img)
                x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N is number of patches
                # Apply patch and position embeddings, including the class token
                x += self.pos_embedding[:, :(x.shape[1])]
                out = self.dropout(x)
                
                # Repeat x for each parallel branch
                x_parallel = x.unsqueeze(0).repeat(self.parallel, 1, 1, 1)  # [P, B, S, D]

                outputs = []

                for i in range(self.parallel):
                    out = x_parallel[i]  # [B, S, D]

                    for j in range(self.num_transformer_blocks - 1):
                        out, _ = self.Encoder.encoder_layers[i][j](out)  # [B, S, D], attn: [B, H, S, S] or similar #type: ignore
                    
                    last_layer = self.Encoder.encoder_layers[i][-1]
                    attn_input = last_layer.attn_norm(out)
                    attn_output, attn_map = last_layer.attn(attn_input)
                    attn_output = last_layer.attn_dropout(attn_output)
                    out = out + attn_output

                    y = last_layer.mlp_norm(out)
                    mlp_out = last_layer.mlp_sel.fc1(y)
                    mlp_out = last_layer.mlp_sel.vqc(mlp_out)
                    mlp_out = mlp_out.to(y.device)  # Ensure the output is on the same device as the input
                    mlp_out = last_layer.mlp_sel.dropout(mlp_out)
                    outputs.append( last_layer.mlp_sel.gelu(mlp_out) )
                
                latent_representations =  torch.stack(outputs, dim=0)

                return latent_representations  # Shape: (parallel, batch_size, num_patches + 1, mlp_hidden_size)
                    
            
                        
            def forward(self, img):
                
                # 1. Prepare the input image
                # Check and handle channel-last format if needed
                if self.channels_last:
                    img = img.permute(0, 3, 1, 2)
                
                # Create image patches and project them to the hidden size
                x = self.patch_embedding(img)
                x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N is number of patches
                
                # 2.Run the Encoder and get latent representations

                latent_representations = self.Encoder(x, self.pos_embedding)

                # 3. Run the Decoder and recoconstruct original patches
                
                reconstructed_patches = self.Decoder(latent_representations)

                # 4. Final Reconstruction: reshape the tensor so that you get the original image from the patches             
                
                # Transpose to get the channel dimension for reshaping
                reconstructed_patches = reconstructed_patches.transpose(1, 2)
                
                # Reshape to a 4D tensor
                reconstructed_imgs = reconstructed_patches.reshape(img.shape)
                
                return reconstructed_imgs  # Return the reconstructed images 
            

class DeViT(nn.Module):
            """ Vision Transformer for classification on top of latent representations.
                First, a linear layer adjusts the dimension of the latent representation to match the ViT input size.
                Then, a standard ViT is applied for classification.
            """
            def __init__(self, num_classes, p, shape, dim_latent):
                super(DeViT, self).__init__()

                self.num_classes = num_classes
                self.p = p
                self.shape = shape
                self.dimension_adjustment = nn.Linear(dim_latent, p['hidden_size']) if dim_latent != p['hidden_size'] else nn.Identity()
                self.dim_latent = dim_latent

                self.trainlosslist = []
                self.trauclist = []
                self.tracclist = []
                self.vallosslist = []
                self.auclist = []
                self.acclist = []
                self.attention_maps = []

                self.vit = VisionTransformer(
                    img_size=shape[-1], num_channels=shape[0], num_classes=num_classes,
                    patch_size=p['patch_size'], hidden_size= p['hidden_size'], num_heads=p['num_head'], Attention_N = p['Attention_N'],
                    num_transformer_blocks=p['num_transf'], attention_selection= p['attention_selection'], special_cls = p['special_cls'], 
                    mlp_hidden_size=p['mlp_size'], quantum_mlp = False, dropout = p['dropout'], channels_last=False, quantum_classification = False,
                    parallel = p['parallel'], RD = p['RD'], q_stride = p['q_stride'], connectivity = 'chain', patch_embedding_required= False
                )

                
            def forward(self, x):  
                
                assert x.shape[-1] == self.dim_latent, f"Input feature dimension ({x.shape[-1]}) does not match expected size ({self.dim_latent})"
                x = self.dimension_adjustment(x)  
            
                return self.vit(x)  # Skip patch embedding in ViT

