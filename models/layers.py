# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified code of Openfold's IPA."""

import copy
import numpy as np
import torch
import math
import torch.nn as nn
from scipy.stats import truncnorm
from einops import rearrange
from torch_geometric.utils import scatter, softmax
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed, edge_index):
        """
        Args:
            node_embed (Tensor): [M, C]
            edge_embed (Tensor): [E, C]
            edge_index (Tensor): [2, E]
        """
        src, tgt = edge_index
        
        # Extract node features
        node_embed = self.initial_embed(node_embed)
        h_src = node_embed[src] # [E, C]
        h_tgt = node_embed[tgt] # [E, C]
    
        # Concatenate
        edge_embed = torch.cat([edge_embed, h_src, h_tgt], dim=-1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        
        return edge_embed


class MOFPointAttention(nn.Module):
    """
    Modification of InvariantPointAttention (OpenFold)
    """
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(MOFPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.linear_l = Linear(6, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_hidden + self.no_v_points * 3
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim + 6, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    @staticmethod
    def _pairwise_distances(edge_index, pos_source, pos_target):
        """        
        Args:
            edge_index (Tensor): [2, E] edge index
            pos_source (Tensor): [M, H, P, 3] node positions
            pos_target (Tensor): [M, H, P, 3] node positions
        Returns:
            dists (Tensor): [E, H, P] pairwise distances
        """
        row, col = edge_index
        dists = torch.norm(pos_source[col] - pos_target[row], p=2, dim=-1)

        return dists
    
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        L: torch.Tensor,
        edge_index: torch.Tensor,
        num_bbs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [M, C_s] single representation
            z:
                [E, C_z] pair representation
            r:
                [M] transformation object
            L:
                [B, 6] lattice parameters with angles in radians
            edge_index:
                [2, E] edge index
            num_bbs:
                [B] number of building blocks
        Returns:
            [M, C_s] single representation update
        """
        #######################################
        # Generate scalar and point activations
        #######################################
        # Compute q, k, v
        q = self.linear_q(s)    # [M, H * C_hidden]
        kv = self.linear_kv(s)  # [M, H * 2 * C_hidden]

        # Rearrange shapes
        q = rearrange(q, "m (h c) -> m h c", h=self.no_heads)       # [M, H, C_hidden]
        kv = rearrange(kv, "m (h d) -> m h d", h=self.no_heads)     # [M, H, 2 * C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)               # [M, H, C_hidden]
        
        # Compute q_pts
        q_pts = self.linear_q_points(s)                                     # [M, H * P_q * 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)                                  # [M, H * P_q, 3]
        q_pts = r[..., None].apply(q_pts)                                   # [M, H * P_q, 3]
        q_pts = rearrange(q_pts, "m (h p) d -> m h p d", h=self.no_heads)   # [M, H, P_q, 3]

        # Compute kv_pts
        kv_pts = self.linear_kv_points(s)                                   # [M, H * (P_q + P_v) * 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)                                # [M, H * (P_q + P_v), 3]
        kv_pts = r[..., None].apply(kv_pts)                                 # [M, H * (P_q + P_v), 3]
        kv_pts = rearrange(kv_pts, "m (h p) d -> m h p d", h=self.no_heads) # [M, H, (P_q + P_v), 3]

        # Split kv_pts into k_pts and v_pts
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )                                                                   # [M, H, P_q, 3], [M, H, P_v, 3]

        ##########################
        # Compute attention scores
        ##########################
        # Compute b, l
        b = self.linear_b(z)    # [E, H]
        l = self.linear_l(L)    # [B, H]
        l = l.repeat_interleave(num_bbs, dim=0)  # [M, H]

        # Compute attention scores
        src, tgt = edge_index
        
        # Query-key attention
        a = torch.einsum('ehc,ehc->eh', q[tgt], k[src])             # [E, H]
        a *= math.sqrt(1.0 / (4 * self.c_hidden))                   # [E, H]
        
        # Edge, lattice attention
        a += (math.sqrt(1.0 / 4) * b)                               # [E, H]
        assert torch.allclose(l[src], l[tgt])
        a += (math.sqrt(1.0 / 4) * l[src])                          # [E, H]
        
        # Point attention
        pt_displacement = q_pts[tgt] - k_pts[src]                   # [E, H, P_q, 3]
        pt_att = pt_displacement ** 2
        pt_att = sum(torch.unbind(pt_att, dim=-1))                  # [E, H, P_q]
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))              # [1, H, 1]
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (4 * (self.no_qk_points * 9.0 / 2))               # [1, H, 1]
        )
        pt_att = pt_att * head_weights                              # [E, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)                 # [E, H]
        a = a + pt_att
        a = softmax(a, tgt, dim=0)                                  # [E, H]

        ################
        # Compute output
        ################
        # Compute o 
        o_msg = a[..., None] * v[src]                               # [E, H, C_hidden]
        o = scatter(o_msg, tgt, dim=0, reduce='sum')                # [M, H, C_hidden]
        o = rearrange(o, "m h c -> m (h c)")                        # [M, H * C_hidden]

        # Compute o_pt
        o_pt_msg = a[..., None, None] * v_pts[src]                  # [E, H, P_v, 3]
        o_pt = scatter(o_pt_msg, tgt, dim=0, reduce='sum')          # [M, H, P_v, 3]
        o_pt = rearrange(o_pt, "m h p d -> m (h p) d")              # [M, (H * P_v), 3]

        # Compute o_l 
        o_l = L.repeat_interleave(num_bbs, dim=0)                   # [M, 6]

        # Concatenate o, o_pt, o_l
        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_l]             
        o_feats = torch.cat(o_feats, dim=-1)                        # [M, (H * C_hidden + H * P_v * 3 + 6)]
        
        # Compute s
        s = self.linear_out(o_feats)                                # [M, C_s]
        
        return s


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        
        # Parameters
        self.d_model = d_model
        self.nheads = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * self.nheads == self.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Modules
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_kv = nn.Linear(d_model, 2 * d_model)
        self.to_out = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        s: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s (Tensor): [M, C] node representation
            edge_index (Tensor): [2, E] edge index
        Returns:
            o (Tensor): [M, C] updated node representation
        """
        # Compute q, k, v
        q = self.linear_q(s)    # [M, H * C]
        kv = self.linear_kv(s)  # [M, H * 2 * C]
        
        # Rearrange shapes
        q = rearrange(q, "m (h c) -> m h c", h=self.nheads)       # [M, H, C]
        kv = rearrange(kv, "m (h d) -> m h d", h=self.nheads)     # [M, H, 2 * C]
        k, v = torch.split(kv, self.head_dim, dim=-1)             # [M, H, C]
        
        # Compute attention scores
        src, tgt = edge_index
        a = torch.einsum('ehc,ehc->eh', q[tgt], k[src])          # [E, H]
        a *= self.scale                                          # [E, H]
        a = softmax(a, tgt, dim=0)                               # [E, H]
        
        # Aggregate messages
        o_msg = a[..., None] * v[src]                            # [E, H, C]
        o = scatter(o_msg, tgt, dim=0, reduce='sum')             # [M, H, C]
        o = rearrange(o, "m h c -> m (h c)")                     # [M, H * C]
        
        return self.to_out(o)


class TransformerEncoderLayer(nn.Module):
    """
    Pre-ln Transformer encoder layer for pyg batch
    """
    def __init__(
        self, 
        d_model: int,
        nhead: int,
        dim_feedforward: int
    ):
        super(TransformerEncoderLayer, self).__init__()
        
        # Layers
        self.mha = MultiHeadAttention(d_model, nhead)
        self.mha_ln = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model))
        self.ffn_ln = nn.LayerNorm(d_model)
    
    def forward(
        self,
        s: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s (Tensor): [M, C] node representation
            edge_index (Tensor): [2, E] edge index
        Returns:
            s (Tensor): [M, C] updated node representation
        """
        # Multi-head attention
        s = self.mha(self.mha_ln(s), edge_index) + s
        
        # Feed-forward network
        s = self.ffn(self.ffn_ln(s)) + s
        
        return s

    
class TransformerEncoder(nn.Module):
    def __init__(self, 
        encoder_layer: nn.Module, num_layers: int):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        
    def forward(self, node_embed, edge_index):
        for layer in self.layers:
            node_embed = layer(node_embed, edge_index)
        
        return node_embed


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, use_rot_updates):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self._use_rot_updates = use_rot_updates
        update_dim = 6 if use_rot_updates else 3
        self.linear = Linear(self.c_s, update_dim, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [M, C_s] single representation
        Returns:
            [M, 6] update vector 
        """
        update = self.linear(s)

        return update


class LatticeOutput(nn.Module):

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(LatticeOutput, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, 6),
        )

    def forward(self, s: torch.Tensor, batch_vec: torch.Tensor):
        """
        Args:
            s (Tensor): [M, C_s] node representation
            batch_vec (Tensor): [M,] batch vector
        Returns:
            output (Tensor): [B, 6] output lattice matrix
        """
        s = scatter(s, batch_vec, dim=0, reduce='mean')         # [B, C_s]
        output = torch.nn.functional.softplus(self.output(s))   # [B, 6]

        return output
