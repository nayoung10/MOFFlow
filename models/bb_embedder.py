import torch
import torch.nn as nn
from einops import rearrange
from torch_cluster import radius_graph
from torch_geometric.nn.pool import global_mean_pool

from data import utils as du
from models.egnn import E_GCL



class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class BuildingBlockEmbedder(nn.Module):

    def __init__(self, bb_cfg):
        super().__init__()
        self._bb_cfg = bb_cfg
        self.atom_type_embedder = torch.nn.Embedding(bb_cfg.max_atoms, bb_cfg.c_node_dim)
        self.edge_dist_embedder = GaussianSmearing(0.0, bb_cfg.max_radius, bb_cfg.c_edge_dim)    

        self.egnn_layers = nn.ModuleList()
        for i in range(bb_cfg.num_layers):
            self.egnn_layers.append(
                E_GCL(
                    input_nf=bb_cfg.c_node_dim,
                    output_nf=bb_cfg.c_node_dim,
                    hidden_nf=bb_cfg.c_hidden_dim,
                    edges_in_d=bb_cfg.c_edge_dim,
                    act_fn=nn.ReLU(),
                    residual=True,
                    attention=False,
                    normalize=False,
                    coords_agg='mean',
                    tanh=False
                )
            )

    @staticmethod
    def _pairwise_distances(x):
        """        
        Args:
            x (torch.Tensor): Tensor of shape [B, N, 3], representing the coordinates.
        """

        dists = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # Shape: [B, N, N]
        return dists

    def forward(self, batch):
        batch_size, num_atoms = batch['atom_types'].shape
        atom_types = batch['atom_types']        # [B, N]
        local_coords = batch['local_coords']    # [B, N, 3]
        bb_num_vec = batch['bb_num_vec'][0]     # [M]
        device = local_coords.device

        # Compute node features
        node_attr = self.atom_type_embedder(atom_types - 1)     # [B, N, D]
        node_attr = rearrange(node_attr, 'b n d -> (b n) d')    # [B*N, D]

        # Compute edge distance features
        pair_dist = self._pairwise_distances(local_coords)      # [B, N, N]
        edge_feats = self.edge_dist_embedder(pair_dist)         # [B*N*N, D]
        edge_feats = rearrange(edge_feats, '(b n1 n2) d -> b n1 n2 d', b=batch_size, n1=num_atoms)  # [B, N, N, D]

        # Compute edge index
        bb_vector = torch.cat([torch.full((num,), i) for i, num in enumerate(bb_num_vec)]).int()
        bb_vector = torch.tensor([i + b*len(bb_num_vec) for b in range(batch_size) for i in bb_vector]).to(device)
        local_coords = rearrange(batch['local_coords'], 'b n d -> (b n) d')
        intra_edge_index = radius_graph(local_coords, r=self._bb_cfg.max_radius, batch=bb_vector, loop=False)

        # Select edge attributes 
        row, col = intra_edge_index
        batch_vec = torch.arange(batch_size, device=device).repeat_interleave(num_atoms)
        batch_indices = batch_vec[row]
        intra_edge_attr = edge_feats[batch_indices, row % num_atoms, col % num_atoms]
        assert intra_edge_attr.shape[0] == intra_edge_index.shape[-1]

        # Message passing
        local_coords = local_coords * du.ANG_TO_NM_SCALE
        for i in range(len(self.egnn_layers)):
            node_update, _ = self.egnn_layers[i](
                h=node_attr,
                coord=local_coords,
                edge_index=intra_edge_index,
                edge_attr=intra_edge_attr
            )

            node_attr = node_attr + node_update
        
        # Pooling
        bb_attr = global_mean_pool(node_attr, bb_vector.long())
        bb_attr = rearrange(bb_attr, '(b m) d -> b m d', b=batch_size)
        # assert torch.allclose(bb_attr[0], bb_attr[-1]), f"bb_attr[0]: {bb_attr[0]}, bb_attr[-1]: {bb_attr[-1]}"

        return bb_attr