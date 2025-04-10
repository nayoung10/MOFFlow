import torch
import torch.nn as nn
from torch_geometric.nn.pool import (
    radius_graph,
    global_mean_pool
)

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
    def _pairwise_distances(edge_index, pos):
        """        
        Args:
            edge_index (Tensor): [2, E] edge index
            pos (Tensor): [N, 3] node positions
        Returns:
            dists (Tensor): [E] pairwise distances
        """
        row, col = edge_index
        dists = torch.norm(pos[col] - pos[row], p=2, dim=-1)

        return dists

    @staticmethod
    def _repeat_interleave(repeats):
        outs = [torch.full((n, ), i) for i, n in enumerate(repeats)]
        return torch.cat(outs, dim=0).to(repeats.device)
    
    def forward(self, batch):
        local_coords = batch['local_coords']    # [N, 3]

        # Compute node features
        node_attr = self.atom_type_embedder(batch['atom_types'] - 1)     # [N, D]

        # Compute edge index
        batch_bb = self._repeat_interleave(batch['bb_num_vec'])
        bb_edge_index = radius_graph(
            x=local_coords, 
            r=self._bb_cfg.max_radius, 
            batch=batch_bb,
            loop=False,
            max_num_neighbors=self._bb_cfg.max_neighbors)

        # Compute edge distance features
        pair_dist = self._pairwise_distances(bb_edge_index, local_coords)    # [E]
        bb_edge_feats = self.edge_dist_embedder(pair_dist)                   # [E, D]
                     
        # Message passing
        local_coords = local_coords * du.ANG_TO_NM_SCALE
        for i in range(len(self.egnn_layers)):
            node_update, _ = self.egnn_layers[i](
                h=node_attr,
                coord=local_coords,
                edge_index=bb_edge_index,
                edge_attr=bb_edge_feats
            )

            node_attr = node_attr + node_update
        
        # Pooling
        bb_attr = global_mean_pool(node_attr, batch_bb)  # [M, D]
        
        return bb_attr