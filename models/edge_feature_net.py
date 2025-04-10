import torch
from torch import nn

from torch_geometric.nn.pool import radius_graph
from models.utils import calc_distogram

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)

        # Compute final edge feature dimension
        total_edge_feats = self.feat_dim * 2 + self._cfg.num_bins * 2

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def _cross_concat(self, feats_1d, num_batch, num_bb):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_bb, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_bb, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_bb, num_bb, -1])
        
    @staticmethod
    def _pairwise_distances(edge_index, pos):
        """        
        Args:
            edge_index (Tensor): [2, E] edge index
            pos (Tensor): [N, 3] node positions
        """
        row, col = edge_index
        dists = torch.norm(pos[col] - pos[row], p=2, dim=-1)

        return dists

    @staticmethod
    def _calc_distogram(dist, min_bin, max_bin, num_bins):
        dist = dist[..., None]      # [E, 1]
        lower = torch.linspace(
            min_bin,
            max_bin,
            num_bins,
            device=dist.device)
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        dgram = ((dist > lower) * (dist < upper)).type(dist.dtype)
        return dgram
    
    def forward(self, batch_vector, init_node_embed, trans_t, trans_sc):
        # Compute edge index
        edge_index = radius_graph(
            x=trans_t,
            r=self._cfg.max_radius,
            batch=batch_vector,
            loop=True,
            max_num_neighbors=self._cfg.max_neighbors
        )
        
        # Source and target node features
        src, tgt = edge_index
        p_i = self.linear_s_p(init_node_embed)
        cross_node_feats = torch.cat([p_i[tgt], p_i[src]], dim=-1)   # [E, feat_dim * 2]
    
        # Edge features
        dist_t = self._pairwise_distances(edge_index, trans_t)      # [E]
        dist_sc = self._pairwise_distances(edge_index, trans_sc)    # [E]
        dist_feats = self._calc_distogram(
            dist_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)   # [E, num_bins]
        sc_feats = self._calc_distogram(
            dist_sc, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)  # [E, num_bins]

        # Aggregate edge features
        all_edge_feats = [cross_node_feats, dist_feats, sc_feats]
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        return edge_feats, edge_index