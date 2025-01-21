import torch
from torch import nn

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
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2

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

    def forward(self, init_node_embed, trans_t, trans_sc, edge_mask, diffuse_mask):
        # Input: [b, num_bb, c_s]
        num_batch, num_bb, _ = init_node_embed.shape

        # [b, num_bb, num_bb, feat_dim * 2]
        p_i = self.linear_s_p(init_node_embed)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_bb)

        # [b, num_bb, num_bb, num_bins]
        dist_feats = calc_distogram(
            trans_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(
            trans_sc, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, dist_feats, sc_feats]
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_bb)
            all_edge_feats.append(diff_feat)
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= edge_mask.unsqueeze(-1)
        return edge_feats