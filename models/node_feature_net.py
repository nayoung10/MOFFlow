import torch
from torch import nn
from models.utils import get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_bb_input = self._cfg.c_bb_input
        self.c_bb_emb = self._cfg.c_bb_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.bb_embedder = nn.Linear(self.c_bb_input, self.c_bb_emb, bias=False)

        embed_size = self._cfg.c_bb_emb + self._cfg.c_timestep_emb * 2 + 1
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, res_mask, diffuse_mask, bb_emb):
        # s: [b]

        b, num_bb, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, num_bb, c_bb_emb]
        bb_emb = self.bb_embedder(bb_emb)
        bb_emb = bb_emb * res_mask.unsqueeze(-1)

        # [b, num_bb, c_timestep_emb]
        input_feats = [
            bb_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask)
        ]

        return self.linear(torch.cat(input_feats, dim=-1))
