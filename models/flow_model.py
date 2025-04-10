
import torch
from torch import nn
from torch_scatter import scatter_mean
from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.bb_embedder import BuildingBlockEmbedder
from models import layers
from data import utils as du


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._mpa_conf = model_conf.mpa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.bb_embedder = BuildingBlockEmbedder(model_conf.bb_embedder)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        self.trunk[f'lattice_output'] = layers.LatticeOutput(
            self._mpa_conf.c_s)
        for b in range(self._mpa_conf.num_blocks):
            self.trunk[f'mpa_{b}'] = layers.MOFPointAttention(self._mpa_conf)
            self.trunk[f'mpa_ln_{b}'] = nn.LayerNorm(self._mpa_conf.c_s)
            tfmr_in = self._mpa_conf.c_s
            tfmr_layer = layers.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._mpa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in)
            self.trunk[f'tfmr_{b}'] = layers.TransformerEncoder(
                encoder_layer=tfmr_layer,
                num_layers=self._mpa_conf.seq_tfmr_num_layers)
            self.trunk[f'post_tfmr_{b}'] = layers.Linear(
                tfmr_in, self._mpa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = layers.StructureModuleTransition(
                c=self._mpa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = layers.BackboneUpdate(
                self._mpa_conf.c_s, use_rot_updates=True)

            if b < self._mpa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = layers.EdgeTransition(
                    node_embed_size=self._mpa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def _lattice_to_nm_radians(self, lattice_t):
        lattice_t = lattice_t.clone()
        lattice_t[:, :3] *= du.ANG_TO_NM_SCALE
        lattice_t[:, 3:] *= torch.pi / 180.0
        return lattice_t

    def _lattice_to_ang_degrees(self, lattice_t):
        lattice_t = lattice_t.clone()
        lattice_t[:, :3] *= du.NM_TO_ANG_SCALE
        lattice_t[:, 3:] *= 180.0 / torch.pi
        return lattice_t
    
    def forward(self, input_feats, remove_mean=True):
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        lattice_t = input_feats['lattice_t']

        # Initialize building block embeddings
        bb_emb = self.bb_embedder(input_feats)

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            bb_emb
        )
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed, edge_index = self.edge_feature_net(
            input_feats['batch'],
            init_node_embed,
            trans_t,
            trans_sc
        )

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        curr_lattice = self._lattice_to_nm_radians(lattice_t)
        node_embed = init_node_embed
        edge_embed = init_edge_embed
        for b in range(self._mpa_conf.num_blocks):
            mpa_embed = self.trunk[f'mpa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                curr_lattice,
                edge_index,
                input_feats['num_bbs'])
            node_embed = self.trunk[f'mpa_ln_{b}'](node_embed + mpa_embed)
            tfmr_out = self.trunk[f'tfmr_{b}'](node_embed, edge_index)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed)
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update)
            if b < self._mpa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, edge_index)

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        # Compute cell
        pred_lattice = self.trunk[f'lattice_output'](node_embed, input_feats['batch'])
        pred_lattice = self._lattice_to_ang_degrees(pred_lattice)

        if remove_mean:
            pred_trans_mean = scatter_mean(pred_trans, input_feats['batch'], dim=0)
            pred_trans_mean = pred_trans_mean.repeat_interleave(input_feats['num_bbs'], dim=0)
            pred_trans = pred_trans - pred_trans_mean
        
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_lattice': pred_lattice,
        }