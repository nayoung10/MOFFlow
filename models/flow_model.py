
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.bb_embedder import BuildingBlockEmbedder
from models import layers
from data import utils as du


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.bb_embedder = BuildingBlockEmbedder(model_conf.bb_embedder)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        self.trunk[f'lattice_output'] = layers.LatticeOutput(
            self._ipa_conf.c_s)
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = layers.MOFPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=True
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = layers.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = layers.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = layers.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = layers.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
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
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
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
            node_mask,
            diffuse_mask,
            bb_emb
        )
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        curr_lattice = self._lattice_to_nm_radians(lattice_t)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                curr_lattice,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        # Compute cell
        pred_lattice = self.trunk[f'lattice_output'](
            node_embed * node_mask[..., None])
        pred_lattice = self._lattice_to_ang_degrees(pred_lattice)

        if remove_mean:
            pred_trans = pred_trans - pred_trans.mean(dim=1, keepdim=True)
        
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_lattice': pred_lattice,
        }