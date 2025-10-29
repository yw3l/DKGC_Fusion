
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

from compounde_model import KGEModel
from models import CustomBertModel

class DKGC(nn.Module):
    def __init__(self, simkgc_args, compounde_args, d_p=256):
        super(DKGC, self).__init__()

        # Load SimKGC
        self.simkgc = CustomBertModel(simkgc_args)
        self.simkgc.load_state_dict(torch.load(simkgc_args.checkpoint_path)['model_state_dict'])

        # Load CompoundE
        self.compounde = KGEModel(
            model_name=compounde_args['model'],
            nentity=compounde_args['nentity'],
            nrelation=compounde_args['nrelation'],
            hidden_dim=compounde_args['hidden_dim'],
            gamma=compounde_args['gamma'],
            double_entity_embedding=compounde_args['double_entity_embedding'],
            double_relation_embedding=compounde_args['double_relation_embedding'],
            triple_relation_embedding=compounde_args['triple_relation_embedding'],
            quad_relation_embedding=compounde_args['quad_relation_embedding'],
            evaluator=None
        )
        checkpoint = torch.load(os.path.join(compounde_args.init_checkpoint, 'checkpoint'))
        self.compounde.load_state_dict(checkpoint['model_state_dict'])


        # Freeze pre-trained models (optional, for phased training)
        for param in self.simkgc.parameters():
            param.requires_grad = False
        for param in self.compounde.parameters():
            param.requires_grad = False

        # Projection layers
        self.d_p = d_p
        compounde_entity_dim = compounde_args['hidden_dim'] * 2 if compounde_args['double_entity_embedding'] else compounde_args['hidden_dim']
        self.W_s_tail = nn.Linear(compounde_entity_dim, d_p)
        self.W_t = nn.Linear(self.simkgc.config.hidden_size, d_p)

        # Gating mechanism based on a simpler interpretation
        gate_input_dim = compounde_entity_dim * 2 + self.simkgc.config.hidden_size * 2
        self.gate_mlp_simple = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Nonlinear interaction
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_p, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, batch):
        # Unpack compounde data
        h_idx, r_idx, t_idx = batch['compounde_data'][:, 0], batch['compounde_data'][:, 1], batch['compounde_data'][:, 2]

        # SimKGC forward pass to get textual embeddings
        hr_vector = self.simkgc._encode(self.simkgc.hr_bert, **batch['simkgc_hr'])
        tail_vector = self.simkgc._encode(self.simkgc.tail_bert, **batch['simkgc_t'])
        head_text_vector = self.simkgc._encode(self.simkgc.tail_bert, **batch['simkgc_h'])

        # CompoundE forward pass to get structural embeddings
        head_struct = self.compounde.entity_embedding[h_idx]
        relation_struct = self.compounde.relation_embedding[r_idx]
        tail_struct = self.compounde.entity_embedding[t_idx]

        # Original scores
        s_text = torch.sum(hr_vector * tail_vector, dim=1)
        s_struct = self.compounde.CompoundE(head_struct.unsqueeze(1), relation_struct.unsqueeze(1), tail_struct.unsqueeze(1), mode='single').squeeze(1)

        # Gating-based adaptive fusion
        gate_input_simple = torch.cat([head_struct, tail_struct, hr_vector, tail_vector], dim=-1)
        alpha_simple = self.gate_mlp_simple(gate_input_simple)
        gated_score = alpha_simple.squeeze(-1) * s_text + (1 - alpha_simple.squeeze(-1)) * s_struct

        # Nonlinear interaction
        e_tail_struct_p = self.W_s_tail(tail_struct)
        e_tail_text_p = self.W_t(tail_vector)
        fusion_input = torch.cat([e_tail_struct_p, e_tail_text_p], dim=-1)
        fusion_score = self.fusion_mlp(fusion_input)

        # Final score
        final_score = gated_score + fusion_score.squeeze(-1)
        
        return final_score


