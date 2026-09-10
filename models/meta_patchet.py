import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from mamba_ssm import Mamba


class BiMambaBlock(nn.Module):
    """Bidirectional residual Mamba block for intra-patch encoding."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_norm = self.norm(x)
        out_fwd = self.mamba_fwd(x_norm)
        out_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        return x + out_fwd + out_bwd


class SequenceMambaBackbone(nn.Module):
    """Apply bidirectional Mamba layers to local patch sequences."""

    def __init__(self, c_in, n_layers=3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=c_in, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class Simple_mLSTM_Cell(nn.Module):
    """Minimal matrix-memory LSTM cell compatible with higher-order gradients."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model * 6)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        projs = self.proj(x)
        q, k, v, i_gate, f_gate, o_gate = torch.chunk(projs, 6, dim=-1)

        q = q / (dim ** 0.5)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)

        memory = torch.zeros(batch_size, dim, dim, device=x.device)
        h_out = []

        for t in range(seq_len):
            k_t = k[:, t, :].unsqueeze(-1)
            v_t = v[:, t, :].unsqueeze(1)
            i_t = i_gate[:, t, :].unsqueeze(-1)
            f_t = f_gate[:, t, :].unsqueeze(-1)

            memory = f_t * memory + i_t * torch.bmm(k_t, v_t)
            q_t = q[:, t, :].unsqueeze(-1)
            h_t = o_gate[:, t, :] * torch.bmm(memory, q_t).squeeze(-1)
            h_out.append(h_t)

        h_out = torch.stack(h_out, dim=1)
        return self.layer_norm(self.out_proj(h_out))


class Bi_mLSTM_Block(nn.Module):
    """Bidirectional matrix-memory LSTM block."""

    def __init__(self, d_model):
        super().__init__()
        self.forward_cell = Simple_mLSTM_Cell(d_model)
        self.backward_cell = Simple_mLSTM_Cell(d_model)
        self.merge = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        fwd_out = self.forward_cell(x)
        bwd_out = self.backward_cell(torch.flip(x, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])
        return self.merge(torch.cat([fwd_out, bwd_out], dim=-1))


class InterPatchMultiViewFusion(nn.Module):
    """Fuse mLSTM and Transformer views across patches."""

    def __init__(self, embed_dim, xlstm_layers=2, n_heads=4):
        super().__init__()
        print(f"Initializing inter-patch fusion (mLSTM + Transformer, dim={embed_dim})")

        self.xlstm_branch = nn.ModuleList([
            Bi_mLSTM_Block(d_model=embed_dim)
            for _ in range(xlstm_layers)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )
        self.transformer_branch = nn.TransformerEncoder(encoder_layer, num_layers=xlstm_layers)

        self.fusion_weights = nn.Parameter(torch.tensor([0.90, 1.10]))
        self.temperature = 0.2
        self.ln_xlstm = nn.LayerNorm(embed_dim)
        self.ln_tf = nn.LayerNorm(embed_dim)

    def forward(self, x):
        xlstm_out = x
        for layer in self.xlstm_branch:
            xlstm_out = layer(xlstm_out)

        tf_out = self.transformer_branch(x)

        xlstm_out = self.ln_xlstm(xlstm_out)
        tf_out = self.ln_tf(tf_out)
        weights = torch.softmax(self.fusion_weights / self.temperature, dim=0)
        return (weights[0] * xlstm_out) + (weights[1] * tf_out)


class MetaPatchET(nn.Module):
    """Meta-learning model for protein optimum-temperature prediction."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        print(f"Loading ESM Backbone: {config['pretrain_model']}")
        self.pretrain_model = EsmModel.from_pretrained(config['pretrain_model'])
        self.pretrain_model.gradient_checkpointing_enable()
        print("ESM gradient checkpointing enabled")
        self.esm_dim = config.get('esm_dim', self.pretrain_model.config.hidden_size)
        print(f"ESM Dim: {self.esm_dim}")

        print("Initializing intra-patch Mamba backbone")
        self.patch_intra_layers = SequenceMambaBackbone(
            c_in=self.esm_dim,
            n_layers=config.get('n_mamba_layers', 3),
            d_state=config.get('mamba_d_state', 16),
            d_conv=config.get('mamba_d_conv', 4),
            expand=config.get('mamba_expand', 2)
        )

        self.esm_projection = nn.Linear(self.esm_dim, config['target_window'])

        self.inter_multiview_fusion = InterPatchMultiViewFusion(
            embed_dim=config['target_window'],
            xlstm_layers=config.get('inter_xlstm_layers', 2),
            n_heads=config.get('inter_tf_heads', 4)
        )
        self.inter_fusion_norm = nn.LayerNorm(config.get('target_window', 128))

        self.patch_inter_heads = config['n_patch_inter_heads']
        self.patch_inter_kernel = config['patch_inter_kernel']
        self.patch_inter_conv = nn.Conv1d(
            config['target_window'],
            config['target_window'],
            kernel_size=2 * self.patch_inter_kernel + 1,
            padding=self.patch_inter_kernel
        )

        self.patch_inter_layers = nn.ModuleList([
            nn.Conv1d(
                config['target_window'],
                config['target_window'],
                kernel_size=2 * self.patch_inter_kernel + 1,
                padding=self.patch_inter_kernel
            ) for _ in range(self.patch_inter_heads)
        ])

        self.final_feature_dim = config['target_window'] * 2 * self.patch_inter_heads

        mid_dim_1 = min(self.final_feature_dim, max(128, self.final_feature_dim // 4))
        mid_dim_2 = min(mid_dim_1, max(32, self.final_feature_dim // 16))

        self.bottleneck_proj = nn.Linear(self.final_feature_dim, mid_dim_2)
        self.bottleneck_path = nn.Sequential(
            nn.Linear(self.final_feature_dim, mid_dim_1),
            nn.LayerNorm(mid_dim_1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid_dim_1, mid_dim_2),
            nn.GELU()
        )
        self.pred_head = nn.Linear(mid_dim_2, 1, bias=True)

        nn.init.xavier_normal_(self.pred_head.weight, gain=0.5)
        nn.init.constant_(self.pred_head.bias, 0)

    def build_meta_sgd_lrs(self, base_inner_lr=0.005):
        """Create learnable inner-loop rates for the prediction head."""
        self.inner_lrs = nn.ParameterDict()
        current_params = list(self.named_parameters())

        for name, param in current_params:
            if not param.requires_grad:
                continue

            if 'pred_head' in name:
                safe_name = name.replace('.', '_')
                self.inner_lrs[safe_name] = nn.Parameter(
                    torch.tensor(base_inner_lr, dtype=torch.float32)
                )

        print(f"Created {len(self.inner_lrs)} learnable Meta-SGD inner-loop rates")

    def extract_features(self, input_ids, attention_mask):
        """Extract fused local and inter-patch sequence features."""
        outputs = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        x = hidden_state.transpose(1, 2)

        _, _, current_len = x.shape
        target_len = self.config['context_window']
        patch_len = self.config['patch_len']
        # Preserve the original policy of retaining complete patches only.
        num_patches_real = current_len // patch_len

        if current_len < target_len:
            pad_len = target_len - current_len
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x[:, :, :target_len]

        batch_size, dim, seq_len = x_padded.shape
        num_patches = seq_len // patch_len
        # Fold the patch axis into the batch axis for independent local scans.
        x_patches = (
            x_padded.view(batch_size, dim, num_patches, patch_len)
            .transpose(1, 2)
            .reshape(batch_size * num_patches, dim, patch_len)
        )

        intra_out = self.patch_intra_layers(x_patches)
        intra_pooled = intra_out.mean(dim=-1)
        backbone_output = intra_pooled.view(batch_size, num_patches, dim)
        global_features = self.esm_projection(backbone_output)
        # Remove patches introduced only by context-window padding.
        hidden_state_combined = global_features[:, :num_patches_real, :]

        residual = hidden_state_combined
        fused_features = self.inter_multiview_fusion(hidden_state_combined)
        hidden_state_combined = self.inter_fusion_norm(residual + fused_features)

        x_inter_in = hidden_state_combined.transpose(1, 2)
        patch_inter_values = self.patch_inter_conv(x_inter_in)

        cat_xsum = []
        cat_xmax = []

        for i in range(self.patch_inter_heads):
            weights = F.softmax(self.patch_inter_layers[i](x_inter_in), dim=-1)
            x_sum = torch.sum(patch_inter_values * weights, dim=-1)
            x_max, _ = torch.max(patch_inter_values, dim=-1)

            cat_xsum.append(x_sum)
            cat_xmax.append(x_max)

        cat_xsum = torch.cat(cat_xsum, dim=1)
        cat_xmax = torch.cat(cat_xmax, dim=1)
        return torch.cat([cat_xsum, cat_xmax], dim=1)

    def predict_with_params(self, features, params_dict):
        """Predict with an explicit parameter dictionary for inner-loop updates."""
        x = F.layer_norm(features, [features.size(-1)])
        x = self.bottleneck_path(x) + self.bottleneck_proj(x)
        w_final = params_dict['pred_head.weight']
        b_final = params_dict.get('pred_head.bias', None)
        return F.linear(x, w_final, b_final).squeeze(-1)

    def predict_from_features(self, features):
        """Predict from cached features with the model's current parameters."""
        x = F.layer_norm(features, [features.size(-1)])
        x = self.bottleneck_path(x) + self.bottleneck_proj(x)
        return self.pred_head(x).squeeze(-1)

    def forward(self, input_ids, attention_mask):
        """Run end-to-end temperature prediction."""
        features = self.extract_features(input_ids, attention_mask)
        return self.predict_from_features(features)
