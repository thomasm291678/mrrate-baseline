"""
Single-modality DenseNet encoder v2 — per-position token output.

Removes the flat → Linear bottleneck. Instead:
  DenseNet3D → AdaptiveAvgPool3d(G,G,G) → Conv3d 1×1 (per-position proj)
  → G³ independent spatial tokens, each 2048-dim.

Keeps full backward compatibility with existing pre-trained DenseNet weights.
"""
import torch
import torch.nn as nn

from .densenet3d import create_densenet3d


class SingleModalEncoder(nn.Module):
    """DenseNet3D encoder for ONE modality.

    v2: outputs (B, G³, feature_dim) spatial tokens instead of
    a single flattened vector.
    """

    def __init__(self, backbone_name="densenet3d_201", in_channels=1,
                 feature_dim=2048, grid=2):
        super().__init__()
        self.backbone = create_densenet3d(backbone_name, in_channels=in_channels)
        final_ch = self.backbone.stage_channels[-1]
        self.final_ch = final_ch
        self.grid = grid
        self.feature_dim = feature_dim

        self.pool = nn.AdaptiveAvgPool3d((grid, grid, grid))

        self.token_proj = nn.Conv3d(final_ch, feature_dim, kernel_size=1)

        self.token_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """x: (B, 1, D, H, W) → (B, G³, feature_dim)"""
        stage_out = self.backbone(x, return_stages=(-1,))
        last_key = max(stage_out.keys())
        feats = stage_out[last_key]

        pooled = self.pool(feats)

        tokens = self.token_proj(pooled)

        B, C, D, H, W = tokens.shape
        tokens = tokens.flatten(2).permute(0, 2, 1)
        tokens = self.token_norm(tokens)

        return tokens


class SingleModalEncoderV1(nn.Module):
    """Original v1 encoder — kept for loading old checkpoints.

    DenseNet3D → AdaptiveAvgPool3d(4,4,4) → flatten → Linear(141312, 2048)
    → single vector output.
    """

    def __init__(self, backbone_name="densenet3d_201", in_channels=1,
                 feature_dim=2048, grid=4):
        super().__init__()
        self.backbone = create_densenet3d(backbone_name, in_channels=in_channels)
        final_ch = self.backbone.stage_channels[-1]
        self.final_ch = final_ch
        self.pool = nn.AdaptiveAvgPool3d((grid, grid, grid))
        self.projection = nn.Sequential(
            nn.Linear(final_ch * grid * grid * grid, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x):
        stage_out = self.backbone(x, return_stages=(-1,))
        last_key = max(stage_out.keys())
        feats = stage_out[last_key]
        pooled = self.pool(feats)
        flat = pooled.flatten(1)
        return self.projection(flat)


class SingleModalMAE(nn.Module):
    """Masked Autoencoder for DenseNet3D pre-training.

    Encoder: DenseNet3D → projection (v1 single-vector output for MAE task)
    Decoder: lightweight 3D transposed conv + final conv
    """

    def __init__(self, backbone_name="densenet3d_201", in_channels=1,
                 feature_dim=2048, grid=4, mask_ratio=0.75):
        super().__init__()
        self.encoder = SingleModalEncoderV1(
            backbone_name=backbone_name,
            in_channels=in_channels,
            feature_dim=feature_dim,
            grid=grid,
        )
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio
        decoder_dim = 256
        self.decoder_proj = nn.Linear(feature_dim, decoder_dim * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(decoder_dim, 128, 4, 2, 1),
            nn.BatchNorm3d(128), nn.GELU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64), nn.GELU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, in_channels, 3, 1, 1),
        )

    def mask_input(self, x):
        B, C, D, H, W = x.shape
        mask = torch.zeros(B, 1, D, H, W, device=x.device)
        patch_size = 8
        pd, ph, pw = D // patch_size, H // patch_size, W // patch_size
        n_patches = pd * ph * pw
        n_mask = int(n_patches * self.mask_ratio)
        for b in range(B):
            idx = torch.randperm(n_patches, device=x.device)[:n_mask]
            for i in idx:
                dz = i // (ph * pw)
                dy = (i // pw) % ph
                dx = i % pw
                d0, d1 = dz * patch_size, min((dz + 1) * patch_size, D)
                h0, h1 = dy * patch_size, min((dy + 1) * patch_size, H)
                w0, w1 = dx * patch_size, min((dx + 1) * patch_size, W)
                mask[b, :, d0:d1, h0:h1, w0:w1] = 1.0
        masked_input = x * (1 - mask)
        return masked_input, mask

    def forward(self, x):
        masked_input, mask = self.mask_input(x)
        features = self.encoder(masked_input)
        x_dec = self.decoder_proj(features)
        x_dec = x_dec.view(x.shape[0], 256, 4, 4, 4)
        recon = self.decoder(x_dec)
        recon = F.interpolate(recon, size=x.shape[2:], mode='trilinear', align_corners=False)
        loss = F.mse_loss(recon * mask, x * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-8)
        return recon, mask, loss
