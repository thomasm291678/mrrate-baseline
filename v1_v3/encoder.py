import torch, torch.nn as nn
from densenet3d import create_densenet3d


class TransformerBlock(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class HybridEncoder(nn.Module):
    def __init__(self, backbone="densenet3d_201", in_channels=1,
                 vit_dim=512, vit_heads=8, vit_depth=2, grid=2,
                 out_dim=2048, dropout=0.1, freeze_cnn=False):
        super().__init__()
        self.densenet = create_densenet3d(backbone, in_channels=in_channels)
        self.channels = self.densenet.stage_channels
        self.grid = grid
        self.n_tokens = 4 * (grid ** 3)
        self.vit_dim = vit_dim

        self.pool = nn.AdaptiveAvgPool3d((grid, grid, grid))
        self.stage_projs = nn.ModuleList([
            nn.Conv3d(ch, vit_dim, kernel_size=1) for ch in self.channels])

        self.scale_emb = nn.Parameter(torch.randn(4, 1, vit_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_tokens, vit_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(vit_dim, vit_heads, dropout=dropout)
            for _ in range(vit_depth)])
        self.norm = nn.LayerNorm(vit_dim)
        self.head = nn.Linear(vit_dim, out_dim)

        if freeze_cnn:
            for p in self.densenet.parameters():
                p.requires_grad = False

    def forward(self, x):
        stage_out = self.densenet(x, return_stages=(0, 1, 2, 3))
        tokens_list = []
        for i in range(4):
            feats = stage_out[i]
            pooled = self.pool(feats)
            proj = self.stage_projs[i](pooled)
            B = proj.shape[0]
            tok = proj.flatten(2).permute(0, 2, 1) + self.scale_emb[i]
            tokens_list.append(tok)

        tokens = torch.cat(tokens_list, dim=1) + self.pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens))


class ReportingModel(nn.Module):
    def __init__(self, llm_dim=2048, grid=2, vit_dim=512,
                 vit_heads=8, vit_depth=2, freeze_cnn=False, use_compile=False):
        super().__init__()
        self.grid = grid
        self.n_tokens_per_mod = 4 * (grid ** 3)
        self.total_tokens = 3 * self.n_tokens_per_mod
        self.llm_dim = llm_dim

        def make_enc():
            enc = HybridEncoder(grid=grid, vit_dim=vit_dim,
                                vit_heads=vit_heads, vit_depth=vit_depth,
                                out_dim=llm_dim, freeze_cnn=freeze_cnn)
            if use_compile:
                enc = torch.compile(enc, mode="reduce-overhead")
            return enc

        self.t1_enc = make_enc()
        self.t2_enc = make_enc()
        self.flair_enc = make_enc()
        self.mod_emb = nn.Parameter(torch.randn(3, 1, llm_dim) * 0.02)

    def _encode(self, enc, vol, present, mod_id, B, dev):
        if not present.any():
            return torch.zeros(B, self.n_tokens_per_mod, self.llm_dim, device=dev)
        idx = torch.where(present)[0]
        tok = enc(vol[idx])
        out = torch.zeros(B, self.n_tokens_per_mod, tok.shape[-1], device=dev)
        out[idx] = tok + self.mod_emb[mod_id]
        return out

    def forward(self, t1, flair, t2, has_t1, has_flair, has_t2):
        B, dev = t1.shape[0], t1.device
        v1 = self._encode(self.t1_enc, t1, has_t1, 0, B, dev)
        v2 = self._encode(self.t2_enc, t2, has_t2, 1, B, dev)
        v3 = self._encode(self.flair_enc, flair, has_flair, 2, B, dev)
        return torch.cat([v1, v2, v3], dim=1)

    def load_densenet_weights(self, v1_state_dict):
        for enc, prefix in [(self.t1_enc, "t1_enc."),
                            (self.t2_enc, "t2_enc."),
                            (self.flair_enc, "flair_enc.")]:
            dst = enc.densenet.state_dict()
            mapped = {}
            for k, v in v1_state_dict.items():
                if k.startswith(prefix):
                    rest = k[len(prefix):].replace("backbone.", "", 1)
                    if rest in dst:
                        mapped[rest] = v
            if mapped:
                dst.update(mapped)
                enc.densenet.load_state_dict(dst, strict=False)


class SingleModalEncoderV1(nn.Module):
    def __init__(self, backbone_name="densenet3d_201", in_channels=1,
                 feature_dim=2048, grid=4):
        super().__init__()
        self.backbone = create_densenet3d(backbone_name, in_channels=in_channels)
        final_ch = self.backbone.stage_channels[-1]
        self.pool = nn.AdaptiveAvgPool3d((grid, grid, grid))
        self.projection = nn.Sequential(
            nn.Linear(final_ch * grid * grid * grid, feature_dim),
            nn.LayerNorm(feature_dim))

    def forward(self, x):
        sf = self.backbone(x, return_stages=(-1,))
        feats = sf[max(sf.keys())]
        pooled = self.pool(feats)
        return self.projection(pooled.flatten(1))


class SingleModalMAE(nn.Module):
    def __init__(self, backbone_name="densenet3d_201", in_channels=1,
                 feature_dim=2048, grid=4, mask_ratio=0.75):
        super().__init__()
        self.encoder = SingleModalEncoderV1(backbone_name, in_channels,
                                            feature_dim, grid)
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
            nn.Conv3d(32, in_channels, 3, 1, 1))

    @staticmethod
    def mask_input(x, mask_ratio=0.75, patch_size=8):
        B, C, D, H, W = x.shape
        mask = torch.zeros(B, 1, D, H, W, device=x.device)
        pd, ph, pw = D // patch_size, H // patch_size, W // patch_size
        n_patches = pd * ph * pw
        n_mask = int(n_patches * mask_ratio)
        for b in range(B):
            idx = torch.randperm(n_patches, device=x.device)[:n_mask]
            for j in idx:
                dz, dy, dx = j // (ph * pw), (j // pw) % ph, j % pw
                d0, d1 = dz * patch_size, min((dz + 1) * patch_size, D)
                h0, h1 = dy * patch_size, min((dy + 1) * patch_size, H)
                w0, w1 = dx * patch_size, min((dx + 1) * patch_size, W)
                mask[b, :, d0:d1, h0:h1, w0:w1] = 1.0
        return x * (1 - mask), mask

    def forward(self, x):
        masked, mask = self.mask_input(x, self.mask_ratio)
        features = self.encoder(masked)
        x_dec = self.decoder_proj(features).view(x.shape[0], 256, 4, 4, 4)
        recon = self.decoder(x_dec)
        recon = nn.functional.interpolate(recon, size=x.shape[2:],
                                          mode='trilinear', align_corners=False)
        loss = nn.functional.mse_loss(recon * mask, x * mask, reduction='sum')
        return recon, mask, loss / (mask.sum() + 1e-8)
