import torch, torch.nn as nn
from uniformer_blocks import UniFormer, uniformer_small


def _uniformer_small_1ch(**kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        in_chans=1,
        norm_layer=nn.LayerNorm, **kwargs)
    return model


class StageProjection(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.norm(x)


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


class MLPProjector(nn.Module):
    def __init__(self, vit_dim=512, grid=2, out_dim=2048):
        super().__init__()
        self.grid = grid
        n_tokens = 4 * (grid ** 3)
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 2),
            nn.GELU(),
            nn.Linear(vit_dim * 2, out_dim),
        )

    def forward(self, tokens):
        return self.proj(tokens)


class AttnProjector(nn.Module):
    def __init__(self, vit_dim=512, vit_heads=8, vit_depth=2, grid=2,
                 out_dim=2048, dropout=0.1):
        super().__init__()
        self.grid = grid
        n_tokens = 4 * (grid ** 3)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, vit_dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(vit_dim, vit_heads, dropout=dropout)
            for _ in range(vit_depth)])
        self.norm = nn.LayerNorm(vit_dim)
        self.head = nn.Linear(vit_dim, out_dim)

    def forward(self, tokens):
        tokens = tokens + self.pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens))


class UniFormerEncoder(nn.Module):
    def __init__(self, projector="attn", vit_dim=512, vit_heads=8, vit_depth=2,
                 grid=2, out_dim=2048, dropout=0.1):
        super().__init__()
        self.projector_type = projector
        self.uniformer = _uniformer_small_1ch()
        self.stage_channels = [64, 128, 320, 512]
        self.grid = grid
        self.n_tokens = 4 * (grid ** 3)
        self.vit_dim = vit_dim

        self.pool = nn.AdaptiveAvgPool3d((grid, grid, grid))
        self.stage_projs = nn.ModuleList([
            nn.Conv3d(ch, vit_dim, kernel_size=1) for ch in self.stage_channels])
        self.scale_emb = nn.Parameter(torch.randn(4, 1, vit_dim) * 0.02)

        if projector == "mlp":
            self.projector = MLPProjector(vit_dim, grid, out_dim)
        else:
            self.projector = AttnProjector(vit_dim, vit_heads, vit_depth,
                                           grid, out_dim, dropout)

    def forward(self, x):
        _, x1, x2, x3, x4 = self.uniformer.forward_features(x)

        tokens_list = []
        for i, feat in enumerate([x1, x2, x3, x4]):
            pooled = self.pool(feat)
            proj = self.stage_projs[i](pooled)
            tok = proj.flatten(2).permute(0, 2, 1) + self.scale_emb[i]
            tokens_list.append(tok)
        tokens = torch.cat(tokens_list, dim=1)

        if self.projector_type == "mlp":
            return self.projector(tokens)
        else:
            return self.projector(tokens)

    def load_brainmvp_weights(self, ckpt_path, dev="cpu"):
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        raw_state = ckpt.get("state_dict", ckpt)
        model_dict = self.uniformer.state_dict()
        mapped = {}
        for k, v in raw_state.items():
            key = k.replace("encoder.uniformer.", "").replace("module.", "")
            if key in model_dict:
                mapped[key] = v
        for ignore_key in ["patch_embed1.proj.weight", "patch_embed2.proj.weight",
                           "patch_embed3.proj.weight", "patch_embed4.proj.weight"]:
            mapped.pop(ignore_key, None)
        model_dict.update(mapped)
        self.uniformer.load_state_dict(model_dict, strict=False)
        loaded = sum(1 for k in mapped if k in model_dict)
        print(f"  BrainMVP UniFormer: loaded {loaded} / {len(model_dict)} params")


class ReportingModelV4(nn.Module):
    def __init__(self, projector="attn", llm_dim=2048, grid=2, vit_dim=512,
                 vit_heads=8, vit_depth=2, use_compile=False,
                 brainmvp_ckpt=""):
        super().__init__()
        self.grid = grid
        self.n_tokens_per_mod = 4 * (grid ** 3)
        self.total_tokens = 3 * self.n_tokens_per_mod
        self.llm_dim = llm_dim

        def make_enc():
            enc = UniFormerEncoder(projector=projector, grid=grid,
                                    vit_dim=vit_dim, vit_heads=vit_heads,
                                    vit_depth=vit_depth, out_dim=llm_dim)
            if brainmvp_ckpt:
                enc.load_brainmvp_weights(brainmvp_ckpt)
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


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128, 128)

    for proj in ["attn", "mlp"]:
        enc = UniFormerEncoder(projector=proj)
        out = enc(x)
        params_uni = sum(p.numel() for p in enc.uniformer.parameters())
        params_total = sum(p.numel() for p in enc.parameters())
        params_proj = params_total - params_uni
        print(f"[{proj}] Input: {tuple(x.shape)} -> {tuple(out.shape)}")
        print(f"     UniFormer: {params_uni:,}   Projector: {params_proj:,}   Total: {params_total:,}")
        print()

    has = torch.tensor([True, True])
    dummy = torch.zeros(2, 1, 128, 128, 128)
    model = ReportingModelV4(projector="attn")
    vt = model(dummy, dummy, dummy, has, has, has)
    print(f"ReportingModelV4(attn) output: {tuple(vt.shape)}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
