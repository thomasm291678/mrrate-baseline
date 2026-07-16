import torch, torch.nn as nn
import torch.nn.functional as F


class SE3D(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se_ratio=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SE3D(out_ch, se_ratio)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class StageAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MRRCNN(nn.Module):
    def __init__(self, in_chans=1, base_ch=32, grid=2):
        super().__init__()
        ch_list = [base_ch, base_ch * 2, base_ch * 4, base_ch * 6, base_ch * 8, base_ch * 10]
        blocks_per_stage = [2, 2, 3, 2, 2]
        self.grid = grid
        self.n_tokens_per_stage = grid ** 3
        n_stages = len(ch_list) - 1
        self.n_tokens = n_stages * self.n_tokens_per_stage
        self.cnn_dim = 512

        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, ch_list[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(ch_list[0]),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        for i in range(n_stages):
            stage = nn.Sequential()
            for j in range(blocks_per_stage[i]):
                stage.add_module(f"block{j}", ConvBlock(
                    ch_list[i] if j == 0 else ch_list[i + 1],
                    ch_list[i + 1],
                    stride=2 if j == 0 else 1,
                ))
            self.stages.append(stage)

        self.stage_attn = StageAttention(dim=self.cnn_dim, heads=4, dropout=0.1)

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d((grid, grid, grid)) for _ in range(n_stages)
        ])

        self.stage_projs = nn.ModuleList([
            nn.Conv3d(ch_list[i + 1], self.cnn_dim, 1) for i in range(n_stages)
        ])

        self.scale_emb = nn.Parameter(torch.randn(n_stages, 1, self.cnn_dim) * 0.02)

    def forward(self, x):
        x = self.stem(x)
        tokens_list = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            pooled = self.pools[i](x)
            proj = self.stage_projs[i](pooled)
            tok = proj.flatten(2).permute(0, 2, 1) + self.scale_emb[i]
            tokens_list.append(tok)
        cat_tokens = torch.cat(tokens_list, dim=1)
        return self.stage_attn(cat_tokens)


class ContrastiveHead(nn.Module):
    def __init__(self, cnn_dim=512, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cnn_dim, cnn_dim),
            nn.LayerNorm(cnn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cnn_dim, proj_dim),
        )

    def forward(self, tok):
        pooled = tok.mean(dim=1)
        return nn.functional.normalize(self.mlp(pooled), dim=1)




class ReportingModelV5(nn.Module):
    def __init__(self, llm_dim=2048, grid=2, base_ch=32, proj_dim=256):
        super().__init__()
        self.n_tokens_per_mod = 5 * (grid ** 3)
        self.total_tokens = 3 * self.n_tokens_per_mod
        self.llm_dim = llm_dim
        self.cnn_dim = 512

        self.t1_enc = MRRCNN(in_chans=1, base_ch=base_ch, grid=grid)
        self.t2_enc = MRRCNN(in_chans=1, base_ch=base_ch, grid=grid)
        self.flair_enc = MRRCNN(in_chans=1, base_ch=base_ch, grid=grid)

        self.contra_head = ContrastiveHead(cnn_dim=self.cnn_dim, proj_dim=proj_dim)

        self.t1_proj = nn.Linear(self.cnn_dim, llm_dim)
        self.t2_proj = nn.Linear(self.cnn_dim, llm_dim)
        self.flair_proj = nn.Linear(self.cnn_dim, llm_dim)

        self.mod_emb = nn.Parameter(torch.randn(3, 1, llm_dim) * 0.02)

        self.n_tokens_out = self.total_tokens

    def _encode(self, enc, proj, vol, present, mod_id, B, dev):
        if not present.any():
            return torch.zeros(B, self.n_tokens_per_mod, self.llm_dim, device=dev, dtype=vol.dtype)
        idx_mask = present.nonzero(as_tuple=True)[0]
        tok = enc(vol[idx_mask])
        tok = proj(tok)
        tok = tok + self.mod_emb[mod_id]
        out = torch.zeros(B, self.n_tokens_per_mod, self.llm_dim, device=dev, dtype=vol.dtype)
        out[idx_mask] = tok
        return out

    def encode_raw(self, mod, vol):
        enc_map = {"t1": self.t1_enc, "t2": self.t2_enc, "flair": self.flair_enc}
        return enc_map[mod](vol)

    def forward(self, t1, flair, t2, has_t1, has_flair, has_t2):
        B, dev = t1.shape[0], t1.device
        v1 = self._encode(self.t1_enc, self.t1_proj, t1, has_t1, 0, B, dev)
        v2 = self._encode(self.t2_enc, self.t2_proj, t2, has_t2, 1, B, dev)
        v3 = self._encode(self.flair_enc, self.flair_proj, flair, has_flair, 2, B, dev)
        vt = torch.cat([v1, v2, v3], dim=1)
        return vt




def apply_optimizations():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128, 128)
    enc = MRRCNN(in_chans=1, base_ch=32, grid=2)
    out = enc(x)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"MRRCNN: {tuple(x.shape)} -> {tuple(out.shape)}")
    print(f"  tokens: {enc.n_tokens}, params: {n_params:,}")

    has = torch.tensor([True, True])
    dummy = torch.randn(2, 1, 128, 128, 128)
    model = ReportingModelV5(llm_dim=2048, grid=2, base_ch=32)
    vt = model(dummy, dummy, dummy, has, has, has)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nReportingModelV5: {tuple(vt.shape)}")
    print(f"  total params: {total:,}")
    print(f"  tokens per mod: {model.n_tokens_per_mod}")
    print(f"  total tokens: {model.total_tokens}")
