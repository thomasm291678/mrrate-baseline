import torch
import torch.nn as nn

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        inter = bn_size * growth_rate
        self.n1 = nn.BatchNorm3d(in_channels)
        self.r1 = nn.ReLU(inplace=True)
        self.c1 = nn.Conv3d(in_channels, inter, 1, bias=False)
        self.n2 = nn.BatchNorm3d(inter)
        self.r2 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv3d(inter, growth_rate, 3, 1, 1, bias=False)
    def forward(self, x):
        o = self.c1(self.r1(self.n1(x)))
        return torch.cat([x, self.c2(self.r2(self.n2(o)))], 1)

class _DenseBlock(nn.Module):
    def __init__(self, n, ic, gr, bs=4):
        super().__init__()
        self.blk = nn.Sequential(*[_DenseLayer(ic + i * gr, gr, bs) for i in range(n)])
    def forward(self, x): return self.blk(x)

class _Transition(nn.Module):
    def __init__(self, ic, oc, stride=2):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm3d(ic), nn.ReLU(inplace=True),
            nn.Conv3d(ic, oc, 1, bias=False))
        self.pool = nn.AvgPool3d(stride, stride) if stride > 1 else nn.Identity()
    def forward(self, x): return self.pool(self.seq(x))

class DenseNet3D(nn.Module):
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, compression=0.5, init_channels=64,
                 spatial_size=(128, 128, 96)):
        super().__init__()
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, 7, 2, 3, bias=False),
            nn.BatchNorm3d(init_channels), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, 2, 1))
        nc = init_channels
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self._stage_channels = []
        for i, nl in enumerate(block_config):
            self.blocks.append(_DenseBlock(nl, nc, growth_rate, bn_size))
            nc += nl * growth_rate
            self._stage_channels.append(nc)
            if i < len(block_config) - 1:
                oc = int(nc * compression)
                self.transitions.append(_Transition(nc, oc, 2))
                nc = oc
        self.final_norm = nn.BatchNorm3d(nc)
        self._input_shape = spatial_size
        self._stage_shapes = self._compute_shapes()
    def _compute_shapes(self):
        d, h, w = self._input_shape[0] // 4, self._input_shape[1] // 4, self._input_shape[2] // 4
        sh = []
        for _ in range(len(self.block_config)):
            sh.append((d, h, w)); d = d // 2; h = h // 2; w = w // 2
        return sh
    @property
    def stage_channels(self): return self._stage_channels
    @property
    def stage_shapes(self): return self._stage_shapes

    def forward(self, x, return_stages=(-3, -2, -1)):
        x = self.stem(x)
        sf = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in return_stages or i - len(self.blocks) in return_stages:
                sf[i] = x
            if i < len(self.transitions):
                x = self.transitions[i](x)
        return sf

    def forward_features(self, x):
        sf = self.forward(x, return_stages=(-1,))
        feats = self.final_norm(sf[len(self.blocks) - 1])
        return nn.functional.adaptive_avg_pool3d(feats, (1, 1, 1)).flatten(1)

def _make_dn(name, config, **kw):
    return DenseNet3D(block_config=config, **kw)

DENSENET_FACTORY = {
    "densenet3d_121": lambda **kw: _make_dn("121", (6, 12, 24, 16), **kw),
    "densenet3d_169": lambda **kw: _make_dn("169", (6, 12, 32, 32), **kw),
    "densenet3d_201": lambda **kw: _make_dn("201", (6, 12, 48, 32), **kw),
}

def create_densenet3d(name, **kwargs):
    name = name.lower().replace("-", "_")
    if name not in DENSENET_FACTORY:
        raise ValueError(f"Unknown DenseNet3D: {name}. Available: {list(DENSENET_FACTORY.keys())}")
    return DENSENET_FACTORY[name](**kwargs)
