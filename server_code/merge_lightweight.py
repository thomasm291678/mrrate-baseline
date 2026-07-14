"""
Lightweight merge: use 3 pre-trained SingleModalEncoders as-is,
project each 2048-dim feature → visual tokens, concatenate.

Usage:
  python scripts/merge_lightweight.py
      --t1   outputs/pretrain_densenet/best_t1.pt
      --t2   outputs/pretrain_densenet/best_t2.pt
      --flair outputs/pretrain_densenet/best_flair.pt
      --output outputs/pretrain_densenet/multimodal_encoder.pt
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.single_modal_encoder import SingleModalEncoder


class LightweightMultiModal(nn.Module):
    """3 frozen pre-trained encoders + tiny projection heads.

    Each encoder outputs (B, 2048) → reshaped to (B, n_vt, llm_dim)
    → concatenated → (B, 3*n_vt, llm_dim).

    Encoders are frozen by default (no finetuning needed).
    Only the tiny projection heads (4 × 2048 → 2048) and modality
    embeddings are trainable.
    """

    def __init__(self, llm_dim=2048, n_vt=4, freeze_encoders=True):
        super().__init__()
        self.llm_dim = llm_dim
        self.n_vt = n_vt
        self.freeze_encoders = freeze_encoders

        # 3 independent pre-trained encoders (loaded from checkpoints)
        self.t1_enc = SingleModalEncoder()
        self.t2_enc = SingleModalEncoder()
        self.flair_enc = SingleModalEncoder()

        if freeze_encoders:
            for enc in [self.t1_enc, self.t2_enc, self.flair_enc]:
                for p in enc.parameters():
                    p.requires_grad = False

        # Tiny projection: (B, 2048) → (B, n_vt * 2048)
        self.t1_proj = nn.Linear(2048, n_vt * llm_dim)
        self.t2_proj = nn.Linear(2048, n_vt * llm_dim)
        self.flair_proj = nn.Linear(2048, n_vt * llm_dim)

        # Modality embeddings (learnable)
        self.mod_emb = nn.Parameter(torch.randn(3, n_vt, llm_dim) * 0.02)

    def _encode_one(self, encoder, proj, x, has_m, mod_idx, B, device):
        """Encode one modality. Missing → zero tokens."""
        if not has_m.any():
            return torch.zeros(B, self.n_vt, self.llm_dim, device=device)

        valid_idx = torch.where(has_m)[0]
        feat = encoder(x[valid_idx])  # (V, 2048)
        tokens = proj(feat)           # (V, n_vt * llm_dim)
        tokens = tokens.view(-1, self.n_vt, self.llm_dim)

        out = torch.zeros(B, self.n_vt, self.llm_dim, device=device)
        out[valid_idx] = tokens
        out = out + self.mod_emb[mod_idx].unsqueeze(0)
        return out

    def forward(self, t1, flair, t2, has_t1, has_flair, has_t2):
        B, device = t1.shape[0], t1.device
        v1 = self._encode_one(self.t1_enc, self.t1_proj, t1, has_t1, 0, B, device)
        v2 = self._encode_one(self.t2_enc, self.t2_proj, t2, has_t2, 1, B, device)
        v3 = self._encode_one(self.flair_enc, self.flair_proj, flair, has_flair, 2, B, device)
        return torch.cat([v1, v2, v3], dim=1)  # (B, 3*n_vt, llm_dim)


def merge_lightweight(t1_ckpt, t2_ckpt, flair_ckpt, output_path,
                      llm_dim=2048, n_vt=4, freeze=True):
    model = LightweightMultiModal(llm_dim=llm_dim, n_vt=n_vt, freeze_encoders=freeze)

    # Load encoder weights
    for ckpt_path, enc in [(t1_ckpt, model.t1_enc), (t2_ckpt, model.t2_enc),
                            (flair_ckpt, model.flair_enc)]:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc.load_state_dict(ckpt["encoder_state"])
        print(f"  Loaded {Path(ckpt_path).stem}: epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.6f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\nSaved: {output_path}")
    print(f"  Total params:     {total:,}")
    print(f"  Frozen (encoder): {frozen:,} (~{frozen*4/1e9:.2f} GB)")
    print(f"  Trainable (proj): {trainable:,} (~{trainable*4/1e6:.2f} MB)")
    print(f"  Output: ({3}×{n_vt}, {llm_dim}) = ({3*n_vt}, {llm_dim})")

    # Verify
    model.eval()
    B = 2
    dummy = torch.randn(B, 1, 128, 128, 128)
    has = torch.ones(B, dtype=torch.bool)
    with torch.no_grad():
        out = model(dummy, dummy, dummy, has, has, has)
    print(f"  Input: {list(dummy.shape)} ×3 → Output: {list(out.shape)} ✓")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--flair", required=True)
    parser.add_argument("--output", default="outputs/pretrain_densenet/multimodal_encoder.pt")
    parser.add_argument("--llm_dim", type=int, default=2048)
    parser.add_argument("--n_vt", type=int, default=4)
    parser.add_argument("--trainable", action="store_true",
                        help="Make encoders trainable (for fine-tuning)")
    args = parser.parse_args()

    merge_lightweight(args.t1, args.t2, args.flair, args.output,
                      llm_dim=args.llm_dim, n_vt=args.n_vt,
                      freeze=not args.trainable)


if __name__ == "__main__":
    main()
