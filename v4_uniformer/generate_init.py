import torch
import argparse
from pathlib import Path
from encoder_v5 import ReportingModelV5


def main():
    p = argparse.ArgumentParser("Generate shared initial weights for FedAvg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--out", type=str, default="outputs/init_weights.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    enc = ReportingModelV5(
        llm_dim=args.llm_dim,
        grid=args.grid,
        base_ch=args.base_ch,
        proj_dim=args.proj_dim,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": enc.state_dict(), "seed": args.seed}, out_path)

    total = sum(p.numel() for p in enc.parameters())
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"Saved init weights to {out_path}")
    print(f"  Total params: {total:,}  Trainable: {trainable:,}")
    print(f"  Seed: {args.seed}  Grid: {args.grid}  Base_ch: {args.base_ch}")


if __name__ == "__main__":
    main()
