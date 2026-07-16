import json, os, torch
from safetensors.torch import load_file

QWEN = "/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct"
idx = json.load(open(os.path.join(QWEN, "model.safetensors.index.json")))
p = idx["weight_map"]["model.embed_tokens.weight"]
w = load_file(os.path.join(QWEN, p))["model.embed_tokens.weight"]
print(f"shape={w.shape} nan={w.isnan().any().item()} inf={w.isinf().any().item()} min={w.min().item():.4f} max={w.max().item():.4f}")
