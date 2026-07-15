import tempfile, os, sys, types

# Minimal mock for torch
torch_mod = types.ModuleType("torch")
torch_mod.nn = types.ModuleType("torch.nn")
torch_mod.nn.Module = object
torch_mod.save = lambda state, path: None
sys.modules["torch"] = torch_mod
sys.modules["torch.nn"] = torch_mod.nn

# Read just the save_checkpoint function
code = ""
with open("train.py") as f:
    in_func = False
    for line in f:
        if "def save_checkpoint" in line:
            in_func = True
        if in_func:
            code += line
            if line.strip() == "" and "def save_checkpoint" not in code[:50]:
                break

exec(code)
print("Function loaded, testing...")

from pathlib import Path
log_dir = Path("/tmp")

class Mock:
    def state_dict(self): return {}
Mock.param_groups = [Mock()]
Mock.__init__ = lambda self: None

# Test step save
save_checkpoint(Mock(), Mock(), Mock(), Mock(), Mock(),
                1, 200, 0.5, [], log_dir / "step_000200.pt", step_loss=0.6)
print("step save: OK")

# Test last_model
save_checkpoint(Mock(), Mock(), Mock(), Mock(), Mock(),
                1, 1000, 0.5, [], log_dir / "last_model.pt")
print("last_model: OK")

# Test best_model
save_checkpoint(Mock(), Mock(), Mock(), Mock(), Mock(),
                1, 1000, 0.4, [], log_dir / "best_model.pt")
print("best_model: OK")

# Test epoch
save_checkpoint(Mock(), Mock(), Mock(), Mock(), Mock(),
                1, 1000, 0.5, [], log_dir / "epoch_001.pt")
print("epoch_ckpt: OK")
print("\nAll 4 save patterns verified!")
