import tempfile, os, sys

# Add parent to path so we can import train module
sys.path.insert(0, os.getcwd())

# Mock torch and all heavy imports
class Fake:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __getattr__(self, name):
        if name == 'state_dict': return lambda: {}
        return lambda *a, **kw: Fake()

torch = Fake(device=lambda x: Fake(), amp=Fake(GradScaler=lambda x: None))
torch.nn = Fake(Module=Fake, BatchNorm3d=Fake, GELU=Fake)
torch.nn.MultiheadAttention = lambda *a, **kw: Fake(forward=lambda self,x,y,z: (x, None))
torch.nn.functional = Fake(mse_loss=lambda a,b,reduction: Fake())
torch.optim = Fake(AdamW=lambda **kw: Fake(state_dict=lambda:{}))
torch.cuda = Fake(is_available=lambda: True, memory_allocated=lambda d: 0)
torch.compile = lambda x, **kw: x
torch.amp = Fake(GradScaler=lambda x: Fake(state_dict=lambda:{}))
torch.save = lambda state, path: print(f"  SAVE OK: path={path}")
torch.load = lambda *a, **kw: {}
torch.utils.data = Fake(DataLoader=lambda *a, **kw: Fake())
torch.set_float32_matmul_precision = lambda x: None
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.cuda'] = torch.cuda
sys.modules['torch.utils'] = Fake(data=Fake(DataLoader=lambda *a, **kw: Fake()))

# Verify save_checkpoint signature matches all call sites
from inspect import signature, getmodule

# Read the function directly
exec(open("train.py").read().split("def train(")[0] + "\n")

# Test each call pattern
from pathlib import Path
log_dir = Path("/tmp/test")

# Pattern 1: step save (line 354-358)
try:
    save_checkpoint(Fake(), Fake(), Fake(), Fake(), Fake(),
                    1, 200, 0.5, [], log_dir / "step_000200.pt", step_loss=0.6)
    print("Pattern 1 (step save): OK")
except Exception as e:
    print(f"Pattern 1 FAIL: {e}")

# Pattern 2: last_model save (line 396-398)
try:
    save_checkpoint(Fake(), Fake(), Fake(), Fake(), Fake(),
                    1, 1000, 0.5, [], log_dir / "last_model.pt")
    print("Pattern 2 (last_model): OK")
except Exception as e:
    print(f"Pattern 2 FAIL: {e}")

# Pattern 3: best_model save (line 403-405)
try:
    save_checkpoint(Fake(), Fake(), Fake(), Fake(), Fake(),
                    1, 1000, 0.4, [], log_dir / "best_model.pt")
    print("Pattern 3 (best_model): OK")
except Exception as e:
    print(f"Pattern 3 FAIL: {e}")

# Pattern 4: epoch save (line 409-411)
try:
    save_checkpoint(Fake(), Fake(), Fake(), Fake(), Fake(),
                    1, 1000, 0.5, [], log_dir / "epoch_001.pt")
    print("Pattern 4 (epoch_ckpt): OK")
except Exception as e:
    print(f"Pattern 4 FAIL: {e}")
