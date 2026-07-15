import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272")

BASE = "/home/jiaqigu/mrrate_hidnet"
DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"

# Upload clean train_v5.py from local, with our mods
with open(r"C:\Users\HP\Documents\5555\train_v5_phase1.py") as f:
    code = f.read()

# Fix import path (parent.parent → parent for farm02)
code = code.replace(
    "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
    "sys.path.insert(0, str(Path(__file__).resolve().parent))"
)

# Comment out torch.compile (lines 138-143)
code = code.replace(
    """    # torch.compile the encoder with reduce-overhead (CUDA graph replay for frozen parts)
    try:
        enc = torch.compile(enc, mode="reduce-overhead", dynamic=False)
        log("Encoder compiled (reduce-overhead mode)")
    except Exception as e:
        log(f"torch.compile skipped: {e}")""",
    """    # torch.compile skipped (2080 Ti compatibility)""")

# Add --init_ckpt, --gpu, --output_name args
code = code.replace(
    '    p.add_argument("--batch_id", type=str, default=None)\n    args = p.parse_args()',
    '    p.add_argument("--batch_id", type=str, default=None)\n    p.add_argument("--init_ckpt", type=str, default=None)\n    p.add_argument("--gpu", type=int, default=None)\n    p.add_argument("--output_name", type=str, default="latest")\n    args = p.parse_args()')

# Add gpu device set and init_ckpt loading in train()
code = code.replace(
    'def train(args):\n    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'def train(args):\n    if args.gpu is not None:\n        torch.cuda.set_device(args.gpu)\n    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")')

code = code.replace(
    '    enc = ReportingModelV5(llm_dim=args.llm_dim, grid=G, base_ch=args.base_ch).to(dev)\n    for key in ["t1_proj", "t2_proj", "flair_proj"]:',
    '    enc = ReportingModelV5(llm_dim=args.llm_dim, grid=G, base_ch=args.base_ch).to(dev)\n    if args.init_ckpt:\n        log(f"Loading init weights from {args.init_ckpt}")\n        ckpt = torch.load(args.init_ckpt, map_location=dev, weights_only=False)\n        enc.load_state_dict(ckpt["encoder_state"], strict=True)\n        log("Init weights loaded")\n    for key in ["t1_proj", "t2_proj", "flair_proj"]:')

# Fix auto_resume paths to use output_name
code = code.replace(
    'auto_path = log_dir / "phase1_latest.pt"',
    'auto_path = log_dir / f"phase1_{args.output_name}_latest.pt"')
code = code.replace(
    'log_dir / "phase1_latest.pt")',
    'log_dir / f"phase1_{args.output_name}_latest.pt")')
# But the second one might be in save_checkpoint... let me check - no the log_dir/"phase1_latest.pt" pattern already handled

# Fix save paths
code = code.replace(
    'log_dir / f"phase1_step{global_step}.pt"',
    'log_dir / f"phase1_{args.output_name}_step{global_step}.pt"')

sftp = c.open_sftp()
with sftp.open(f"{BASE}/train_v5.py", "w") as f:
    f.write(code)
sftp.close()

# Verify syntax
_, o, _ = c.exec_command(f"cd {BASE} && python3 -c 'import py_compile; py_compile.compile(\"train_v5.py\", doraise=True); print(\"Syntax OK\")' 2>&1")
print("Syntax:", o.read().decode().strip())

# Kill and restart
c.exec_command("pkill -9 -f train_v5 2>/dev/null; for s in gpu0 gpu1 gpu2 gpu3; do tmux kill-session -t $s 2>/dev/null; done; sleep 1")

for g, batch_ids in [(0,"batch00,batch01,batch02,batch03,batch04"),(1,"batch05,batch06,batch07,batch08,batch09"),(2,"batch10,batch11,batch12,batch13,batch14"),(3,"batch15,batch16,batch17,batch18,batch19")]:
    cmd = (
        f"cd {BASE} && python3 train_v5.py "
        f"--init_ckpt outputs/report_gen/phase1_init.pt "
        f"--gpu {g} --log_dir outputs/report_gen/phase1_gpu{g} "
        f"--data_root {DATA} --batch_id {batch_ids} "
        f"--output_name gpu{g} --epochs 5 --batch_size 8 --lr 3e-4 "
        f"--auto_resume --augment"
    )
    c.exec_command(f'tmux kill-session -t gpu{g} 2>/dev/null; tmux new-session -d -s gpu{g} "{cmd}"')
    print(f"  GPU {g} launched")

time.sleep(45)
for g in range(4):
    _, o, _ = c.exec_command(f"tmux capture-pane -t gpu{g} -p | grep -E 'Train|loss=' | tail -1")
    out = o.read().decode().strip()
    if out:
        print(f"  GPU {g}: {out[:120]}")

_, o, _ = c.exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -4")
print(f"  {o.read().decode().replace(chr(10), ', ').strip()}")

c.close()
