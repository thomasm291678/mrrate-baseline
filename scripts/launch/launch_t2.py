import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check GPU7 free memory
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader | grep '^7,'",
    timeout=10)
print("GPU7:", o.read().decode().strip())

# Check who's on GPU7
s, o, e = c.exec_command(
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep $(fuser /dev/nvidia7 2>/dev/null | tr ' ' '\n' | head -1) 2>/dev/null; "
    "fuser /dev/nvidia7 2>/dev/null",
    timeout=10)
print("GPU7 user:", o.read().decode().strip())

# Kill any old train_v4 on GPU7 first (if any)
c.exec_command("pkill -9 -f 'train_v4.*t2' 2>/dev/null; true", timeout=5)

# Run T2 on GPU7 (batch_size=1 to be safe on limited memory)
PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")
LOG = f"outputs/report_gen/train_v4_t2_{ts}.log"

cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train_v4.py "
    f"--modality t2 --augment "
    f"--brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt "
    f"--projector attn "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 1 --ga_steps 8 --epochs 3 --num_workers 2 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 100 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 200 --auto_save_interval 100 --log_interval 10 "
    f">> {LOG} 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
print(f"T2 launched on GPU7: {LOG}")

time.sleep(180)

s, o, e = c.exec_command(f"tail -10 /home/jiaqigu/mrrate_hidnet/{LOG} 2>/dev/null", timeout=10)
print("\n" + o.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^7,'", timeout=10)
print("GPU7:", o.read().decode().strip())

s, o, e = c.exec_command("pgrep -f 'train_v4' | wc -l", timeout=10)
print(f"Total train_v4 processes: {o.read().decode().strip()}")

c.close()
