import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"

# Check GPU0 owner
s, o, e = c.exec_command("ps -p 3028791 -o pid,user,comm 2>/dev/null; echo '---'; whoami", timeout=10)
print("GPU0 owner:", o.read().decode().strip())

# Check which GPUs are available
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader", timeout=10)
print("All GPUs:\n" + o.read().decode().strip())

# Restart GPU2 (only one we can use now)
ts = time.strftime("%Y%m%d_%H%M%S")
cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 200 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 200 --log_interval 10 "
    f"> outputs/report_gen/train_gpu2_{ts}.log 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
print(f"GPU2 restarted: train_gpu2_{ts}.log")

time.sleep(150)

s, o, e = c.exec_command(
    f"tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_gpu2_{ts}.log; "
    f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^2,'",
    timeout=10)
print(o.read().decode())

c.close()
