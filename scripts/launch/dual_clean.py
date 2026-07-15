import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"

# Kill ALL zombie processes on GPU0 and GPU2
c.exec_command("pkill -9 -f 'train.py' 2>/dev/null; true", timeout=5)
time.sleep(5)
c.exec_command("fuser -k /dev/nvidia0 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia2 2>/dev/null; true", timeout=5)
time.sleep(10)

# Verify clean
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^0,|^2,'", timeout=10)
print("GPU0+2 after cleanup:", o.read().decode().strip())

# Launch GPU0 from zero
ts0 = time.strftime("%Y%m%d_%H%M%S")
cmd0 = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 "
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
    f"> outputs/report_gen/train_gpu0_{ts0}.log 2>&1 &"
)
c.exec_command(cmd0, timeout=5)

# Launch GPU2 from zero
ts2 = time.strftime("%Y%m%d_%H%M%S")
cmd2 = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 "
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
    f"> outputs/report_gen/train_gpu2_{ts2}.log 2>&1 &"
)
c.exec_command(cmd2, timeout=5)

print(f"GPU0 train_gpu0_{ts0}.log")
print(f"GPU2 train_gpu2_{ts2}.log")

time.sleep(160)

s, o, e = c.exec_command(
    f"echo GPU0; tail -3 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_gpu0_{ts0}.log; "
    f"echo GPU2; tail -3 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_gpu2_{ts2}.log",
    timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E '^0,|^2,'", timeout=10)
print("GPU0+2:", o.read().decode().strip())

c.close()
