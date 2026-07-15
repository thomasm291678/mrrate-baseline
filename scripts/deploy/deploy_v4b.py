import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill running v4
c.exec_command("pkill -9 -f 'train_v4' 2>/dev/null; true", timeout=5)
time.sleep(3)

sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")
cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train_v4.py "
    f"--brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt "
    f"--projector attn --augment --auto_resume "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 1 --ga_steps 5 --epochs 5 --num_workers 2 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 100 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 200 --auto_save_interval 50 --log_interval 10 "
    f"> outputs/report_gen/train_v4_gpu3_{ts}.log 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
print(f"Started: train_v4_gpu3_{ts}.log")

time.sleep(240)

s, o, e = c.exec_command(
    f"tail -10 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_{ts}.log 2>/dev/null",
    timeout=10)
print("\n" + o.read().decode())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
    timeout=10)
print("GPU3:", o.read().decode().strip())

c.close()
