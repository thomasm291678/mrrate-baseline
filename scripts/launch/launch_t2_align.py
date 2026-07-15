import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill only T2 (GPU7), leave T1 (GPU3) running
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4.*t2' | grep -v grep | awk '{print $1}'", timeout=10)
t2_pid = o.read().decode().strip()
if t2_pid:
    c.exec_command(f"kill -9 {t2_pid} 2>/dev/null; true", timeout=5)
    print(f"Killed T2 PID {t2_pid}")
time.sleep(5)

# Upload updated script
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")

# T2 on GPU7: align mode, no Qwen forward, 50k samples
cmd_t2 = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train_v4.py "
    f"--modality t2 --augment --train_mode align "
    f"--brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt "
    f"--projector attn "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 4 --ga_steps 2 --epochs 2 --num_workers 2 --max_samples 50000 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 100 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 500 --auto_save_interval 200 --log_interval 10 "
    f">> outputs/report_gen/train_v4_t2_align_{ts}.log 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd_t2}' &", timeout=5)
print(f"T2 (align mode): train_v4_t2_align_{ts}.log")
print(f"  no Qwen forward | 50k samples | batch=4 ga=2 (eff=8) | 2 epochs")

time.sleep(180)

s, o, e = c.exec_command(
    f"tail -10 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t2_align_{ts}.log 2>/dev/null",
    timeout=10)
print("\n" + o.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^7,'", timeout=10)
print("GPU7:", o.read().decode().strip())

c.close()
