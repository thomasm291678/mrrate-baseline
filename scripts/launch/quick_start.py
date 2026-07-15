import paramiko, time

FARM01 = {"hostname": "10.176.60.71", "username": "jiaqigu", "password": "lijia7272"}

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(**FARM01, timeout=30)

c.exec_command("pkill -9 -f train.py 2>/dev/null; true")
time.sleep(3)

# Skip pre-tokenize — compile + batch_size=6 are the real speedups
c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "CUDA_VISIBLE_DEVICES=6 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py "
    "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    "--v1_ckpt outputs/report_gen/best_model.pt "
    "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    "--batch_size 6 --ga_steps 1 --epochs 5 "
    "--num_workers 4 "
    "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    "--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    "--use_amp --compile --no-pre_tokenize "
    "--log_dir outputs/report_gen "
    "--save_interval 2000 --log_interval 10 "
    "> outputs/report_gen/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)

print("Waiting 180s (compile warmup + model load)...")
time.sleep(180)

stdin, stdout, stderr = c.exec_command(
    "tail -20 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log|head -1)")
print(stdout.read().decode(errors="replace"))

stdin2, stdout2, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader|grep '^6'")
print("GPU6:", stdout2.read().decode().strip())

stdin3, stdout3, stderr = c.exec_command(
    "ps -u jiaqigu|grep python")
print("Py:", stdout3.read().decode().strip() or "none")

c.close()
