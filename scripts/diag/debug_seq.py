import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check sequential log
s, o, e = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/v4_sequential_20260713_160403.log 2>/dev/null",
    timeout=10)
print("=== Sequential log ===")
print(o.read().decode())

# Check if process running
s, o, e = c.exec_command("ps aux | grep -E 'v4_sequential|train_v4' | grep -v grep", timeout=10)
print("=== Processes ===")
print(o.read().decode())

# Any T1 log?
s, o, e = c.exec_command(
    "ls -la /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_*.log 2>/dev/null || echo 'no t1 log'",
    timeout=10)
print("=== T1 log ===")
print(o.read().decode())

# Direct test
s, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "/home/jiaqigu/hidnet_env/bin/python -c 'from scripts.train_v4 import *; print(IGNORE_INDEX)' 2>&1",
    timeout=30)
print("=== Import test ===")
print(o.read().decode())

# Run train_v4 directly for 5 sec to see immediate error
s, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "timeout 30 /home/jiaqigu/hidnet_env/bin/python -u scripts/train_v4.py "
    "--modality t1 --projector attn --phase uniformer --augment --auto_resume "
    "--brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt "
    "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    "--batch_size 4 --ga_steps 2 --epochs 3 --num_workers 2 "
    "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    "--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    "--use_amp --eval_samples 100 "
    "--log_dir outputs/report_gen "
    "--save_interval 200 --auto_save_interval 100 --log_interval 10 "
    "2>&1",
    timeout=60)
print("\n=== Dry run ===")
print(o.read().decode())

c.close()
