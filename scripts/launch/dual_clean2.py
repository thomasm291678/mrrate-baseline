import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"
BASE = "cd /home/jiaqigu/mrrate_hidnet &&"

# Kill all
c.exec_command("pkill -9 -f 'python.*train.py' 2>/dev/null; true", timeout=5)
time.sleep(10)
c.exec_command("fuser -k /dev/nvidia0 /dev/nvidia2 2>/dev/null; true", timeout=5)
time.sleep(10)

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^0,|^2,'",
    timeout=10)
print("After cleanup:", o.read().decode().strip())

# Launch GPU0
ts = time.strftime("%Y%m%d_%H%M%S")
cmd = BASE + " CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
cmd += PY + " -u scripts/train.py "
cmd += "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
cmd += "--v1_ckpt outputs/report_gen/best_model.pt "
cmd += "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
cmd += "--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
cmd += "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
cmd += "--vit_dim 512 --vit_heads 8 --vit_depth 2 "
cmd += "--use_amp --eval_samples 200 "
cmd += "--log_dir outputs/report_gen "
cmd += "--save_interval 200 --log_interval 10 "
cmd += "> outputs/report_gen/train_gpu0_" + ts + ".log 2>&1 &"
c.exec_command("nohup bash -c '" + cmd + "' &", timeout=5)
gpu0_ts = ts
print("GPU0 launched: train_gpu0_" + ts)

time.sleep(5)

# Launch GPU2
ts = time.strftime("%Y%m%d_%H%M%S")
cmd = BASE + " CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
cmd += PY + " -u scripts/train.py "
cmd += "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
cmd += "--v1_ckpt outputs/report_gen/best_model.pt "
cmd += "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
cmd += "--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
cmd += "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
cmd += "--vit_dim 512 --vit_heads 8 --vit_depth 2 "
cmd += "--use_amp --eval_samples 200 "
cmd += "--log_dir outputs/report_gen "
cmd += "--save_interval 200 --log_interval 10 "
cmd += "> outputs/report_gen/train_gpu2_" + ts + ".log 2>&1 &"
c.exec_command("nohup bash -c '" + cmd + "' &", timeout=5)
gpu2_ts = ts
print("GPU2 launched: train_gpu2_" + ts)

time.sleep(160)

s, o, e = c.exec_command(
    "echo GPU0; tail -3 outputs/report_gen/train_gpu0_" + gpu0_ts + ".log; "
    "echo GPU2; tail -3 outputs/report_gen/train_gpu2_" + gpu2_ts + ".log; "
    "echo GPUS; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E '^0,|^2,'",
    timeout=10)
print(o.read().decode())

c.close()
