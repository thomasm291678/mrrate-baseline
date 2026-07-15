import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

# Write install+launch script on server
script = """#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
/home/jiaqigu/hidnet_env/bin/pip install evaluate bert_score scikit-learn rouge_score -q 2>/dev/null
echo "DEPS DONE $(date)"
TS=$(date +%Y%m%d_%H%M%S)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --v1_ckpt outputs/report_gen/best_model.pt \
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
    --batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 \
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \
    --vit_dim 512 --vit_heads 8 --vit_depth 2 \
    --use_amp --eval_samples 200 \
    --log_dir outputs/report_gen \
    --save_interval 2000 --log_interval 10 \
    > outputs/report_gen/train_eval_${TS}.log 2>&1 &
echo "LAUNCHED PID=$! LOG=train_eval_${TS}.log"
"""

# Write script to server
c.exec_command(f"cat > /home/jiaqigu/mrrate_hidnet/start_eval.sh << 'ENDOFSCRIPT'\n{script}\nENDOFSCRIPT")
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/start_eval.sh")

# Run in background on server
c.exec_command("nohup bash /home/jiaqigu/mrrate_hidnet/start_eval.sh > /home/jiaqigu/mrrate_hidnet/outputs/report_gen/startup.log 2>&1 &")

# Upload files
print("Uploading code files...")
sftp = c.open_sftp()
for local, remote in [
    ("eval_report.py", "/home/jiaqigu/mrrate_hidnet/eval_report.py"),
    ("train.py", "/home/jiaqigu/mrrate_hidnet/scripts/train.py"),
    ("encoder.py", "/home/jiaqigu/mrrate_hidnet/src/encoder.py"),
]:
    sftp.put(f"C:/Users/HP/Documents/5555/{local}", remote)
sftp.close()

print("Waiting 180s for install + launch...")
time.sleep(180)

# Check
stdin, stdout, stderr = c.exec_command("cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/startup.log", timeout=15)
print("Startup log:", stdout.read().decode(errors="replace").strip())

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'", timeout=15)
print("GPU1:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command(
    "tail -5 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_*.log 2>/dev/null | head -1) 2>/dev/null", timeout=15)
print("Train log:", stdout.read().decode(errors="replace").strip()[:300])

c.close()
