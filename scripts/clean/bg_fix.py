import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Write a self-contained fix+launch script on server
script = '''#!/bin/bash
set -e
cd /home/jiaqigu/mrrate_hidnet
PIP=/home/jiaqigu/hidnet_env/bin/pip
PY=/home/jiaqigu/hidnet_env/bin/python

echo "=== $(date) Fixing torch ==="
$PIP install evaluate scikit-learn rouge_score -q 2>/dev/null
$PIP install torch==2.5.1 --force-reinstall --no-deps -q 2>/dev/null
$PIP install torchvision --force-reinstall --no-deps -q 2>/dev/null

echo "=== $(date) Verifying ==="
$PY -c "import torch, evaluate, sklearn; print(torch.__version__, torch.cuda.is_available())"

echo "=== $(date) Launching ==="
pkill -9 -f train.py 2>/dev/null; sleep 3
TS=$(date +%Y%m%d_%H%M%S)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 nohup $PY -u scripts/train.py \
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
echo "PID=$! LOG=train_eval_${TS}.log"
echo "=== $(date) DONE ==="
'''

# Write and execute
c.exec_command(f"cat > /home/jiaqigu/mrrate_hidnet/fix_launch.sh << 'EOF'\n{script}\nEOF", timeout=5)
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/fix_launch.sh", timeout=5)
c.exec_command("nohup bash /home/jiaqigu/mrrate_hidnet/fix_launch.sh > /home/jiaqigu/mrrate_hidnet/outputs/report_gen/fix_launch.log 2>&1 &", timeout=5)
print("fix_launch.sh triggered in background on server")
print("Check: cat outputs/report_gen/fix_launch.log")
c.close()
