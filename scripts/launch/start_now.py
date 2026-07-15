import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill old GPU1 processes
c.exec_command("pkill -f 'train_eval_' 2>/dev/null; true")

# Launch
ts_cmd = '$(date +%Y%m%d_%H%M%S)'
cmd = (
    "cd /home/jiaqigu/mrrate_hidnet && "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "CUDA_VISIBLE_DEVICES=1 "
    "nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py "
    "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    "--v1_ckpt outputs/report_gen/best_model.pt "
    "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    "--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    "--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    "--use_amp --eval_samples 200 "
    "--log_dir outputs/report_gen "
    "--save_interval 2000 --log_interval 10 "
    f"> outputs/report_gen/train_eval_{ts_cmd}.log 2>&1 &"
)
c.exec_command(cmd)

print("Launched. Will check in 120s via peek...")
c.close()
