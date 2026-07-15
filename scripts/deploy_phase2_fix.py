import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=10)

D = "/home/jiaqigu/mrrate_hidnet"
CKPT = f"{D}/outputs/report_gen/phase1_latest.pt"
QWEN = "/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct"
DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"

c.exec_command("pkill -9 -f train_v5_phase2 2>/dev/null; sleep 1")
c.exec_command(f"rm -f {D}/outputs/report_gen/phase2*.pt {D}/outputs/report_gen/phase2.log")

scr = f"""#!/bin/bash
source ~/hidnet_env/bin/activate
cd {D}
python train_v5_phase2.py \\
  --encoder_ckpt {CKPT} \\
  --qwen_path {QWEN} \\
  --data_root {DATA} \\
  --log_dir {D}/outputs/report_gen \\
  --batch_id batch27 \\
  --epochs 3 \\
  --batch_size 4 \\
  --lr 1e-4 \\
  --max_text_len 256 \\
  2>&1 | tee outputs/report_gen/phase2.log
echo "EXIT=$?" >> outputs/report_gen/phase2.log
"""

sf = c.open_sftp()
sf.putfo(__import__("io").BytesIO(scr.encode()), f"{D}/run_phase2.sh")
sf.close()

c.exec_command(f"chmod +x {D}/run_phase2.sh")
_, o, _ = c.exec_command(f"cd {D} && nohup bash run_phase2.sh &>/dev/null & echo $!")
print("PID:", o.read().decode().strip())

time.sleep(90)
_, o, _ = c.exec_command(f"tail -8 {D}/outputs/report_gen/phase2.log 2>&1")
print(o.read().decode())
_, o, _ = c.exec_command("ps aux | grep train_v5 | grep -v grep | head -1")
print(o.read().decode().strip()[:120] or "none")
_, o, _ = c.exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -1")
print("GPU:", o.read().decode().strip())
c.close()
