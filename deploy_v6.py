import paramiko, io

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=10)
D = "/home/jiaqigu/mrrate_hidnet"

c.exec_command("pkill -9 -f train_phase 2>/dev/null; pkill -9 -f train_v5 2>/dev/null; sleep 1")
c.exec_command(f"mkdir -p {D}/v6; rm -f {D}/outputs/report_gen/phase1*.pt {D}/outputs/report_gen/phase1*.log")

V6 = r"C:\Users\HP\Documents\5555\v6"
sf = c.open_sftp()
for fname in ["model.py", "train_phase1.py", "train_phase3.py", "mrrate_dataset.py"]:
    with open(f"{V6}\\{fname}", "rb") as f:
        sf.putfo(f, f"{D}/v6/{fname}")
    print(f"  Uploaded: v6/{fname}")
sf.close()

scr = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
source ~/hidnet_env/bin/activate
cd {D}/v6
rm -f {D}/outputs/report_gen/phase1*.log {D}/outputs/report_gen/phase1*.pt
python train_phase1.py \\
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
  --log_dir {D}/outputs/report_gen \\
  --batch_id batch27 \\
  --epochs 5 \\
  --batch_size 16 \\
  --lr 3e-4 \\
  2>&1 | tee {D}/outputs/report_gen/phase1.log
"""

sf = c.open_sftp()
sf.putfo(io.BytesIO(scr.encode()), f"{D}/v6/run_phase1.sh")
sf.close()
c.exec_command(f"chmod +x {D}/v6/run_phase1.sh")
t = c.get_transport()
ch = t.open_session()
ch.exec_command(f"cd {D}/v6 && nohup bash run_phase1.sh > /dev/null 2>&1 < /dev/null & echo OK")
print(f"\nPhase 1 launched on farm05 GPU1: {ch.recv(1024).decode()}")
ch.close()
c.close()
