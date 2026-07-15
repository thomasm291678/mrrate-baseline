"""Deploy V5 code + checkpoint to farm05 and launch training"""
import paramiko, time, sys

FARM04 = ("10.176.60.71", "jiaqigu", "lijia7272")
FARM05 = ("10.176.60.72", "jiaqigu", "lijia7272")

print("=== 1. Deploy code to farm05 ===")
c5 = paramiko.SSHClient()
c5.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c5.connect(FARM05[0], username=FARM05[1], password=FARM05[2], timeout=15)

# Create directories
c5.exec_command("mkdir -p /home/jiaqigu/mrrate_hidnet/outputs/report_gen")
c5.exec_command("mkdir -p /home/jiaqigu/mrrate_hidnet/scripts")
time.sleep(1)

# Upload code files
sftp = c5.open_sftp()
print("  Uploading encoder_v5.py...")
sftp.put("encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py")
print("  Uploading train_v5.py...")
sftp.put("train_v5.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5.py")
print("  Uploading mrrate_dataset.py...")
sftp.put("server_code/mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py")
sftp.close()
print("  Code deployed.")

# Verify
stdin, out, err = c5.exec_command("ls -la /home/jiaqigu/mrrate_hidnet/encoder_v5.py /home/jiaqigu/mrrate_hidnet/scripts/train_v5.py /home/jiaqigu/mrrate_hidnet/mrrate_dataset.py")
print("  Files:", out.read().decode().strip())

print("\n=== 2. Transfer checkpoint farm04 -> farm05 ===")
# Step 1: Get checkpoint size
c4 = paramiko.SSHClient()
c4.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c4.connect(FARM04[0], username=FARM04[1], password=FARM04[2], timeout=15)
stdin, out, err = c4.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt")
print("  Checkpoint:", out.read().decode().strip())

# Step 2: Copy ckpt from farm04 to farm05 via farm04 -> local -> farm05 (since no direct scp between servers)
# Actually let's try scp between servers
print("  Attempting direct scp farm04->farm05...")
stdin, out, err = c4.exec_command(
    "sshpass -p 'lijia7272' scp -o StrictHostKeyChecking=no "
    "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt "
    "jiaqigu@10.176.60.72:/home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt 2>&1"
)
result = out.read().decode().strip()
err_result = err.read().decode().strip()
print("  scp stdout:", result)
print("  scp stderr:", err_result)

# Fallback: upload from local if direct scp failed
verify_stdin, verify_out, verify_err = c5.exec_command(
    "ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt 2>/dev/null"
)
ckpt_ok = verify_out.read().decode().strip()
if not ckpt_ok or "No such file" in ckpt_ok:
    print("  Direct scp failed, uploading from local...")
    # Download from farm04 to local first
    sftp4 = c4.open_sftp()
    sftp4.get("/home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt", "latest_step_temp.pt")
    sftp4.close()
    print(f"  Downloaded, uploading to farm05...")
    sftp5 = c5.open_sftp()
    sftp5.put("latest_step_temp.pt", "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt")
    sftp5.close()
    import os
    os.remove("latest_step_temp.pt")
    print("  Checkpoint transferred.")

c4.close()

# Verify checkpoint
stdin, out, err = c5.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt")
print("  farm05 ckpt:", out.read().decode().strip())

print("\n=== 3. Launch training on farm05 GPU0 ===")
train_cmd = (
    "cd /home/jiaqigu/mrrate_hidnet && "
    "CUDA_VISIBLE_DEVICES=0 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train_v5.py "
    "--phase encoder --modality all --augment "
    "--batch_id batch27 --epochs 5 --batch_size 16 --num_workers 4 "
    "--lr 3e-4 --wd 1e-4 --grad_clip 1.0 "
    "--grid 2 --base_ch 32 "
    "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    "--log_dir outputs/report_gen "
    "--save_interval 500 --auto_save_interval 50 --log_interval 5 "
    "--auto_resume "
    "> outputs/report_gen/train_v5_b27_farm05.log 2>&1 &"
)
print(f"  Command: {train_cmd[:200]}...")
stdin, out, err = c5.exec_command(train_cmd)
time.sleep(3)

# Verify running
stdin, out, err = c5.exec_command("ps aux | grep train_v5 | grep -v grep")
proc = out.read().decode().strip()
if proc:
    print("  ✅ Training started!")
    print(" ", proc)
else:
    print("  ⚠️ No process found, checking log...")
    stdin, out, err = c5.exec_command("tail -10 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_b27_farm05.log 2>/dev/null")
    print(" ", out.read().decode().strip())

c5.close()
print("\nDone!")
