import paramiko, time

# Step 1: Download BrainMVP_ckpt from farm04 to local temp
print("Downloading BrainMVP_uniformer.pt from farm04...")
c4 = paramiko.SSHClient()
c4.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c4.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)
sftp4 = c4.open_sftp()
tmp = r"C:\Users\HP\Documents\5555\BrainMVP_uniformer.pt"
sftp4.get("/home/jiaqigu/mrrate_hidnet/outputs/report_gen/BrainMVP_uniformer.pt", tmp)
sftp4.close()
c4.close()
print(f"Downloaded to {tmp}")

# Step 2: Upload to farm03
print("Uploading to farm03...")
c3 = paramiko.SSHClient()
c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c3.connect("10.176.60.70", username="jiaqigu", password="lijia7272", timeout=30)
sftp3 = c3.open_sftp()
sftp3.put(tmp, "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/BrainMVP_uniformer.pt")
sftp3.close()

# Verify
s, o, e = c3.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/BrainMVP_uniformer.pt", timeout=10)
print(o.read().decode().strip())

# Check available python
s, o, e = c3.exec_command(
    "for py in /home/jiaqigu/hidnet_env/bin/python /home/jiaqigu/*/bin/python /home/jiaqigu/miniconda*/bin/python $(which python) ; do "
    "  [ -x $py ] && echo FOUND: $py && $py -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>/dev/null && break; "
    "done",
    timeout=30)
print(o.read().decode().strip())

c3.close()

# Clean up local temp
import os
os.remove(tmp)
print("Cleaned up temp file")
