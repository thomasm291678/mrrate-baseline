import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272", timeout=15)

# Check everything
checks = [
    "ls /home/jiaqigu/mrrate_hidnet/outputs/report_gen/BrainMVP_uniformer.pt 2>/dev/null && echo 'BrainMVP: OK' || echo 'BrainMVP: MISSING'",
    "ls /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv 2>/dev/null && echo 'Data: OK' || echo 'Data: MISSING'",
    "ls /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct/ 2>/dev/null | wc -l",
    "df -h /home/jiaqigu/ | tail -1",
    "which python && python --version",
    "/home/jiaqigu/hidnet_env/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())' 2>/dev/null || echo 'NO hidnet_env'",
]
for cmd in checks:
    s, o, e = c.exec_command(cmd, timeout=15)
    print(f"  {o.read().decode().strip()}")

c.close()
