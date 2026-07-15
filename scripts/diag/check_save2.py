import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check training log directly for save config
stdin, stdout, stderr = c.exec_command(
    r"grep save_interval /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_20260713_004901.log 2>/dev/null", timeout=10)
print("Config:", stdout.read().decode().strip())

# Check all output files
stdin, stdout, stderr = c.exec_command(
    "ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/ 2>/dev/null", timeout=10)
print("Output_dir:", stdout.read().decode().strip())

c.close()
