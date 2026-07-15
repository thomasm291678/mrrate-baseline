import paramiko

host = "10.176.60.70"
port = 22
username = "jiaqigu"
password = "lijia7272"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, port=port, username=username, password=password, timeout=15)
    
    print("=" * 70)
    print("  NVIDIA-SMI GPU Utilization")
    print("=" * 70)
    stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null")
    print(stdout.read().decode())
    
    ssh.close()
except Exception as e:
    print(f"Error: {e}")
