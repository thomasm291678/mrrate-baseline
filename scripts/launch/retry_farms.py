import paramiko, socket

servers = [
    ("farm01", "10.128.71.149"),
    ("farm02", "10.176.60.29"),
]

for name, ip in servers:
    # First just test TCP connectivity
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    result = s.connect_ex((ip, 22))
    s.close()
    
    if result == 0:
        print(f"{name} ({ip}): PORT 22 OPEN")
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
            s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null", timeout=10)
            print(o.read().decode().strip())
            s, o, e = c.exec_command("ls /home/jiaqigu/mrrate_hidnet/ 2>/dev/null || echo 'NO PROJ'", timeout=10)
            print(o.read().decode().strip())
            c.close()
        except Exception as ex:
            print(f"  SSH failed: {ex}")
    else:
        print(f"{name} ({ip}): PORT 22 CLOSED (err={result})")
