import paramiko, socket

ips = ["10.176.60.71", "10.176.60.70", "10.128.71.149", "10.176.60.29"]

for ip in ips:
    s = socket.socket()
    s.settimeout(8)
    r = s.connect_ex((ip, 22))
    s.close()
    if r == 0:
        print(f"{ip}:22 OPEN")
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
            s, o, e = c.exec_command("hostname && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4", timeout=15)
            print(o.read().decode())
            c.close()
        except Exception as ex:
            print(f"  SSH failed: {ex}")
    else:
        print(f"{ip}:22 CLOSED")
