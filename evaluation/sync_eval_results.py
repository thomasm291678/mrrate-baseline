import paramiko

HOST = "10.176.60.72"; USER = "jiaqigu"; PASS = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet/evaluation_v2.py"
NAS = "/mnt/nas1/disk07/public/jiaqigu/evaluation/v2/evaluation_v2.py"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(HOST, username=USER, password=PASS, timeout=10)
    _, o, _ = client.exec_command(f"cp {REMOTE} {NAS}")
    print(o.read().decode().strip())

    _, o, _ = client.exec_command(f"ls -la {NAS}")
    print(o.read().decode().strip())

    client.close()
    print(f"\nDone → {NAS}")
except Exception as e:
    print(f"Server unreachable: {e}")
