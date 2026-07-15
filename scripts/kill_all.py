import paramiko

FARMS = {
    "farm02": "10.176.60.70",
    "farm03": "10.176.60.71",
    "farm04": "10.176.60.72",
    "farm05": "10.176.60.73",
}

for name, host in FARMS.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username="jiaqigu", password="lijia7272", timeout=5)
        c.exec_command("pkill -9 -f train_v5; pkill -9 -f gen_and_eval; pkill -9 -f run_eval; tmux kill-server 2>/dev/null")
        print(f"{name} ({host}): killed")
        c.close()
    except Exception as e:
        print(f"{name} ({host}): {e}")

print("Done")
