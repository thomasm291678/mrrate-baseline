import paramiko

HOST = "10.176.60.72"; USER = "jiaqigu"; PASS = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet"
NAS = "/mnt/nas1/disk07/public/jiaqigu"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)
print("Connected.")

# First test: can farm05 reach github on port 22 (SSH)?
print("\n=== Testing GitHub SSH connectivity ===")
_, o, _ = client.exec_command("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -T git@github.com 2>&1")
out = o.read().decode().strip()
print(out[:300])

# Also test HTTPS with different approaches
print("\n=== Testing GitHub HTTPS (alternative routes) ===")
tests = [
    "curl -s --connect-timeout 5 https://github.com 2>&1 | head -1",
    "curl -s --connect-timeout 5 https://api.github.com 2>&1 | head -1",
    "ping -c 1 -W 3 github.com 2>&1 | head -2",
]
for t in tests:
    print(f"\n--- {t} ---")
    _, o, _ = client.exec_command(t)
    print(o.read().decode().strip()[:200])

client.close()
