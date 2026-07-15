import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272")
for g in range(4):
    _, o, _ = c.exec_command("tmux capture-pane -t gpu" + str(g) + " -p | grep -E 'loss=|E00[0-9].*%' | tail -2")
    out = o.read().decode().strip()
    if out: print("GPU", g, ":", out[:130])
_, o, _ = c.exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -4")
print(o.read().decode().replace("\n", ", "))
c.close()
