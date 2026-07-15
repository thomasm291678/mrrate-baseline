import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272")

NAS = "/mnt/nas1/disk07/public/jiaqigu"
checks = [
    f"echo '=== evaluation/v2/ ===' && ls -la {NAS}/evaluation/v2/",
    f"echo '=== prompt ===' && wc -c {NAS}/evaluation/v2/llm_prompt_v2.md",
    f"echo '=== root py ===' && ls {NAS}/*.py 2>&1",
    f"echo '=== root md ===' && ls {NAS}/*.md 2>&1",
]
for cmd in checks:
    _, o, _ = c.exec_command(cmd)
    print(o.read().decode().strip())
    print()

c.close()
