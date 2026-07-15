import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# 关键：检查 semaphore
s, o, e = c.exec_command("ipcs -s | head -30 && echo '---' && ipcs -s | wc -l", timeout=15)
print("=== Semaphores ===")
print(o.read().decode().strip())

# 系统 semaphore 限制
s, o, e = c.exec_command("cat /proc/sys/kernel/sem 2>/dev/null", timeout=10)
print("\n=== Kernel sem limits (SEMMSL SEMMNS SEMOPM SEMMNI) ===")
print(o.read().decode().strip())

# 检查 jiaqigu 的 semaphore 残留
s, o, e = c.exec_command("ipcs -s | grep jiaqigu | wc -l", timeout=10)
print(f"\n=== jiaqigu's semaphores: {o.read().decode().strip()} ===")

# 也可能是 nohup + SSH 断开导致 SIGHUP 杀进程
s, o, e = c.exec_command(
    "cat /proc/sys/kernel/hung_task_timeout_secs 2>/dev/null", timeout=10)
print(f"\n=== hung_task_timeout: {o.read().decode().strip()} ===")

# 检查上次崩溃前是否有 nvidia Xid error
s, o, e = c.exec_command(
    "dmesg -T 2>/dev/null | grep -i 'nvidia\|xid\|nvrm' | tail -20", timeout=15)
print("\n=== Nvidia kernel messages ===")
out = o.read().decode().strip()
print(out if out else "(none or no access)")

# 检查 recent crash 日志有没有任何异常
s, o, e = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_20260713_153727.log 2>/dev/null",
    timeout=10)
print("\n=== Most recent log FULL ===")
print(o.read().decode().strip())

c.close()
