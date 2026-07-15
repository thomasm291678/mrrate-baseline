import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Check watchdog
stdin, stdout, stderr = c.exec_command("ps aux | grep watchdog | grep -v grep")
print("Watchdog:", stdout.read().decode())

# Check if watchdog died
stdin, stdout, stderr = c.exec_command("cat /tmp/watchdog_out.log 2>/dev/null")
wd = stdout.read().decode()
print(f"Watchdog log: {wd[:1000] if wd else 'EMPTY'}")

# Directly run run.sh to see errors
print("\n=== Running run.sh directly ===")
stdin, stdout, stderr = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && bash -x run.sh 2>&1 | head -50",
    timeout=120)
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print(out[-3000:] if len(out) > 3000 else out)
if err:
    print(f"STDERR: {err[:500]}")

# Check if train.py exists and is readable
stdin, stdout, stderr = c.exec_command(
    "head -5 /home/jiaqigu/mrrate_hidnet/scripts/train.py && echo '---' && head -5 /home/jiaqigu/mrrate_hidnet/src/encoder.py")
print("\nCode check:", stdout.read().decode())

c.close()
