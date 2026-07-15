import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check if there's a watchdog/job killer
for check in [
    "systemctl list-units --type=service 2>/dev/null | grep -iE 'watchdog|killer|oom|sentry|monitor' | head -10",
    "ps aux | grep -iE 'watchdog|monit|sentry' | grep -v grep | head -10",
    "crontab -l 2>/dev/null | head -20",
    "cat /etc/cron.d/* 2>/dev/null | grep -iE 'kill|clean|purge' | head -10",
    "which tmux screen 2>/dev/null",
]:
    s, o, e = c.exec_command(check, timeout=10)
    out = o.read().decode().strip()
    if out:
        print(f"\n=== {check[:60]} ===")
        print(out)

c.close()
