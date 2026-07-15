import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check if t1 tmux session is actually alive
s, o, e = c.exec_command("tmux ls -F '#{session_name} #{session_attached} #{?session_alive,alive,dead}' 2>/dev/null", timeout=10)
print("tmux sessions:", o.read().decode().strip())

# Check if any python still running
s, o, e = c.exec_command("pgrep -a python | grep -i 'train_v4\|mrrate' | head -5", timeout=10)
print("Python procs:", o.read().decode().strip() or "(none)")

# Check exit code of tmux pane (if dead)
s, o, e = c.exec_command("tmux display-message -t t1 -p '#{pane_dead}' 2>/dev/null", timeout=10)
dead = o.read().decode().strip()
print(f"Pane dead: {dead}")

if dead == "1":
    s, o, e = c.exec_command("tmux display-message -t t1 -p '#{pane_exited}' 2>/dev/null", timeout=10)
    print(f"Pane exited: {o.read().decode().strip()}")

# Get full tmux buffer
s, o, e = c.exec_command("tmux capture-pane -t t1 -p -S -300 2>/dev/null | tail -30", timeout=10)
print("\n=== tmux tail ===")
out = o.read().decode().strip()
print(out if out else "(empty/not running)")

# Check if GPU3 has any process still
s, o, e = c.exec_command("fuser /dev/nvidia3 2>/dev/null", timeout=10)
print(f"\nGPU3 fuser: {o.read().decode().strip()}")

c.close()
