import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Kill all
c.exec_command("pkill -9 -f 'train.py' 2>/dev/null; pkill -9 -f 'watchdog.sh' 2>/dev/null; true")
time.sleep(2)

# Check train.py fix
stdin, stdout, stderr = c.exec_command("sed -n '142,160p' /home/jiaqigu/mrrate_hidnet/scripts/train.py")
print("train.py lines 142-160:")
print(stdout.read().decode())

# Check which checkpoints exist
stdin, stdout, stderr = c.exec_command("ls -la /home/jiaqigu/mrrate_hidnet/outputs/report_gen/best_model.pt")
print("Model:", stdout.read().decode().strip())

# Check tokenizer size in checkpoint
stdin, stdout, stderr = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "/home/jiaqigu/hidnet_env/bin/python -c \""
    "import torch; "
    "ckpt = torch.load('outputs/report_gen/best_model.pt', map_location='cpu', weights_only=False); "
    "llm_st = ckpt.get('llm_state', {}); "
    "keys = [k for k in llm_st.keys() if 'embed' in k or 'lm_head' in k]; "
    "print('Checkpoint embed/lm_head keys:'); "
    "for k in sorted(keys): print(f'  {k}: {list(llm_st[k].shape)}')"
    "\" 2>&1", timeout=120)
print("\nCheckpoint info:")
print(stdout.read().decode())

c.close()
