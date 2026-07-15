import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"

# Check what's broken
tests = [
    ("import torch; print('torch', torch.cuda.is_available())", "torch"),
    ("from transformers import AutoModelForCausalLM; print('trans OK')", "transformers"),
    ("from peft import LoraConfig; print('peft OK')", "peft"),
    ("import torch.nn as nn; print('nn OK')", "nn"),
]

for code, name in tests:
    stdin, stdout, stderr = c.exec_command(
        f"{PY} -c '{code}' 2>&1", timeout=10)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err and "Error" in err:
        print(f"  BROKEN: {name} - {err[:100]}")
    else:
        print(f"  OK: {name} - {out[:80]}")

c.close()
