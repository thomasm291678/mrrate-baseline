import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Write a debug script on farm01
debug_script = """
import torch
ckpt = torch.load('/home/jiaqigu/mrrate_hidnet/outputs/report_gen/best_model.pt', map_location='cpu', weights_only=False)
llm_st = ckpt.get('llm_state', {})
print("Total keys:", len(llm_st))
embed_keys = [k for k in llm_st.keys() if 'embed' in k.lower() or 'lm_head' in k.lower()]
print("Embed/lm_head keys:")
for k in sorted(embed_keys):
    print(f"  {k}: shape={list(llm_st[k].shape)}")
"""

c.exec_command(f"cat > /tmp/debug_ckpt_info.py << 'ENDOFPYTHON'\n{debug_script}\nENDOFPYTHON")

stdin, stdout, stderr = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && /home/jiaqigu/hidnet_env/bin/python /tmp/debug_ckpt_info.py 2>&1",
    timeout=120)
print("Checkpoint keys:\n", stdout.read().decode(errors="replace"))

c.close()
