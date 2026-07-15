import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_b27_20260713_221953.log"

# Extract loss curve
s, o, e = c.exec_command(f"grep ' S0' {log} | grep -oP 'loss=[\\d.]+' | sed 's/loss=//'", timeout=10)
losses = [float(x) for x in o.read().decode().strip().split('\n')]
s, o, e = c.exec_command(f"grep ' S0' {log} | grep -oP 'avg_loss=[\\d.]+' | sed 's/avg_loss=//'", timeout=10)
avg_losses = [float(x) for x in o.read().decode().strip().split('\n')]

# Analyze
end_avg = avg_losses[-1] if avg_losses else 0
last10 = losses[-10:] if len(losses) >= 10 else losses
last10_avg = sum(last10) / len(last10)

print(f"Epoch1 avg_loss endpoints:")
e1 = avg_losses[:len(avg_losses)//2] if len(avg_losses) > 1 else avg_losses
e2 = avg_losses[len(avg_losses)//2:] if len(avg_losses) > 1 else []
if e1:
    print(f"  Epoch1 start: {e1[0]:.4f}  end: {e1[-1]:.4f}  (Δ {e1[-1]-e1[0]:.4f})")
if e2:
    print(f"  Epoch2 start: {e2[0]:.4f}  end: {e2[-1]:.4f}  (Δ {e2[-1]-e2[0]:.4f})")

print(f"\nTrainable params: 86.8M (encoder only, no Qwen)")
print(f"Batch27: 4587 train, 190 val")
print(f"Current epochs: 2, avg_loss: {end_avg:.4f}")
print(f"Below log(48)=3.87? {'YES' if end_avg < 3.87 else 'NO'}")
print(f"Still improving? {'YES (Δ>' + str(abs(e2[-1]-e2[0]))[:4] + ')' if e2 and abs(e2[-1]-e2[0]) > 0.01 else 'maybe plateauing'}")

rnd = 3 * 572
remain = (572 + rnd)
print(f"\nSuggested: 5 total epochs (~{remain} steps, ~{rnd*35/60:.0f}min more)")
print(f"Re-run with: --epochs 5 --auto_resume") 

c.close()
