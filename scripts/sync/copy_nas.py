import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

SRC = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
NAS_QCK = "/mnt/nas1/disk07/public/jiaqigu_ckpts"

# Copy step checkpoints + best_model
stdin, o, e = c.exec_command(
    f"cp -v {SRC}/step_*.pt {NAS_QCK}/ 2>&1 && "
    f"cp -v {SRC}/best_model.pt {NAS_QCK}/ 2>&1 && "
    f"ls -lh {NAS_QCK}/",
    timeout=600)
print(o.read().decode())

c.close()
