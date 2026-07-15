import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Check the new log
stdin, stdout, stderr = c.exec_command("cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260712_153036.log")
print("New log:", stdout.read().decode())

# Run train.py directly to see errors
stdin, stdout, stderr = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "CUDA_VISIBLE_DEVICES=6 /home/jiaqigu/hidnet_env/bin/python -c \""
    "import sys; sys.path.insert(0, 'src'); "
    "sys.path.insert(0, '.'); "
    "from train import train; "
    "import argparse; "
    "class A: pass; "
    "a=A(); "
    "a.data_root='/mnt/nas1/disk07/public/mr_data/MR-RATE'; "
    "a.v1_ckpt='outputs/report_gen/best_model.pt'; "
    "a.qwen_path='/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct'; "
    "a.log_dir='outputs/report_gen'; "
    "a.batch_size=2; a.epochs=1; a.lr=1e-4; a.cnn_lr=1e-5; "
    "a.weight_decay=0.01; a.max_grad_norm=1.0; a.ga_steps=2; "
    "a.num_workers=2; a.use_amp=True; a.log_interval=1; a.save_interval=999999; "
    "a.max_samples=10; a.lora_r=8; a.lora_alpha=16; a.lora_drop=0.1; "
    "a.llm_dim=2048; a.grid=2; a.vit_dim=512; a.vit_heads=8; a.vit_depth=2; "
    "train(a)"
    "\" 2>&1 | tail -40",
    timeout=300)
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print(f"\nDirect run:\n{out[-3000:]}")
if err:
    print(f"STDERR: {err[-500:]}")

c.close()
