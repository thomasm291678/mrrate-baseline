import paramiko, io

README = """# MR-RATE 多版本项目

## 编码器版本

| Version | 脚本 | 说明 |
|---------|------|------|
| V4 (legacy) | `encoder_v4.py` / `train_v4.py` | UniFormer 编码器，已弃用 |
| **V5** | **`encoder_v5.py`** | MRRCNN 6 级 CNN + SE + Stage5 Attention，当前主力 |
| **V5 Phase 1** | **`train_v5_phase1.py`** | Encoder 对比学习（单模态 T1/T2/Flair） |
| **V5 Phase 2** | **`train_v5_phase2.py`** | Projector 对齐 Qwen embedding 空间 |
| **V5 Phase 3** | **`train_v5_phase3.py`** | Qwen LoRA 报告生成（冻结 V5+projector） |

## 评估版本 (evaluation/)

| Version | 文件 | 说明 |
|---------|------|------|
| v1 (legacy) | `evaluation/v1_legacy/unified_eval.py` | 14 类 Clinical 标签 + NLG + Diversity |
| evaluation.v1 (JSS) | `evaluation/reference/evaluation_v1_jss/` | 江上姝基于 VLM3D-Dockers 的 37 类标签版本 |
| **v2 (current)** | **`evaluation/v2/evaluation_v2.py`** | 融合 v1 + JSS，37 类逐病理 F1 + 14 类并行 + Composite |

## 数据

- **MR-RATE 数据集**：`/mnt/nas1/disk07/public/mr_data/MR-RATE/`
- **batch27 子集**：4,587 train + 190 val（快速实验用）
- **全量**：88,985 train + 3,781 val

## 服务器

| 名称 | IP | GPU |
|------|-----|-----|
| farm01 | 10.154.32.185 | RTX 6000 Ada × 8 |
| farm02 | 10.154.32.115 | RTX 4090D × 8 |
| farm03 | 10.176.60.70 | 2080Ti/Titan × 8 |
| farm04 | 10.176.60.71 | RTX 4090 × 8 |
| farm05 | 10.176.60.72 | RTX PRO 6000 × 8 |

## 当前进度

- **Phase 1** ✅ 完成 (batch27, 5 epochs, avg_loss=0.889, 基线=3.87)
- **Phase 2** ✅ 完成 (batch27, 3 epochs, avg_loss=0.0002)
- **Phase 3** ⏳ 待启动（等选定非 Qwen 模型）

## Quick Start (Phase 1)

```bash
python train_v5_phase1.py \\
  --batch_id batch27 --epochs 5 --batch_size 16 \\
  --augment --auto_resume \\
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE
```

## Quick Start (Phase 2)

```bash
python train_v5_phase2.py \\
  --encoder_ckpt outputs/report_gen/phase1_latest.pt \\
  --batch_id batch27 --epochs 3 --batch_size 4 \\
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct
```

## Quick Start (Evaluation)

```bash
python evaluation/v2/evaluation_v2.py \\
  --preds preds.json --refs refs.json \\
  --output results.json

# 多版本对比
python evaluation/v2/evaluation_v2.py \\
  --preds v0_preds.json --refs refs.json \\
  --compare v1_preds.json v2_preds.json
```
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(README.encode()), "/home/jiaqigu/mrrate_eval_git/README.md")
sftp.close()

c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git add README.md")
c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git commit -m 'docs: update README with multi-version overview (V4/V5/Phase1-3/eval v1-v2)'")
s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git push origin main 2>&1")
print(o.read().decode(), e.read().decode())

s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git log --oneline -5")
print("Log:", o.read().decode())

c.close()
print("Done!")
