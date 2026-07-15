# FedAvg 多卡并行训练操作手册 — MRRCNN Phase 1

> 目标：将 MR-RATE 全量数据（88,985 样本）按 batch 切分为 4 个子集，在 4 张 A6000 上并行训练 MRRCNN Phase 1 对比学习，然后用 FedAvg 加权平均 + Head 对齐 + BN 重校准合并权重。

---

## 环境信息

- **服务器**：farm04 (10.176.60.71)
- **用户**：jiaqigu
- **项目目录**：`/home/jiaqigu/mrrate_hidnet`
- **GPU**：8× NVIDIA RTX A6000 48GB（使用 GPU0-3）
- **数据集**：`/mnt/nas1/disk07/public/mr_data/MR-RATE`（88,985 train + 3,781 val）

---

## 文件清单

| 文件 | 操作 | 路径 |
|------|------|------|
| `server_code/mrrate_dataset.py` | **已修改** ✅ | `batch_id` 支持逗号分隔多 batch |
| `train_v5.py` | **已修改** ✅ | 新增 `--init_from` 参数 |
| `generate_init.py` | **新建** ✅ | 生成共享初始权重 |
| `merge_encoders.py` | **新建** ✅ | FedAvg + Head对齐 + BN重算 |

---

## 第一步：部署代码到服务器

上传 5 个文件到 farm04：

```python
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272")

sftp = c.open_sftp()

# 核心文件
sftp.put("encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py")
sftp.put("train_v5.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5.py")
sftp.put("server_code/mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py")

# 新增脚本
sftp.put("generate_init.py", "/home/jiaqigu/mrrate_hidnet/generate_init.py")
sftp.put("merge_encoders.py", "/home/jiaqigu/mrrate_hidnet/merge_encoders.py")

sftp.close()
c.close()
```

---

## 第二步：生成共享初始权重

在服务器上执行（单卡，约 5 秒）：

```bash
cd /home/jiaqigu/mrrate_hidnet

python generate_init.py \
    --seed 42 \
    --grid 2 \
    --base_ch 32 \
    --out outputs/init_weights.pt
```

输出文件：`outputs/init_weights.pt`

---

## 第三步：数据切分方案

| GPU | batch_id | 预计样本数 |
|-----|----------|-----------|
| GPU0 | batch00~batch06 | ~22,000 |
| GPU1 | batch07~batch13 | ~22,000 |
| GPU2 | batch14~batch20 | ~22,000 |
| GPU3 | batch21~batch27 | ~23,000 |

> 按 batch 切分保证了同一 patient 的所有扫描（T1/T2/Flair）在同一个子集中，InfoNCE 对比学习的 positive pair 不会被打散。

---

## 第四步：4 卡并行启动 Phase 1

### GPU0：batch00-06

```bash
cd /home/jiaqigu/mrrate_hidnet

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_v5.py \
    --phase encoder --modality all --augment \
    --batch_id batch00,batch01,batch02,batch03,batch04,batch05,batch06 \
    --epochs 5 --batch_size 16 --num_workers 4 \
    --lr 3e-4 --wd 1e-4 --grad_clip 1.0 \
    --grid 2 --base_ch 32 \
    --init_from outputs/init_weights.pt \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --log_dir outputs/gpu0 --auto_resume \
    --save_interval 500 --auto_save_interval 100 --log_interval 10 \
    > outputs/gpu0/train.log 2>&1 &
```

### GPU1：batch07-13

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/train_v5.py \
    --phase encoder --modality all --augment \
    --batch_id batch07,batch08,batch09,batch10,batch11,batch12,batch13 \
    --epochs 5 --batch_size 16 --num_workers 4 \
    --lr 3e-4 --wd 1e-4 --grad_clip 1.0 \
    --grid 2 --base_ch 32 \
    --init_from outputs/init_weights.pt \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --log_dir outputs/gpu1 --auto_resume \
    --save_interval 500 --auto_save_interval 100 --log_interval 10 \
    > outputs/gpu1/train.log 2>&1 &
```

### GPU2：batch14-20

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/train_v5.py \
    --phase encoder --modality all --augment \
    --batch_id batch14,batch15,batch16,batch17,batch18,batch19,batch20 \
    --epochs 5 --batch_size 16 --num_workers 4 \
    --lr 3e-4 --wd 1e-4 --grad_clip 1.0 \
    --grid 2 --base_ch 32 \
    --init_from outputs/init_weights.pt \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --log_dir outputs/gpu2 --auto_resume \
    --save_interval 500 --auto_save_interval 100 --log_interval 10 \
    > outputs/gpu2/train.log 2>&1 &
```

### GPU3：batch21-27

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/train_v5.py \
    --phase encoder --modality all --augment \
    --batch_id batch21,batch22,batch23,batch24,batch25,batch26,batch27 \
    --epochs 5 --batch_size 16 --num_workers 4 \
    --lr 3e-4 --wd 1e-4 --grad_clip 1.0 \
    --grid 2 --base_ch 32 \
    --init_from outputs/init_weights.pt \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --log_dir outputs/gpu3 --auto_resume \
    --save_interval 500 --auto_save_interval 100 --log_interval 10 \
    > outputs/gpu3/train.log 2>&1 &
```

---

## 第五步：监控训练

```bash
# 分别查看 4 张卡的日志
tail -f outputs/gpu0/train.log
tail -f outputs/gpu1/train.log
tail -f outputs/gpu2/train.log
tail -f outputs/gpu3/train.log

# 查看 GPU 使用情况
watch -n 2 nvidia-smi

# 查看训练进程
ps -u jiaqigu | grep train_v5

# 杀掉所有训练进程
pkill -9 -f train_v5
```

---

## 第六步：合并权重（Phase 1 结束后）

先获取各 GPU 的实际样本数（从日志中找 `Train: XXXX`），然后执行：

```bash
cd /home/jiaqigu/mrrate_hidnet

python merge_encoders.py \
    --ckpts \
        outputs/gpu0/latest_step.pt \
        outputs/gpu1/latest_step.pt \
        outputs/gpu2/latest_step.pt \
        outputs/gpu3/latest_step.pt \
    --sample_counts 22000 22000 22000 22000 \
    --out outputs/merged_phase1.pt \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --recal_limit 20000 \
    --recal_batch_size 8
```

### 这个脚本做了什么

```
Step 1: 加载 4 个 checkpoint 的 encoder_state
Step 2: Head 对齐 — 对每个模态 (T1/T2/Flair) 的 StageAttention
         4 heads × 24 种排列 → 找最优匹配，重排 MHA 权重
Step 3: FedAvg 加权平均 — 按样本量加权，自动跳过：
         - ContrasiveHead（Phase 2/3 不需要）
         - BN running_mean / running_var（需要重新计算）
         全部卷积核、BN γβ、LayerNorm γβ、投影层、MLP 参与平均
Step 4: BN 重校准 — 用随机 20k 全量样本跑 forward，更新 BN 统计量
```

输出文件：`outputs/merged_phase1.pt`

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample_counts` | 必填 | 4 个 GPU 各自的训练样本数（从日志获取） |
| `--recal_limit` | 20000 | BN 重校准样本数。设 `0` 用全量 88k |
| `--recal_batch_size` | 8 | BN 重校准的 batch size |
| `--skip_recalibrate` | — | 跳过 BN 重校准（不推荐） |

---

## 第七步：Phase 2 投影对齐（单卡）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_v5.py \
    --phase projector --modality all \
    --batch_id batch27 \
    --epochs 3 --batch_size 4 --num_workers 2 \
    --lr 1e-4 --wd 1e-4 \
    --init_from outputs/merged_phase1.pt \
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --log_dir outputs/phase2 --auto_resume \
    > outputs/phase2/train.log 2>&1 &
```

> 约 30 分钟完成。只训练 projector heads + modality embedding，encoder 冻结。

---

## 第八步：Phase 3 Qwen LoRA（单卡）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_v5.py \
    --phase qwen --modality all \
    --batch_id batch27 \
    --epochs 3 --batch_size 2 --num_workers 2 \
    --lr 5e-5 --wd 0.01 --lora_r 16 --lora_alpha 32 \
    --init_from outputs/phase2/latest_step.pt \
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --log_dir outputs/phase3 --auto_resume \
    > outputs/phase3/train.log 2>&1 &
```

---

## 完整流程图

```
generate_init.py (seed=42)
       │
       └──→ outputs/init_weights.pt
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
  GPU0      GPU1      GPU2      GPU3
batch00-06 batch07-13 batch14-20 batch21-27
    │         │         │         │
    ▼         ▼         ▼         ▼
 step.pt   step.pt   step.pt   step.pt
    └─────────┼─────────┴─────────┘
              │
    merge_encoders.py
    (Head对齐 + FedAvg + BN重算)
              │
              ▼
    outputs/merged_phase1.pt
              │
    Phase 2 projector (单卡, 30min)
              │
              ▼
    outputs/phase2/latest_step.pt
              │
    Phase 3 Qwen LoRA (单卡)
              │
              ▼
    outputs/phase3/best_model.pt
```

---

## 数学原理回顾

### 加权策略：两层分级

| 层级 | 组件 | 加权方式 | 风险 |
|------|------|---------|------|
| **组 A：线性安全层** | 所有 Conv3d、BN γβ、LayerNorm γβ、stage_projs、scale_emb | 按样本量加权平均 | 极低（数学等价于输出平均） |
| **组 B：语义敏感层** | StageAttention MHA（含 head 对齐）+ MLP | 对齐后按样本量加权平均 | 低-中（对齐后大幅降低） |
| **组 C：不参与平均** | BN running_mean/var、ContrasiveHead | 丢弃/重算 | — |

### 为什么不退化

1. **相同初始化（seed=42）**：线性模式连接的前提条件
2. **按 batch 切分（同 patient 完整）**：positive pair 完整，对比学习结构无损
3. **Head 对齐**：消除了 MultiheadAttention 的置换不变性风险
4. **BN 重校准**：用全量数据重新计算 running stats，避免子集偏差
5. **丢弃 ContrasiveHead**：Phase 2/3 不需要的辅助头不参与平均

### 核心数学

对全部卷积层，`Conv(W₁) ⊕ Conv(W₂) = Conv((W₁+W₂)/2)` 精确等价，因为卷积是线性操作。这保证了 CNN backbone 的 FedAvg 在数学上是无偏的。

---

## 注意事项

1. **样本量**：`--sample_counts` 必须从训练日志中获取实际数字，不要用估计值
2. **中断恢复**：`--auto_resume` 会让中断的训练从 `latest_step.pt` 恢复
3. **BN 重校准时间**：2 万样本约 10-15 分钟（取决于 NAS I/O），全量 88k 约 1-2 小时
4. **显存**：Phase 1 每卡约 10-12GB，Phase 2 约 8GB，Phase 3 约 30GB
5. **Head 对齐**：24 种排列枚举 < 1ms，对每个模态各做一次
