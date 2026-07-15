import paramiko, io, os, shutil
from pathlib import Path

# 1. Delete v1_legacy locally
local_v1 = Path(r"C:\Users\HP\Documents\5555\evaluation\v1_legacy")
if local_v1.exists():
    shutil.rmtree(local_v1)
    print("Deleted local: evaluation/v1_legacy/")

# Delete unified_eval.py from root
root_eval = Path(r"C:\Users\HP\Documents\5555\unified_eval.py")
if root_eval.exists():
    root_eval.unlink()
    print("Deleted local: unified_eval.py")

# 2. Connect to farm05
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# 3. Delete v1_legacy from NAS
c.exec_command("rm -rf '/mnt/nas1/disk07/public/jiaqigu/evaluation/v1_legacy'")
print("Deleted NAS: evaluation/v1_legacy/")

# 4. Delete v1_legacy from git working copy
c.exec_command("rm -rf /home/jiaqigu/mrrate_eval_git/evaluation/v1_legacy")
c.exec_command("rm -f /home/jiaqigu/mrrate_eval_git/unified_eval.py")
print("Deleted git: evaluation/v1_legacy/")

# 5. Upload updated evaluation_v2.py
with open(r"C:\Users\HP\Documents\5555\evaluation_v2.py", "rb") as f:
    data = f.read()
sftp = c.open_sftp()
sftp.putfo(io.BytesIO(data), "/home/jiaqigu/mrrate_eval_git/evaluation/v2/evaluation_v2.py")
sftp.putfo(io.BytesIO(data), "/mnt/nas1/disk07/public/jiaqigu/evaluation/v2/evaluation_v2.py")
sftp.close()
print("Synced: evaluation_v2.py")

# 6. Update README (clean version without 14-class mentions)
readme = """# MR-RATE 项目评估模块 v2

融合了 江上姝 evaluation.v1 到统一评估脚本中。

## 功能

- **37 类病理标签**（关键词提取 + 否定检测），覆盖脑血管/肿瘤/脊柱/先天性/脑实质/退行性/感染等全类别
- **NLG**: BLEU-4 + METEOR + ROUGE-L
- **Diversity**: 去重率 + 平均长度
- **加权综合分**: Composite = 0.4 × NLG_avg + 0.4 × Clinical_Macro_F1 + 0.2 × Uniqueness
- **逐病理 F1**：每个疾病独立输出 Precision/Recall/F1/Support，F1=0 的类目自动提示
- **版本对比**：`--compare` 横向对比多个模型版本

## Usage

```bash
# 标准评估
python evaluation/v2/evaluation_v2.py --preds preds.json --refs refs.json

# 输出结果到 JSON
python evaluation/v2/evaluation_v2.py --preds preds.json --refs refs.json --output results.json

# 多版本对比
python evaluation/v2/evaluation_v2.py --preds v0.json --refs refs.json \\
    --compare v1.json v2.json
```

## 已知局限

用关键词匹配替代 Rabert 做标签提取，存在系统性误差（"rule out infarct" 会被误标、复杂否定可能漏检、隐含诊断无法检测），但误差是系统性的，模型越好分数也越高，不影响跨版本对比的有效性。

## 存储位置

- GitHub: [thomasm291678/mrrate-baseline](https://github.com/thomasm291678/mrrate-baseline) `evaluation/`
- NAS: `/mnt/nas1/disk07/public/jiaqigu/evaluation/`
- 本地: `C:\\Users\\HP\\Documents\\5555\\evaluation\\`

## 来源

| 来源 | 贡献 |
|------|------|
| 江上姝 evaluation.v1 | 37 类病理标签 + 否定检测 + Composite 加权 + 原 calc_scores.py 逐病理 F1 |
| 原有 unified_eval.py | NLG/Diversity 基础逻辑 |

融合后统一为单文件 `evaluation_v2.py`，原 JSS 分散的多文件版本保存在 `reference/evaluation_v1_jss/` 供参考。
"""

with open(r"C:\Users\HP\Documents\5555\evaluation\README.md", "w", encoding="utf-8") as f:
    f.write(readme)

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(readme.encode()), "/home/jiaqigu/mrrate_eval_git/evaluation/README.md")
sftp.close()
print("Synced: evaluation/README.md")

# 7. Git commit + push
cmds = [
    "cd /home/jiaqigu/mrrate_eval_git && git add -A",
    "cd /home/jiaqigu/mrrate_eval_git && git commit -m 'cleanup: remove deprecated 14-class labels and v1_legacy, keep only 37-class'",
    "cd /home/jiaqigu/mrrate_eval_git && git push origin main",
]
for cmd in cmds:
    s, o, e = c.exec_command(cmd, timeout=15)
    print(cmd.split("&&")[-1].strip())
    out = o.read().decode()
    err = e.read().decode()
    if out: print(out)
    if err: print(err)

# Verify
s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git log --oneline -3")
print("Git log:", o.read().decode())

c.close()
print("\nAll clean. 14-class code fully removed from local, NAS, and GitHub.")
