import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

base = "/mnt/nas1/disk07/public/shushangjiang /evaluation.v1"

files_to_read = ["README.md", "eval_local.py", "calc_scores.py", "extract_labels_keyword.py", "test_clinical_f1.py", "test_nlg.py", "test_diversity.py", "crg_score.py", "results.json"]

for f in files_to_read:
    print(f"\n{'='*80}")
    print(f"FILE: {f}")
    print('='*80)
    s, o, e = c.exec_command(f"cat '{base}/{f}'")
    content = o.read().decode()
    if len(content) > 3000:
        content = content[:3000] + "\n\n... [TRUNCATED] ..."
    print(content)

c.close()
