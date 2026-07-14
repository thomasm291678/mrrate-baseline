import re, sys
with open("/home/jiaqigu/mrrate_hidnet/eval_llm.py") as f:
    txt = f.read()
m = re.search(r'SYSTEM_PROMPT = """(.+?)"""', txt, re.DOTALL)
if m:
    prompt = m.group(1)
    path = "/mnt/nas1/disk07/public/jiaqigu/evaluation/llm_prompt_v2.md"
    with open(path, "w") as out:
        out.write(prompt)
    print(f"OK {len(prompt)} chars -> {path}")
else:
    print("FAIL")
    sys.exit(1)
