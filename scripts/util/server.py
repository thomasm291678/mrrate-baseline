import http.server, json, os, re, math, time
import paramiko

PORT = 8899
DIR = os.path.dirname(__file__)

HOST = "10.176.60.72"
USER = "jiaqigu"
PWD = "lijia7272"
LOG_PATH = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_cumul.log"


def fetch_stats():
    stats = {
        "alive": False, "losses": [], "gpus": [], "current_loss": 0,
        "current_epoch": 0, "current_step": 0, "total_epochs": 10,
        "num_processes": 0, "mem_gb": 0, "x_max": 450,
        "l_min": 1.0, "l_max": 2.5, "update_time": "",
        "error": None
    }
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=PWD, timeout=10)

        def run(cmd):
            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
            return stdout.read().decode(errors='ignore').strip()

        procs = run("pgrep -c -f 'python.*train_report_gen' 2>/dev/null || echo 0")
        stats["num_processes"] = int(procs) if procs.isdigit() else 0
        stats["alive"] = stats["num_processes"] > 0

        if stats["alive"]:
            gpu_mem = run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep '^0,' | cut -d',' -f2 | sed 's/ MiB//;s/ GiB//;s/ //g'")
            if gpu_mem and gpu_mem.isdigit():
                stats["mem_gb"] = round(int(gpu_mem) / 1024, 1)

        gpu_samples = []
        for _ in range(2):
            gpu_raw = run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null")
            gpus = []
            for line in gpu_raw.strip().split('\n'):
                parts = [p.strip().replace(' %','').replace(' MiB','') for p in line.split(',')]
                if len(parts) >= 4 and parts[0].isdigit():
                    gpus.append({"index": int(parts[0]), "mem_used": parts[1],
                                 "mem_total": parts[2], "util": int(parts[3])})
            gpu_samples.append(gpus)
            time.sleep(0.5)

        if gpu_samples:
            merged = []
            for i in range(len(gpu_samples[0])):
                max_util = max(s[i]["util"] for s in gpu_samples if i < len(s))
                merged.append({**gpu_samples[0][i], "util": max_util})
            stats["gpus"] = merged

        log_raw = run("cat " + LOG_PATH)
        losses = []
        for line in log_raw.split('\n'):
            m = re.search(r'\[E(\d+)\s+S(\d+)\]\s+loss=([\d.]+)', line)
            if m:
                losses.append({"epoch": int(m.group(1)), "step": int(m.group(2)), "loss": round(float(m.group(3)), 4)})
        stats["losses"] = losses

        if losses:
            stats["current_loss"] = round(sum(l["loss"] for l in losses[-5:]) / min(5, len(losses)), 4)
            stats["raw_loss"] = losses[-1]["loss"]
            stats["current_epoch"] = losses[-1]["epoch"]
            stats["current_step"] = losses[-1]["step"]
            stats["x_max"] = max(450, losses[-1]["step"])
            vals = [l["loss"] for l in losses]
            lmin = 0
            lmax = math.ceil(max(vals) * 10) / 10 + 0.1
            if lmax < 1.5:
                lmax = 1.5
            stats["l_min"] = lmin
            stats["l_max"] = lmax

        ep_match = re.search(r'Epochs:\s*(\d+)', log_raw)
        if ep_match:
            stats["total_epochs"] = int(ep_match.group(1))

        train_match = re.search(r'Train:\s*(\d+)', log_raw)
        batch_match = re.search(r'Batch:\s*(\d+)', log_raw)
        if train_match and batch_match:
            total_samples = int(train_match.group(1))
            batch_size = int(batch_match.group(1))
            stats["steps_per_epoch"] = math.ceil(total_samples / (batch_size * 2))
        else:
            stats["steps_per_epoch"] = 22000  # fallback

        stats["update_time"] = time.strftime("%H:%M:%S")

        # Elapsed time: use training process start time
        pid = run("pgrep -f 'python.*train_report_gen' | head -1")
        if pid and losses:
            etime = run("ps -o etimes= -p " + pid + " 2>/dev/null")
            if etime and etime.strip().isdigit():
                elapsed = int(etime.strip())
                stats["elapsed_seconds"] = elapsed
                total_steps = stats["total_epochs"] * stats["steps_per_epoch"]
                completed_epoch_steps = (stats["current_epoch"] - 1) * stats["steps_per_epoch"]
                global_step = completed_epoch_steps + stats["current_step"]
                if global_step > 0 and elapsed > 10:
                    sps = global_step / elapsed
                    remaining_steps = total_steps - global_step
                    if sps > 0:
                        stats["eta_seconds"] = int(remaining_steps / sps)
                    stats["steps_per_second"] = round(sps, 2)

        ssh.close()
    except Exception as e:
        stats["error"] = str(e)
        stats["update_time"] = time.strftime("%H:%M:%S")

    return stats


class APIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def do_GET(self):
        if self.path == '/api/stats':
            stats = fetch_stats()
            body = json.dumps(stats).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, format, *args):
        if args and isinstance(args[0], str) and '/api/stats' not in args[0]:
            super().log_message(format, *args)


if __name__ == '__main__':
    server = http.server.HTTPServer(('0.0.0.0', PORT), APIHandler)
    print(f"Training monitor server running at http://localhost:{PORT}")
    server.serve_forever()
