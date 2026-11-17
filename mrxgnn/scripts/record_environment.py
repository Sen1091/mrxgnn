"""Record the software and hardware used for the runs (reported in the manuscript automatically).
  python scripts/record_environment.py --out results/environment.json
"""
import argparse, json, os, platform
import sklearn, torch

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/environment.json"); a = ap.parse_args()
if torch.cuda.is_available():
    hw = f"an NVIDIA {torch.cuda.get_device_name(0)} GPU"
else:
    cpu = platform.processor() or platform.machine()
    hw = f"a CPU ({cpu}, {os.cpu_count()} logical cores; no GPU)"
env = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
       "sklearn": sklearn.__version__, "hardware": hw, "os": f"{platform.system()} {platform.release()}"}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(env, open(a.out, "w"), indent=1)
print(env)
