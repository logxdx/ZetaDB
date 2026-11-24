"""Verify a hard-killed parent takes llama-server with it."""
import os, subprocess, sys, time
from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

CHILD = f'''
import sys, time
sys.path.insert(0, {ROOT!r})
from pathlib import Path
from server.models.llama_server import LlamaServer
# Any GGUF will do - this tests llama_server's lifecycle, not a model.
d = Path({ROOT!r}) / "data" / "models"
m = next(iter(sorted(d.rglob("*.gguf"))))
s = LlamaServer(m, ctx=512, ngl=0, embedding=True, cpu_only=True)
s.start()
print(s.proc.pid, flush=True)
time.sleep(600)
'''
parent = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True)
srv_pid = int(parent.stdout.readline().strip())
print(f"parent={parent.pid} llama-server={srv_pid}")
alive = lambda p: os.path.exists(f"/proc/{p}")
print(f"  before kill: llama-server alive={alive(srv_pid)}")
os.kill(parent.pid, 9)          # hard kill: atexit will NOT run
for _ in range(40):
    if not alive(srv_pid): break
    time.sleep(0.25)
ok = not alive(srv_pid)
print(f"  after SIGKILL of parent: llama-server alive={alive(srv_pid)}")
print("RESULT:", "PASS - no orphan" if ok else "FAIL - orphan survived")
if not ok: os.kill(srv_pid, 9)
