# fbcommon.py -- shared helpers for the fast_build FB-1 checks (parity, speed). Box-rule GPU safety lives here:
#   gpu_gate(need_mib): wait (max 15 min) until nvidia-smi reports >= need_mib free, else exit 3; call it BEFORE any CUDA
#   allocation, then cap(): torch.cuda.set_per_process_memory_fraction so this process can never take the trainer's room.
import os, sys, time, json, subprocess, threading

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def smi(q):
    out = subprocess.run(['nvidia-smi', f'--query-{q}', '--format=csv,noheader,nounits'], capture_output=True, text=True,
                         timeout=30).stdout
    return [[c.strip() for c in l.split(',')] for l in out.strip().splitlines() if l.strip()]


def gpu_gate(need_mib, max_wait_s=900):
    t0 = time.time()
    while True:
        free = int(smi('gpu=memory.free')[0][0])
        if free >= need_mib:
            print(f'[gate] GPU free {free} MiB >= need {need_mib}', flush=True)
            return free
        if time.time() - t0 > max_wait_s:
            print(f'[gate] GPU free {free} MiB < need {need_mib} after {max_wait_s}s: refusing to start', flush=True)
            sys.exit(3)
        time.sleep(15)


def cap(cap_gib):
    import torch
    tot = torch.cuda.get_device_properties(0).total_memory
    frac = min(0.95, cap_gib * 2 ** 30 / tot)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f'[cap] per-process CUDA memory fraction {frac:.3f} = {cap_gib} GiB of {tot / 2 ** 30:.2f} GiB', flush=True)
    return frac


class SmiSampler(threading.Thread):
    """Samples nvidia-smi every `dt` s: max total GPU used, max used by this pid, max used by every other pid."""
    def __init__(self, dt=1.0):
        super().__init__(daemon=True)
        self.dt, self.stop_ev = dt, threading.Event()
        self.max_total = 0; self.max_self = 0; self.max_other = {}; self.n = 0
        self.pid = os.getpid()

    def run(self):
        while not self.stop_ev.is_set():
            try:
                self.max_total = max(self.max_total, int(smi('gpu=memory.used')[0][0]))
                for pid, mem in smi('compute-apps=pid,used_memory'):
                    pid, mem = int(pid), int(mem)
                    if pid == self.pid:
                        self.max_self = max(self.max_self, mem)
                    else:
                        self.max_other[pid] = max(self.max_other.get(pid, 0), mem)
                self.n += 1
            except Exception:
                pass
            self.stop_ev.wait(self.dt)

    def reset(self):
        self.max_total = 0; self.max_self = 0; self.max_other = {}; self.n = 0

    def snap(self):
        return dict(smi_max_total_MiB=self.max_total, smi_max_self_MiB=self.max_self,
                    smi_max_other_MiB={str(k): v for k, v in self.max_other.items()}, smi_samples=self.n)
