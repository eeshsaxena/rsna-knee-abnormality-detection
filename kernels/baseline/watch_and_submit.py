#!/usr/bin/env python3
"""Poll the rsna-knee-baseline kernel; when COMPLETE, verify submission.csv and
submit it to the code competition. User-authorized ('submit notebook')."""
import subprocess, re, time, pathlib, sys

KID = "eeshsaxena/rsna-knee-baseline"
COMP = "rsna-knee-abnormality-detection"
VER = "1"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out"
EXPECT = "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
MSG = "plumbing baseline: constant 0.5 x12 (validate code-submission pipeline)"


def cli(args):
    p = subprocess.run(["kaggle"] + args, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    out = "\n".join(l for l in out.splitlines()
                    if "RequestsDependency" not in l and "warnings.warn" not in l)
    return p.returncode, out


def status():
    _, o = cli(["kernels", "status", KID])
    m = re.search(r'KernelWorkerStatus\.(\w+)', o)
    return m.group(1) if m else ("ABSENT" if "404" in o else "UNKNOWN")


def main():
    for i in range(120):  # ~40 min cap at 20s
        st = status()
        print(f"[{i}] {st}", flush=True)
        if st == "COMPLETE":
            break
        if st in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            print(f"KERNEL {st} -> abort, no submit"); return
        time.sleep(20)
    else:
        print("timed out waiting for COMPLETE -> no submit"); return

    OUT.mkdir(exist_ok=True)
    rc, o = cli(["kernels", "output", KID, "-p", str(OUT)])
    csv = OUT / "submission.csv"
    if not csv.exists():
        print(f"NO submission.csv in output -> abort. {o[:200]}"); return
    header = csv.read_text(encoding="utf-8").splitlines()[0].strip()
    rows = sum(1 for _ in open(csv, encoding="utf-8")) - 1
    if header != EXPECT:
        print(f"HEADER MISMATCH -> abort.\n got: {header}\n exp: {EXPECT}"); return
    if rows < 1:
        print(f"only {rows} data rows -> abort"); return
    print(f"CSV OK: {rows} rows, header matches. Submitting...")
    rc, o = cli(["competitions", "submit", COMP, "-k", KID, "-v", VER,
                 "-f", "submission.csv", "-m", MSG])
    print(o.strip())
    if "Successfully submitted" in o or rc == 0:
        print("SUBMITTED OK")
    else:
        print(f"SUBMIT FAILED rc={rc}")


if __name__ == "__main__":
    main()
