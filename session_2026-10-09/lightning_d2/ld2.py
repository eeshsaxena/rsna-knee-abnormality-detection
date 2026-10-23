# ld2.py: BOX-side helper for the Lightning D2 run (TTR224 frac75 seed 1044, fold 3; coordinator option B 10-09).
# 10-09 19:52 IST: GPU jobs are refused on this free account until Eesh adds a verified payment method (LIGHTNING_D2.md).
# Run with ~/lightenv/bin/python ~/lightning_d2/ld2.py <cmd> ...  The Lightning credentials stay in
# ~/.lightning/credentials.json (read by the SDK only; never printed, copied or moved).
#   balance                       credit balance (user + teamspace) and spend
#   studios                       every Studio + job in the teamspace with status and machine
#   exec "CMD"                    run CMD in the Studio (LD2_STUDIO, default rsna-d2-nb; must be running)
#   bg TAG "CMD"                  start CMD detached in the Studio (setsid nohup), log ~/rsna_d2/logs/TAG.log there
#   upload LOCAL REMOTE           upload one file to the Studio (works while it is stopped: goes to its drive)
#   machine FAMILY N              the Nebius machine used by start_cpu / submit (CPU 4 | L40S 1)
#   start_cpu | stop              start the Studio on Nebius CPU-4 (0.40/h) | stop it
#   submit KIND NAME CAP_S [PRE]  KIND dry (Nebius CPU-4) or gpu (Nebius L40S on-demand 2.14/h): Job from the Studio
#                                 snapshot running kit/train_job.sh {dry|train} 3 CAP_S PRE
#   job [NAME] | joblogs [NAME] [N] | jobstop [NAME] | artifacts [NAME]
#   download NAME REMOTE LOCAL    download one teamspace-drive file
import os, sys, json, time, re

os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")
import lightning_sdk as L

USER, TS = "eeshsaxena", "default-project"
# The free CPU Studio (scratch-studio-devbox, lightning-public-prod = AWS) did the zero-credit env build + CPU parity.
# Studio jobs must run in their Studio's cloud account, and the only allowed GPU (Nebius L40S on-demand, $2.14/h) is in
# lightning-nebius-prod, so the run uses the paid Studio rsna-d2-nb there (CPU only for setup, then stopped).
STUDIO = os.environ.get("LD2_STUDIO", "rsna-d2-nb")
CLOUD = "lightning-nebius-prod"
JOB = os.environ.get("LD2_JOB", "rsna-d2-f3-s1044")
RED = re.compile(r"[A-Za-z0-9_\-]{40,}")


def red(s):
    return RED.sub("<redacted>", str(s))


def studio():
    return L.Studio(name=STUDIO, teamspace=TS, user=USER)


def teamspace():
    return L.Teamspace(name=TS, user=USER)


def balance():
    from lightning_sdk.lightning_cloud.rest_client import LightningClient
    c = LightningClient(retry=True)
    ub = c.billing_service_get_user_balance()
    pb = c.billing_service_get_project_balance(project_id=teamspace().id)
    return dict(user_balance=ub.balance, user_total_spent=ub.total_spent, teamspace_balance=pb.balance,
                n_transactions=len(ub.transactions or []))


def nb_machine(family, n):
    """The on-demand Nebius machine: family CPU with n vCPU, or family L40S with n GPUs."""
    ms = teamspace().list_machines(cloud_account=CLOUD)
    if family == "CPU":
        ok = [m for m in ms if m.family == "CPU" and m.accelerator_count == n]
    else:
        ok = [m for m in ms if m.family == family and m.accelerator_count == n]
    assert len(ok) == 1, [str(m) for m in ok]
    return ok[0]


def get_job(name=None):
    return L.Job(name=name or JOB, teamspace=TS, user=USER)


def job_info(name=None):
    j = get_job(name)
    d = dict(name=j.name, status=str(j.status), machine=str(j.machine))
    for k in ("total_cost", "started_at", "stopped_at", "link"):
        try:
            d[k] = str(getattr(j, k))
        except Exception as e:
            d[k] = "ERR " + red(repr(e))[:120]
    return d


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__ or open(__file__).read().split("import os")[0]); return 2
    cmd = a[0]
    if cmd == "balance":
        print(json.dumps(balance()))
    elif cmd == "studios":
        t = teamspace()
        for s in t.studios:
            print("STUDIO", s.name, str(s.status), str(s.machine), "interruptible", s.interruptible)
        for j in t.jobs:
            try:
                print("JOB", j.name, str(j.status), str(j.machine), "cost", j.total_cost)
            except Exception as e:
                print("JOB", j.name, "ERR", red(repr(e))[:200])
    elif cmd == "exec":
        s = studio()
        out, rc = s.run_with_exit_code(a[1])
        print(red(out)); print("RC", rc); return rc
    elif cmd == "bg":
        tag, c = a[1], a[2]
        s = studio()
        line = ("mkdir -p ~/rsna_d2/logs && cd ~ && (setsid nohup bash -c %s > ~/rsna_d2/logs/%s.log 2>&1 < /dev/null & "
                "echo $! > ~/rsna_d2/logs/%s.pid) && sleep 1 && echo started pid $(cat ~/rsna_d2/logs/%s.pid)"
                % (json.dumps(c), tag, tag, tag))
        out, rc = s.run_with_exit_code(line)
        print(red(out)); print("RC", rc); return rc
    elif cmd == "upload":
        s = studio()
        t0 = time.time(); sz = os.path.getsize(a[1])
        s.upload_file(a[1], a[2], progress_bar=False)
        dt = time.time() - t0
        print("UPLOADED %s -> %s: %.2f GB in %.0fs (%.1f MB/s)" % (a[1], a[2], sz / 1e9, dt, sz / 1e6 / max(dt, 1)))
    elif cmd == "machine":   # machine FAMILY VCPU_OR_GPUCOUNT: the Nebius machine object used by start/submit
        print(nb_machine(a[1], int(a[2])))
    elif cmd == "start_cpu":
        s = studio()
        m = nb_machine("CPU", 4)
        print("starting", s.name, "on", m, "cost", m.cost, "/h"); s.start(machine=m, interruptible=False)
        print("STUDIO", s.name, str(s.status), str(s.machine))
    elif cmd == "stop":
        s = studio(); s.stop(); print("STUDIO", s.name, str(s.status), str(s.machine))
    elif cmd == "submit":   # submit KIND(dry|gpu) NAME CAP_S [PRE]
        kind, name, cap = a[1], a[2], int(a[3]); pre = float(a[4]) if len(a) > 4 else 0.40
        m = nb_machine("CPU", 4) if kind == "dry" else nb_machine("L40S", 1)
        mode = "dry" if kind == "dry" else "train"
        command = "bash /teamspace/studios/this_studio/rsna_d2/kit/train_job.sh %s 3 %d %.3f" % (mode, cap, pre)
        print("SUBMIT job %s on %s (%s, $%s/h on-demand) studio %s: %s" % (name, m.name, m.provider, m.cost, STUDIO, command))
        j = L.Job.run(name=name, machine=m, command=command, studio=studio(), teamspace=TS, user=USER,
                      env={"LD2_JOB": name}, interruptible=False, max_run_attempts=1, reuse_snapshot=True)
        print("SUBMITTED", json.dumps(job_info(j.name)))
    elif cmd == "job":
        print(json.dumps(job_info(a[1] if len(a) > 1 else None)))
    elif cmd == "joblogs":
        j = get_job(a[1] if len(a) > 1 and a[1] != "-" else None)
        n = int(a[2]) if len(a) > 2 else 40
        try:
            lg = j.logs
        except Exception as e:
            lg = "ERR " + repr(e)
        print("\n".join(red(lg).splitlines()[-n:]))
    elif cmd == "jobstop":
        j = get_job(a[1] if len(a) > 1 else None)
        j.stop(); print("STOP requested:", json.dumps(job_info(j.name)))
    elif cmd == "artifacts":
        j = get_job(a[1] if len(a) > 1 else None)
        for e in j.list_artifacts(recursive=True):
            print(red(e))
    elif cmd == "download":
        j = get_job(a[1])
        L.Teamspace(name=TS, user=USER).download_file(a[2], a[3])
        print("DOWNLOADED", a[2], "->", a[3], os.path.getsize(a[3]))
    else:
        print("unknown command", cmd); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
