import json, os, subprocess, sys, time
H = os.path.expanduser('~/guards1020'); R = json.load(open(H + '/r6_def/run.json'))
cmd = R['cmd']; i = cmd.index('--deadline'); cmd[i + 1] = str(time.time() + 3600)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', **R['env'])
t = time.time()
with open(H + '/r6_def/worker0/log.txt', 'w') as log:
    rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print('R6_DEF_DONE rc', rc, 'seconds %.0f' % (time.time() - t), flush=True)
