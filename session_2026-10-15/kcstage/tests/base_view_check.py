# base_view_check.py (10-15, Claude subagent, kcstage): check (A) of v224s6fkc_check.py on the REAL base commit outputs.
# OUT = the real S6F v1 commit output (s6f/out_v224flips6f_v1) or the real S6F-J v1 commit output (jpeg/overlay/r2/out_v224flips6fj_v1)
# + KC files from the box e2e run (kc/, kc_setup_receipt.json, kc_receipt.json) + submission_s6f.csv := the base submission.csv, and the
# real base log + KC lines appended, renamed rsna-sub-<KC tag>.log. Only the (A) lines are judged here (the KC rows are 5 train studies,
# so (B)-(D) fail by construction): the base gate on the KC-removed view must PASS and the frozen base must equal the base commit.
# Negative: one base stage csv (raptor_raw.csv) changed by 0.01 -> the base-gate line FAILs.
#   py -3.12 -I tests/base_view_check.py   -> BASE_VIEW PASS n/m
import os, sys, json, glob, shutil, tempfile, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
SESS = os.path.dirname(KS)
PULL = os.path.join(HERE, 'pulled_e2e', 'clean', 'working')
GATE = os.path.join(KS, 'v224s6fkc_check.py')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def build(base_out, tag, tmp, mutate=None):
    out = os.path.join(tmp, tag)
    shutil.copytree(base_out, out)
    logs = [p for p in glob.glob(os.path.join(out, '*.log')) if 'cufile' not in p]
    E = json.load(open(logs[0], encoding='utf-8'))
    os.remove(logs[0])
    kc_lines = [l for l in open(os.path.join(HERE, 'pulled_e2e', 'clean.log'), encoding='utf-8').read().splitlines() if l.startswith('KC_')]
    t = E[-1]['time']
    E += [{'stream_name': 'stdout', 'time': t + 1 + i, 'data': l + '\n'} for i, l in enumerate(kc_lines)]
    json.dump(E, open(os.path.join(out, 'rsna-sub-%s.log' % tag), 'w', encoding='utf-8'))
    shutil.copytree(os.path.join(PULL, 'kc'), os.path.join(out, 'kc'))
    for f in ('kc_setup_receipt.json', 'kc_receipt.json'):
        shutil.copyfile(os.path.join(PULL, f), os.path.join(out, f))
    shutil.copyfile(os.path.join(out, 'submission.csv'), os.path.join(out, 'submission_s6f.csv'))
    if mutate:
        mutate(out)
    r = subprocess.run([sys.executable, '-I', GATE, out, '--build', os.path.join(KS, 'build_receipt_%s.json' % tag)],
                       capture_output=True, text=True, timeout=1200)
    lines = r.stdout.splitlines()
    g = [l for l in lines if 'passes the UNCHANGED base gate' in l]
    f = [l for l in lines if 'frozen base == the base kernel v1 commit submission' in l]
    return g, f, r.stdout


def bump(out):
    p = os.path.join(out, 'raptor_raw.csv')
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    d.iloc[0, 1] = d.iloc[0, 1] + 0.01
    d.to_csv(p, index=False)


tmp = tempfile.mkdtemp(prefix='kcbase_')
for name, base_out, tag in (('S6F', os.path.join(SESS, 's6f', 'out_v224flips6f_v1'), 'v224s6fkcstandin'),
                            ('S6F-J', os.path.join(SESS, 'jpeg', 'overlay', 'r2', 'out_v224flips6fj_v1'), 'v224s6fkcstandinj')):
    g, f, txt = build(base_out, tag, tmp)
    open(os.path.join(HERE, 'base_view_%s.txt' % tag), 'w', encoding='utf-8').write(txt)
    chk('%s: the KC-removed view passes the unchanged base gate' % name, len(g) == 1 and g[0].strip().startswith('OK'), g[:1])
    chk('%s: frozen base == the %s v1 commit submission' % (name, name), len(f) == 1 and f[0].strip().startswith('OK'), f[:1])
    shutil.rmtree(os.path.join(tmp, tag), ignore_errors=True)
    g, f, _ = build(base_out, tag, tmp, mutate=bump)
    chk('%s negative: raptor_raw.csv +0.01 -> the base-gate line FAILs' % name, len(g) == 1 and g[0].strip().startswith('FAIL'), [x[:160] for x in g[:1]])
    shutil.rmtree(os.path.join(tmp, tag), ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print('BASE_VIEW %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
