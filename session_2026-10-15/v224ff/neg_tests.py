# neg_tests.py CLEAN_OUT CONTROL_OUT: the V224-FF gate must PASS on the right output and FAIL on wrong / tampered ones.
# Prints one NEG line per case and NEG PASS / NEG FAIL. Tampering happens on temp copies; the fetched outputs are not touched.
import os, sys, shutil, subprocess, tempfile, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
GATE = os.path.join(HERE, 'v224ff_check.py')
REFS = ['--ref-flip', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1'),
        '--ref-exact', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1')]
clean, ctl = sys.argv[1], sys.argv[2]
ok_all = True


def run(name, out, args, want):
    global ok_all
    r = subprocess.run([sys.executable, GATE, out] + args + REFS, capture_output=True, text=True, encoding='utf-8')
    last = [l for l in r.stdout.splitlines() if l.startswith('V224FF_CHECK ')][-1:] or ['(no verdict)']
    fails = [l.strip()[:150] for l in r.stdout.splitlines() if l.startswith('  FAIL')][:2]
    good = (r.returncode == 0) == (want == 0)
    ok_all &= good
    print('NEG %-40s rc=%d want=%d %s | %s | first fails: %s' % (name, r.returncode, want, 'OK' if good else 'BAD', last[0], fails), flush=True)


def tampered(src, fn):
    d = tempfile.mkdtemp(prefix='v224ff_neg_')
    shutil.copytree(src, d, dirs_exist_ok=True)
    fn(d)
    return d


def bump(d, name, col, delta=0.01):
    p = os.path.join(d, name)
    s = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    s.loc[0, col] = min(1.0, float(s.loc[0, col]) + delta) if s.loc[0, col] < 0.99 else float(s.loc[0, col]) - delta
    s.to_csv(p, index=False)


def receipt_edit(d, fn):
    p = os.path.join(d, 'flip_ta_receipt.json')
    r = json.load(open(p)); fn(r); json.dump(r, open(p, 'w'))


run('clean as clean', clean, ['--wf', '0.20'], 0)
run('control as control', ctl, ['--wf', '0.20', '--control'], 0)
run('control output, clean gate', ctl, ['--wf', '0.20'], 1)
run('clean output, control gate', clean, ['--wf', '0.20', '--control'], 1)
run('clean, wrong w_f .25', clean, ['--wf', '0.25'], 1)
run('clean, wrong flip budget 5.5', clean, ['--wf', '0.20', '--flip-budget-h', '5.5'], 1)
for name, fn, mode in [
        ('clean, submission cell +.01', lambda d: bump(d, 'submission.csv', 'ACL'), []),
        ('clean, submission_V224 cell +.01', lambda d: bump(d, 'submission_V224.csv', 'Effusion'), []),
        ('clean, R7 ours_sub cell +.01', lambda d: bump(d, 'ours_sub.csv', 'Contusion'), []),
        ('clean, a worker marked switched', lambda d: receipt_edit(d, lambda r: r['workers']['worker0'].update(decision='switch')), []),
        ('control, R7 plain row replaced by flip row', lambda d: shutil.copyfile(os.path.join(REFS[1], 'ours_sub.csv'), os.path.join(d, 'ours_sub.csv')), ['--control']),
]:
    src = ctl if mode else clean
    d = tampered(src, fn)
    run(name, d, ['--wf', '0.20'] + mode, 1)
    shutil.rmtree(d, ignore_errors=True)
print('NEG %s' % ('PASS' if ok_all else 'FAIL'), flush=True)
sys.exit(0 if ok_all else 1)
