# kc_regress_cmp.py -- shim regression helpers (box CPU). Modes:
#   args REF_DIR NEW_DIR : ckpt args of the raw best ckpt must be IDENTICAL (bar); weights / OOF diff reported (info)
#   kc   OUT_DIR VERDICT : the --kc run: ckpt args model_mod kc_ttr, mirror 0; verdict has 'SMOKE_TEST_ONLY hook_ok'
#   shim TARGET BACKUP   : TARGET == BACKUP + exactly the shim block (line diff), and ast.parse(TARGET) is fine
import sys, glob, ast, difflib, os
mode = sys.argv[1]
if mode == 'shim':
    tgt, bak = sys.argv[2:4]
    a, b = open(bak).read().splitlines(), open(tgt).read().splitlines()
    ast.parse('\n'.join(b))
    add = [l[1:] for l in difflib.unified_diff(a, b, lineterm='', n=0) if l.startswith('+') and not l.startswith('+++')]
    rem = [l for l in difflib.unified_diff(a, b, lineterm='', n=0) if l.startswith('-') and not l.startswith('---')]
    blk = open(os.path.expanduser('~/kc/code/shim/shim_block.txt')).read().splitlines()
    ok = add == blk and not rem
    print(f"(a) {'PASS' if ok else 'FAIL'} ast ok | added {len(add)} lines == shim block: {add == blk} | removed {len(rem)}")
    sys.exit(0 if ok else 1)
import torch
def raw(d):
    c = sorted(glob.glob(f'{d}/f*_s*_raw_*_best.pt'))
    assert len(c) == 1, (d, c)
    return torch.load(c[0], map_location='cpu', weights_only=False)
if mode == 'args':
    r, n = raw(sys.argv[2]), raw(sys.argv[3])
    ka, kb = r['args'], n['args']
    diff = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
    wd = max(float((r['model'][k].float() - n['model'][k].float()).abs().max()) for k in r['model'] if r['model'][k].is_floating_point())
    ok = not diff
    print(f"(b) {'PASS' if ok else 'FAIL'} ckpt args identical: {ok} ({len(ka)} keys; differing: {diff}) | info: cv {r['cv']:.6f} vs {n['cv']:.6f}, "
          f"max |dW| {wd:.3g}")
    sys.exit(0 if ok else 1)
if mode == 'kc':
    c = raw(sys.argv[2]); a = c['args']
    v = open(sys.argv[3]).read()
    ok = a.get('model_mod') == 'kc_ttr' and a.get('mirror') == 0 and a.get('mirror_aug') == 0.0 and 'SMOKE_TEST_ONLY hook_ok' in v
    print(f"(c) {'PASS' if ok else 'FAIL'} --kc via the shim: model_mod {a.get('model_mod')} mirror {a.get('mirror')} n_views {a.get('n_views')} "
          f"store {a.get('store')} | verdict: {v.splitlines()[0][:120]}")
    sys.exit(0 if ok else 1)
