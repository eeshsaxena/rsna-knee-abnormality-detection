#!/usr/bin/env python
# kc_serve.py (10-15, Claude subagent, kcstage): ONE KC serve process per T4 for the S6F+KC kernel (build_v224s6fkc.py).
# It runs kc_worker.py's OWN __main__ block (kc_worker.py md5 4af60f6f, the frozen A6 serve path, KC-R2 CLI), byte for byte,
# inside the kc_worker module namespace, with ONE substitution:
#   kc_worker.study_inputs takes the knee side, the per-slot geometry and the serve mode from the SIDE TABLE that the KC driver
#   built with the side kernel's own code (kcrop/kernel_kc_side/kc_side_kernel.py sha256 2979c191, its study() on the test split),
#   with exactly the derivation the training store used (kc_build_store.py cdc7a244 row_geom, MAIN_SCHEMA, --geom table):
#     side, rule = kc_common.side_rule(mean_x, tag)            (asserted == the table's side / rule)
#     graw[b]    = s<k>_sign * s<k>_mag when s<k>_sign != 0 else NaN, k = sagfs 0, sagpd 1, corfs 2, ax 4 (corpd: NaN)
#     geom       = graw, NaN -> kc_common.GEOM_DEFAULT;  flips = kc_common.block_flips(side, geom) (asserted == table s<k>_flip)
#     weak       = rule != 'mean_x' (asserted == table weak); gok = kc_common.geom_ok(graw, present KC blocks)
#     serve      = 'both' if weak or not gok or table serve == 'both' (any disagreement of the two geometry rules -> both canons)
#   (kc_worker's own header_side + 3-header slot_geometry are NOT used: A6 G3 "the side matches the header rule" = the side kernel.)
# Unchanged (kc_worker / kc_common): the verbatim d96 reader (build_study, sha 30aa72fd), kc_cut / canon, views_to_tensor,
# load_members (contract hash, mirror 0, model_sha == sha(fleet_model_ttr + fleet_model)), fp16 autocast on cuda:N, ONE pass, mean of
# member sigmoids, the both-canon mean, the fork decode pool with its bounded ordered window, NaN + kc_err per failed study.
# Timing hooks (receipts only, no effect on outputs): t_dec per study (in the decode worker) and t_gpu per study (serve_probs wall).
#   usage: python kc_serve.py --code /tmp/kc/code --side kc_side_test.csv --timing part0_timing.json -- <kc_worker.py args>
import os, sys, json, time, hashlib, textwrap

PINS_MD5 = {'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18', 'kc_common.py': '45255ae0e361ef787b290d7510554a9d',
            'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
D96_SHA256 = '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40'
SLOTK = {'sagfs': 0, 'sagpd': 1, 'corfs': 2, 'ax': 4}                    # = kc_build_store.SLOTK
MAIN_MARK = "\nif __name__ == '__main__':\n"


def check_pins(code):
    for f, h in PINS_MD5.items():
        got = hashlib.md5(open(os.path.join(code, f), 'rb').read()).hexdigest()
        if got != h:
            raise AssertionError('frozen KC file %s md5 %s != %s' % (f, got, h))
    got = hashlib.sha256(open(os.path.join(code, 'd96_reader_src.py'), 'rb').read()).hexdigest()
    if got != D96_SHA256:
        raise AssertionError('d96 reader sha %s != %s' % (got, D96_SHA256))


def load_table(path):
    import pandas as pd
    t = pd.read_csv(path, dtype={'StudyInstanceUID': str, 'tag': str})
    if t['StudyInstanceUID'].duplicated().any():
        raise ValueError('side table has duplicate studies')
    return {r['StudyInstanceUID']: r for r in t.to_dict('records')}


def table_geom(r, kc):
    """= kc_build_store.row_geom for the side kernel's schema (MAIN_SCHEMA, --geom table)."""
    import numpy as np, pandas as pd
    raw = {}
    for b in kc.GEOM_DEFAULT:
        k = SLOTK.get(b)
        sg, mg = (r.get(f's{k}_sign'), r.get(f's{k}_mag')) if k is not None else (None, None)
        ok = k is not None and pd.notna(sg) and pd.notna(mg) and float(sg) != 0
        raw[b] = float(sg) * float(mg) if ok else float('nan')
    return raw, {b: (raw[b] if np.isfinite(raw[b]) else kc.GEOM_DEFAULT[b]) for b in kc.GEOM_DEFAULT}


def side_fields(r, kc):
    """side table row -> (side, rule, graw, geom, flips, weak) with the kc_build_store cross-checks (raise on any drift)."""
    import pandas as pd
    se = r.get('side_err')
    if isinstance(se, str) and se:
        raise ValueError('side pass failed for this study: ' + se[:120])
    mean_x = r.get('mean_x')
    tag = r.get('tag') if isinstance(r.get('tag'), str) else ''
    side, rule = kc.side_rule(mean_x, tag)
    if (side, rule) != (str(r['side']), str(r['rule'])):
        raise ValueError('side rule drift %s vs table %s' % ((side, rule), (r['side'], r['rule'])))
    graw, geom = table_geom(r, kc)
    flips = kc.block_flips(side, geom)
    for b, k in SLOTK.items():
        if pd.notna(r.get(f's{k}_flip')) and bool(flips[b]) != bool(int(r[f's{k}_flip'])):
            raise ValueError('flip %s %s != table s%d_flip %s' % (b, flips[b], k, r[f's{k}_flip']))
    weak = rule != 'mean_x'
    if int(weak) != int(r['weak']):
        raise ValueError('weak %d != table %s' % (int(weak), r['weak']))
    return side, rule, graw, geom, flips, weak


def make_study_inputs(table, kc):
    """kc_worker.study_inputs (same signature, same output keys) with the side table as the side / geometry / serve source."""
    def study_inputs(sid, rows, sdir, R, both=None):
        t0 = time.perf_counter()
        r = table[sid]                                                  # KeyError -> kc_worker._decode: NaN row + kc_err
        side, rule, graw, geom, flips, weak = side_fields(r, kc)
        rd = R['_make_reader']()
        vol, _mask = R['build_study'](sid, {sid: rows}, sdir, rd)
        pres = kc.present_blocks(vol)
        gok = kc.geom_ok(graw, [b for b in ('sagfs', 'sagpd', 'corfs', 'ax') if pres[b]])
        tbl_both = str(r.get('serve')) == 'both'
        out = dict(sid=sid, vol=vol, mean_x=r.get('mean_x'), tag=r.get('tag') if isinstance(r.get('tag'), str) else '', side=side,
                   rule=rule, weak=int(weak), gok=int(gok), geom_raw=graw, geom=geom,
                   serve='both' if (weak or not gok or tbl_both) else 'single',
                   tbl_geo_ok=int(r['geo_ok']), gok_mismatch=int(bool(gok) != bool(int(r['geo_ok']))))
        out['views'] = kc.kc_cut(vol, side, geom)
        if both if both is not None else out['serve'] == 'both':
            out['views_L'] = kc.kc_cut(vol, 'L', geom)
            out['views_R'] = kc.kc_cut(vol, 'R', geom)
        out['t_dec'] = round(time.perf_counter() - t0, 4)
        return out
    return study_inputs


def main():
    i = sys.argv.index('--')
    own, wargs = sys.argv[1:i], sys.argv[i + 1:]
    o = dict(zip(own[::2], own[1::2]))
    code, side_csv, timing = o['--code'], o['--side'], o['--timing']
    check_pins(code)
    sys.path.insert(0, code)
    import kc_worker as kw                                              # defs only (its __main__ block runs below)
    import kc_common as kc
    assert os.path.dirname(os.path.abspath(kw.__file__)) == os.path.abspath(code), kw.__file__
    table = load_table(side_csv)
    kw.study_inputs = make_study_inputs(table, kc)                      # THE substitution (module global, before the fork)
    tim = []
    _orig_serve = kw.serve_probs

    def serve_probs(ms, si, device):                                    # timing hook only
        t0 = time.perf_counter()
        p = _orig_serve(ms, si, device)                                 # returns numpy (.cpu()): already synchronised
        tim.append([si['sid'], si.get('t_dec'), round(time.perf_counter() - t0, 4), si.get('serve'), si.get('gok_mismatch')])
        return p
    kw.serve_probs = serve_probs
    src = open(os.path.join(code, 'kc_worker.py'), encoding='utf-8').read()
    assert src.count(MAIN_MARK) == 1, 'kc_worker.py __main__ marker'
    main_src = textwrap.dedent(src.split(MAIN_MARK, 1)[1])
    sys.argv = [os.path.join(code, 'kc_worker.py')] + wargs
    t0 = time.time()
    exec(compile(main_src, os.path.join(code, 'kc_worker.py'), 'exec'), kw.__dict__)
    wall = time.time() - t0
    rec = {'pid': os.getpid(), 'argv': wargs, 'wall_s': round(wall, 2), 'n_timed': len(tim), 'rows': tim,
           'side_table': side_csv, 'side_table_sha256': hashlib.sha256(open(side_csv, 'rb').read()).hexdigest(),
           'kc_worker_md5': PINS_MD5['kc_worker.py'], 'substitution': 'kc_worker.study_inputs <- side table (kc_serve.make_study_inputs)'}
    try:
        import torch
        rec['cuda'] = bool(torch.cuda.is_available())
        rec['torch'] = torch.__version__
        dev = kw.__dict__.get('A').device if kw.__dict__.get('A') is not None else None
        if dev and str(dev).startswith('cuda'):
            rec['gpu_name'] = torch.cuda.get_device_name(torch.device(dev))
            rec['max_mem_alloc_mb'] = round(torch.cuda.max_memory_allocated(torch.device(dev)) / 2 ** 20, 1)
    except Exception as e:
        rec['torch_info_error'] = type(e).__name__
    with open(timing + '.tmp', 'w') as f:
        json.dump(rec, f)
    os.replace(timing + '.tmp', timing)


if __name__ == '__main__':
    main()
