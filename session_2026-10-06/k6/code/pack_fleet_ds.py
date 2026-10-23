# pack_fleet_ds.py -- laptop side. Build the eeshsaxena/rsna-knee-cnxt-fleet dataset folder:
# fleet_common.py + fleet_model.py + fleet_worker.py (copied from THIS folder, i.e. the files the box
# trained with), the chosen members, fleet_manifest.json and parity_<sid>.npz (the trainer's decode-probe
# studies, from the store). Refuses members that do not match this fleet_common's CONTRACT, bf16/fp32
# members, and two members with the same fold/seed (raw AND ema of one run).
# usage: python pack_fleet_ds.py --out <dir> --store <store dir> [--title T] [--tag K1,K2,...] ckpt1.pt [ckpt2.pt ...]
# fleet2 patch: also ships fleet_model_f2.py (family 2) and checks every member's model_sha against the code being
# shipped, per family, exactly as the fleet2 worker will (a mismatch now fails on the laptop, not in a Kaggle rerun).
# bigjump patch (BIG_JUMP_PLAN B1/B4):
#   P-TTR ships fleet_model_ttr.py too and checks TTR members (args model_mod 'fleet_model_ttr') by their model_sha
#   P-TAG --tag K1,K2,... (comma list or repeated --tag; one entry per member, in member order; '=' keeps the key;
#         argparse reads a leading '-' as an option, so write --tag =,f0_s42v4 or --tag=-,f0_s42v4): re-key a
#         member, i.e. rename its file f{fold}_s{seed}_<rest> -> <KEY>_<rest> in the dataset. Every trainer run names its files
#         f{fold}_s{seed}_{raw|ema}_..._best.pt, so two runs that differ only in labels or recipe (f0_s42 mix50,
#         f0_s42_v4, f0_s42g, TTR f0 seed 42) carry the same member key and often the same file name; the worker
#         refuses a duplicate key and a plain copy would overwrite one file with the other. KEY must be the member's
#         own f{fold}_s{seed} plus an optional [A-Za-z0-9]+ suffix (f0_s42v4, f0_s42g, f0_s42ttr): the worker's
#         fold grouping (first '_' field) and f{fold}_s{seed} parsing stay valid. Only the file NAME changes (bytes
#         identical). The manifest records every member's provenance and re-key under 'sources' (the worker reads
#         only 'checkpoints'): original file name, run folder name, sha256[:16] of the bytes, model_mod, model_sha.
#         FIX-5: no absolute paths (they would ship local user names and directory layout inside the Kaggle dataset).
#   P-RUN two members whose ckpt args are identical (raw AND ema of one run, two epoch snapshots of one run, or the
#         same file twice) are refused even when re-keyed.
# K5 patch (scratchpad/k5): --tta M1,M2,... (comma list or repeated, one entry per member in member order, 'A' or 'AB')
#   writes sources[<staged name>]['tta'] = M for every member (read by the K5 worker, K5-TTA; and by the K5 cell gate).
#   Without --tta the manifest is exactly as before (no 'tta' key = 'A' = pass A only).
import os, re, sys, json, shutil, argparse, hashlib
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fleet_common as fc
SHIP = ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py', 'fleet_worker.py')
MODS = ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr')


def _model_sha(mod='fleet_model'):                     # == fleet_worker._model_sha (bigjump)
    import inspect, fleet_model as _fm
    src = chr(10).join(inspect.getsource(_fm).splitlines())
    if mod == 'fleet_model_f2':
        import fleet_model_f2 as _f2
        src = chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + src
    elif mod == 'fleet_model_ttr':
        import fleet_model_ttr as _tt
        src = chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _sha16(path):                                     # FIX-5 provenance: content hash, no local path
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--store', required=True)
    ap.add_argument('--id', default='eeshsaxena/rsna-knee-cnxt-fleet'); ap.add_argument('--title', default='rsna-knee-cnxt-fleet')
    ap.add_argument('--tag', action='append', default=None,
                    help="P-TAG: comma list, one new key per member in member order ('=' keeps it); repeatable")
    ap.add_argument('--tta', action='append', default=None,                                    # K5
                    help="K5: comma list, one TTA mode per member in member order: A (pass A) or AB (A + B)")
    ap.add_argument('members', nargs='+')
    A = ap.parse_args(argv)
    tags = [t.strip() for x in A.tag for t in x.split(',')] if A.tag else ['-'] * len(A.members)
    assert len(tags) == len(A.members), f'--tag needs one entry per member ({len(A.members)}), got {len(tags)}: {tags}'
    tta = [t.strip() for x in A.tta for t in x.split(',')] if A.tta else None                     # K5
    if tta is not None:
        assert len(tta) == len(A.members) and all(t in ('A', 'AB') for t in tta), \
            f'--tta needs one A or AB per member ({len(A.members)}), got {tta}'
    rows, seen_args = [], {}
    for p, t in zip(A.members, tags):                  # preflight everything before anything is written
        base = os.path.basename(p)
        parts = base.split('_')
        assert len(parts) >= 4 and re.fullmatch(r'f-?\d+', parts[0]) and parts[1].startswith('s'), \
            f'{base}: expected f<fold>_s<seed>_<tag>_..._best.pt'
        a = torch.load(p, map_location='cpu', weights_only=False)['args']
        assert a.get('contract') == fc.CONTRACT, (p, 'contract differs from this fleet_common')
        assert a.get('amp') == 'fp16' and not a.get('bf16', False), (p, a.get('amp'))
        mod = a.get('model_mod', 'fleet_model')
        assert mod in MODS, (p, 'unknown model_mod', mod)
        assert a.get('model_sha') in (None, _model_sha(mod)), (p, f'{mod} source changed since training', a.get('model_sha'))
        sig = json.dumps(a, sort_keys=True, default=str)                                       # P-RUN
        assert sig not in seen_args, f'{base}: same run as {seen_args.get(sig)} (identical ckpt args): one member per run'
        seen_args[sig] = base
        old = '_'.join(parts[:2])
        own = f'f{int(a["fold"])}_s{int(a["seed"])}'
        if t in ('=', '-', ''):
            new = old
        else:                                                                                   # P-TAG
            m = re.fullmatch(r'(f-?\d+_s\d+)([A-Za-z0-9]*)', t)
            assert m and m.group(1) == own, (f'{base}: --tag {t!r} must be this member\'s own {own} plus an optional '
                                             f'[A-Za-z0-9] suffix (e.g. {own}v4)')
            new = t
        rows.append(dict(src=p, base=base, staged=new + base[len(old):], key=new, old=old, a=a, mod=mod))
    keys = [r['key'] for r in rows]
    assert len(keys) == len(set(keys)), f'duplicate member keys {sorted(keys)} (re-key one of them with --tag)'
    names = [r['staged'] for r in rows]
    assert len(names) == len(set(names)), f'duplicate file names after re-key: {sorted(names)}'
    shutil.rmtree(A.out, ignore_errors=True); os.makedirs(A.out)
    for f in SHIP:
        shutil.copy2(f'{HERE}/{f}', A.out)
    probe = set()
    for r in rows:
        a = r['a']
        probe |= set(a.get('decode_probe', {}))
        shutil.copy2(r['src'], f'{A.out}/{r["staged"]}')
        print('member', r['staged'] + (f' (re-keyed from {r["base"]})' if r['key'] != r['old'] else ''),
              '| fold', a['fold'], 'seed', a['seed'], '| sel', a.get('sel_labels'), '| train_slices', a.get('train_slices'),
              '|', r['mod'], a['backbone'], (f'ttr_size {a.get("ttr_size")} ' if r['mod'] == 'fleet_model_ttr' else '')
              + '| train_labels', a.get('train_labels'), '| sha', a.get('model_sha'))
    for s in sorted(probe):
        shutil.copy2(f'{A.store}/{s}.npz', f'{A.out}/parity_{s}.npz')   # flat: no sub-folder
    json.dump({'checkpoints': names,
               'sources': {r['staged']: dict(original=r['base'],                                  # FIX-5
                                             run_dir=os.path.basename(os.path.dirname(os.path.abspath(r['src']))),
                                             sha256=_sha16(r['src']), model_mod=r['mod'],
                                             model_sha=r['a'].get('model_sha'),
                                             **({'tta': tta[k]} if tta is not None else {}))          # K5
                           for k, r in enumerate(rows)}},
              open(f'{A.out}/fleet_manifest.json', 'w'), indent=1)
    json.dump({'title': A.title, 'id': A.id, 'licenses': [{'name': 'CC0-1.0'}]}, open(f'{A.out}/dataset-metadata.json', 'w'), indent=1)
    sha = hashlib.sha256(open(f'{A.out}/fleet_common.py', 'rb').read()).hexdigest()[:16]
    print(f'{A.out}: {len(rows)} members, parity {sorted(probe)}, fleet_common sha256[:16] {sha}')
    return names


if __name__ == '__main__':
    main()
