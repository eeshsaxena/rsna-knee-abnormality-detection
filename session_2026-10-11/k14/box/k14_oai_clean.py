#!/usr/bin/env python3
"""k14_oai_clean.py (box, CPU, read-only; 2026-10-09, Claude). K14-OAI-CLEAN (PREREG_K14 sec 3): a D-OAI checkpoint may
leave the box only if it carries no OAI data. For each --ckpt PATH[=OOF_CSV[=REF_OOF_CSV]]:
  ckpt dict   top-level keys exactly {model, args, cv, epoch}; `model` = tensors only (a state dict); recursively over
              everything else (args incl. nested dicts / lists / tuples, cv, epoch): no string contains 'oai_' followed by a
              digit (an OAI knee id), no container with > 64 string entries (an id list / label table), no pandas object,
              no numpy array with > 64 elements, no bytes blob > 4 KB; args['decode_probe'] keys are RSNA study UIDs only
              (^1\\.2\\.826\\.0\\.1\\.3680043\\.) and there are <= 8 of them
  OOF csv     no id containing 'oai'; == the reference OOF's id set (set D's OOF of the same fold: RSNA val rows only)
Strings that merely NAME an OAI-derived file (args train_labels 'labels_frac75oai_T123v00.csv') carry no data and are
listed in the receipt, not refused. Writes --out (a json receipt) only. Last line: K14CLEAN PASS|FAIL n/m.
  ~/miniconda3/envs/kmimic/bin/python k14_oai_clean.py --out R.json --ckpt CKPT=OOF=REFOOF [--ckpt ...]
"""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd

OAI_ID = re.compile(r'oai_\d', re.I)
RSNA_UID = re.compile(r'^1\.2\.826\.0\.1\.3680043\.')


def walk(x, path, hits, names):
    import torch
    if isinstance(x, str):
        if OAI_ID.search(x):
            hits.append(f'{path}: OAI id pattern in {x[:80]!r}')
        elif 'oai' in x.lower():
            names.append(f'{path}={x[:120]}')
        return
    if isinstance(x, (bytes, bytearray)):
        if len(x) > 4096:
            hits.append(f'{path}: bytes blob {len(x)} B')
        return
    if isinstance(x, (pd.DataFrame, pd.Series, pd.Index)):
        hits.append(f'{path}: pandas {type(x).__name__}')
        return
    if isinstance(x, np.ndarray):
        if x.size > 64:
            hits.append(f'{path}: numpy array {x.shape}')
        return
    if isinstance(x, torch.Tensor):
        hits.append(f'{path}: tensor outside the model state {tuple(x.shape)}')
        return
    if isinstance(x, dict):
        ks = [k for k in x if isinstance(k, str)]
        if len(ks) > 64:
            hits.append(f'{path}: dict with {len(ks)} string keys')
        for k, v in x.items():
            walk(k, f'{path}.<key>', hits, names)
            walk(v, f'{path}.{k}', hits, names)
        return
    if isinstance(x, (list, tuple, set)):
        if sum(isinstance(v, str) for v in x) > 64:
            hits.append(f'{path}: container with > 64 strings')
        for i, v in enumerate(x):
            walk(v, f'{path}[{i}]', hits, names)


def scan(ckpt, oof, ref):
    import torch
    hits, names = [], []
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if set(ck) != {'model', 'args', 'cv', 'epoch'}:
        hits.append(f'top-level keys {sorted(ck)}')
    m = ck.get('model')
    if not isinstance(m, dict) or not m or not all(isinstance(v, torch.Tensor) for v in m.values()):
        hits.append('model is not a pure tensor state dict')
    for k, v in ck.items():
        if k != 'model':
            walk(v, k, hits, names)
    dp = (ck.get('args') or {}).get('decode_probe') or {}
    bad = [s for s in dp if not RSNA_UID.match(str(s))]
    if bad or not dp or len(dp) > 8:
        hits.append(f'decode_probe keys {list(dp)[:8]} (non-RSNA {bad[:3]})')
    rec = dict(ckpt=ckpt, n_model_tensors=len(m or {}), decode_probe=sorted(dp), oai_named_files=names)
    if oof:
        o = pd.read_csv(oof, dtype={'StudyInstanceUID': str})
        ids = set(o['StudyInstanceUID'])
        oai = [s for s in ids if 'oai' in s.lower()]
        if oai:
            hits.append(f'OOF {oof}: {len(oai)} oai ids')
        if ref:
            rids = set(pd.read_csv(ref, dtype={'StudyInstanceUID': str})['StudyInstanceUID'])
            if ids != rids:
                hits.append(f'OOF ids != reference {ref} ({len(ids ^ rids)} differ)')
        rec.update(oof=oof, oof_rows=len(o), ref=ref)
    rec['hits'] = hits
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), host=os.uname().nodename, items=[])
    for spec in a.ckpt:
        parts = spec.split('=')
        ck, oof, ref = parts[0], (parts[1] if len(parts) > 1 else None), (parts[2] if len(parts) > 2 else None)
        r = scan(os.path.expanduser(ck), oof and os.path.expanduser(oof), ref and os.path.expanduser(ref))
        out['items'].append(r)
        print(f'{"PASS" if not r["hits"] else "FAIL"}  {os.path.basename(ck)}: {len(r["hits"])} hit(s) {r["hits"][:3]} | '
              f'decode_probe {len(r["decode_probe"])} RSNA | OOF rows {r.get("oof_rows")} | names {r["oai_named_files"]}', flush=True)
    n = sum(not r['hits'] for r in out['items'])
    out['pass'] = n == len(out['items'])
    json.dump(out, open(os.path.expanduser(a.out), 'w'), indent=1)
    print(f'K14CLEAN {"PASS" if out["pass"] else "FAIL"} {n}/{len(out["items"])} -> {a.out}')
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
