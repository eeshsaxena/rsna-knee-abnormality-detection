#!/usr/bin/env python3
"""cc_collect.py (2026-10-12, Claude; prereg_v3.json): collects the in-session answers cc_out/<arm>/<pid>.json into
results/soft_<arm>_cc-opus-5-5.csv in the relabel_call.py row format (pilot_eval.py reads it unchanged).
  python cc_collect.py --status     # per arm: done / invalid / missing pids (no answer content printed)
  python cc_collect.py              # write both CSVs (status ok only for schema-valid answers)
"""
import os, sys, json, argparse
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompt_v1 as P

ap = argparse.ArgumentParser(); ap.add_argument('--status', action='store_true'); A = ap.parse_args()
TAG = 'cc-opus-5-5'
idmap = pd.read_csv(os.path.join(HERE, 'cc_id_map.csv'), dtype=str)
blocks = json.load(open(os.path.join(HERE, 'cc_tasks', 'blocks.json')))


def check(obj):
    if not isinstance(obj, dict) or set(obj) != {k for k, _ in P.LABELS}:
        return 'keys'
    for k, _ in P.LABELS:
        d = obj[k]
        if not isinstance(d, dict):
            return f'{k} not an object'
        if d.get('report_evidence') not in P.REPORT_EVIDENCE or d.get('image_evidence') not in P.IMAGE_EVIDENCE \
                or d.get('call') not in P.CALLS:
            return f'{k} enum'
        pr = d.get('probability')
        if not isinstance(pr, int) or isinstance(pr, bool) or not 0 <= pr <= 100:
            return f'{k} probability'
    return ''


def load(arm, pid):
    p = os.path.join(HERE, 'cc_out', arm, pid + '.json')
    if not os.path.isfile(p):
        return 'missing', None, ''
    try:
        obj = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        return 'invalid', None, f'json: {e!r}'[:120]
    err = check(obj)
    return ('ok', obj, '') if not err else ('invalid', None, err)


for arm in ('text', 'img'):
    rows, st = [], {'ok': [], 'invalid': [], 'missing': []}
    for sid, pid in idmap.itertuples(index=False):
        s, obj, err = load(arm, pid)
        st[s].append(pid)
        r = dict(StudyInstanceUID=sid, pid=pid, arm=arm, model_requested='claude-opus-5-5',
                 served_model='claude-opus-5-5' if s == 'ok' else None, status=s, error=err, input_tokens=None,
                 output_tokens=None, cache_read_tokens=None, prompt_sha=P.prompt_sha()[:16], effort='cc-default',
                 channel='claude-code-subagent')
        for k, name in P.LABELS:
            d = (obj or {}).get(k) or {}
            r[name] = d['probability'] / 100.0 if 'probability' in d else float('nan')
            r[f'{name}__call'] = d.get('call'); r[f'{name}__report_ev'] = d.get('report_evidence')
            r[f'{name}__image_ev'] = d.get('image_evidence'); r[f'{name}__note'] = d.get('note')
        rows.append(r)
    miss_blocks = [i for i, b in enumerate(blocks[arm]) if any(p in st['missing'] or p in st['invalid'] for p in b)]
    print(f'{arm}: ok {len(st["ok"])} | invalid {len(st["invalid"])} {st["invalid"][:5]} | missing {len(st["missing"])} '
          f'| blocks with gaps {miss_blocks}')
    if not A.status:
        os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
        out = os.path.join(HERE, 'results', f'soft_{arm}_{TAG}.csv')
        pd.DataFrame(rows).to_csv(out, index=False); print('  wrote', out)
