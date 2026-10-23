#!/usr/bin/env python3
"""spend_tally.py (gold_prep/relabel, 2026-10-08): actual API spend of the relabel pilot from the fetched batch results,
plus the pre-registered projection check (prereg_v2.json "spend").

Reads results/raw_<arm>_<tag>.jsonl (written by relabel_call.py batch_fetch / sync), dedups by (batch id or 'sync',
custom_id) because batch_fetch re-appends earlier batches, and prices message.usage:
  batch: 0.5 x (input x P_in + cache_creation x P_cw + cache_read x P_cr + output x P_out) / 1e6
  sync : the same without the 0.5 factor.
No network, no key.

  python spend_tally.py [--tag opus-5-5] [--plan text=64,img=64]
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PR = json.load(open(os.path.join(HERE, 'prereg_v2.json'), encoding='utf-8'))
PX = PR['spend']['prices_usd_per_mtok']
CAP = float(PR['spend']['cap_usd'])
MARGIN = 1.3

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='opus-5-5')
ap.add_argument('--plan', default='', help='requests still to submit per arm, e.g. text=64,img=64')
A = ap.parse_args()


def cost(u, batch):
    c = (u.get('input_tokens') or 0) * PX['input'] + (u.get('cache_creation_input_tokens') or 0) * PX['cache_write_5m'] \
        + (u.get('cache_read_input_tokens') or 0) * PX['cache_read'] + (u.get('output_tokens') or 0) * PX['output']
    return c / 1e6 * (PX['batch_discount'] if batch else 1.0)


tot, per = 0.0, {}
for arm in ('text', 'img'):
    p = os.path.join(HERE, 'results', f'raw_{arm}_{A.tag}.jsonl')
    seen, n, c_arm, tok = set(), 0, 0.0, dict(input=0, cache_creation=0, cache_read=0, output=0)
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            key = (r.get('batch') or 'sync:' + r.get('ts', ''), r['custom_id'])
            if key in seen:
                continue
            seen.add(key)
            u = ((r.get('message') or {}).get('usage')) or {}
            if not u:
                continue                                   # errored / expired batch request: not billed
            n += 1
            c_arm += cost(u, bool(r.get('batch')))
            tok['input'] += u.get('input_tokens') or 0
            tok['cache_creation'] += u.get('cache_creation_input_tokens') or 0
            tok['cache_read'] += u.get('cache_read_input_tokens') or 0
            tok['output'] += u.get('output_tokens') or 0
    per[arm] = dict(n_billed=n, usd=round(c_arm, 4), usd_per_req=round(c_arm / n, 5) if n else None,
                    tokens=tok, mean_out=round(tok['output'] / n) if n else None,
                    mean_in=round((tok['input'] + tok['cache_creation'] + tok['cache_read']) / n) if n else None)
    tot += c_arm
print(json.dumps(dict(actual_usd=round(tot, 4), cap_usd=CAP, per_arm=per), indent=1))
if A.plan:
    plan = {k: int(v) for k, v in (x.split('=') for x in A.plan.split(','))}
    proj = tot + MARGIN * sum(plan[a] * (per[a]['usd_per_req'] or 0) for a in plan)
    print(f'projection: actual {tot:.4f} + {MARGIN} x planned {plan} = {proj:.4f} USD vs cap {CAP:.2f}: '
          f'{"OK to submit" if proj <= CAP else "DO NOT SUBMIT"}' + (' (> $6.00: chunk <= 20 per arm)' if 6.0 < proj <= CAP else ''))
