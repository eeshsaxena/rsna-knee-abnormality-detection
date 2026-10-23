#!/usr/bin/env python3
"""cc_make_tasks.py (2026-10-12, Claude; prereg_v3.json): blinded task files for the IN-SESSION relabel pilot.
For every gold74 study and both arms writes cc_tasks/<arm>/<pid>/task.md (+ img arm: the key-slice PNGs copied as
img_01.png ..). pid = 'S' + 12 hex of sha256(StudyInstanceUID) (relabel_call.pid_of). task.md = prompt_v1 SYSTEM +
REPORT_BLOCK + the arm's notes + the JSON schema, i.e. the same text the API request would carry. Also writes
cc_tasks/blocks.json (the fixed agent blocks, gold74 order) and cc_tasks/id_map.csv (kept local; agents never read it).
No gold label is read: the gold file is used only for the study ids and their order.
"""
import os, sys, json, shutil, hashlib
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompt_v1 as P

H = os.path.expanduser
GOLD = H('~/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train/gold74_eval_set.csv')
TRAIN = H('~/Downloads/rsna-knee/_data/train.csv')
MAN = os.path.join(HERE, 'manifest_gold74.csv')
OUT = os.path.join(HERE, 'cc_tasks')
pid_of = lambda sid: 'S' + hashlib.sha256(sid.encode()).hexdigest()[:12]

ids = pd.read_csv(GOLD, dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID']).StudyInstanceUID.tolist()
tr = pd.read_csv(TRAIN, dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID', 'Report']).set_index('StudyInstanceUID')
man = pd.read_csv(MAN, dtype={'StudyInstanceUID': str})
assert len(ids) == 74 and not (set(ids) - set(man.StudyInstanceUID))
SCHEMA = json.dumps(P.label_schema(), indent=1)
OUTFMT = ('\n\nRESPONSE FORMAT\nReturn ONE JSON object with exactly the keys ' + ', '.join(k for k, _ in P.LABELS) +
          '. Each value is an object with report_evidence (one of ' + ', '.join(P.REPORT_EVIDENCE) + '), image_evidence (one of '
          + ', '.join(P.IMAGE_EVIDENCE) + '), call (one of ' + ', '.join(P.CALLS) + '), probability (integer 0-100) and note '
          '(<= 20 words). JSON schema:\n' + SCHEMA + '\n')


def rep(sid):
    r = tr.Report.get(sid, '')
    return r if isinstance(r, str) and r.strip() else '(empty report)'


if os.path.isdir(OUT):
    shutil.rmtree(OUT)
idmap = []
for sid in ids:
    pid = pid_of(sid); idmap.append((sid, pid))
    for arm in ('text', 'img'):
        d = os.path.join(OUT, arm, pid); os.makedirs(d)
        parts = ['SYSTEM PROMPT\n' + P.SYSTEM, 'USER MESSAGE\n' + P.REPORT_BLOCK.format(pid=pid, report=rep(sid))]
        if arm == 'text':
            parts.append(P.NO_IMAGES_NOTE)
        else:
            rows = man[man.StudyInstanceUID == sid].sort_values('img_idx')
            parts.append(P.IMAGES_INTRO.format(n=len(rows)))
            for k, r in enumerate(rows.itertuples(index=False), 1):
                cap = P.CAPTION.format(k=k, n=len(rows), plane_text=P.PLANE_TEXT[r.view], contrast=r.contrast,
                                       i=int(r.stack_idx) + 1, n_stack=int(r.n_stack), direction=r.direction,
                                       targets=', '.join(r.targets.split(';')))
                fn = f'img_{k:02d}.png'
                shutil.copyfile(os.path.join(HERE, r.png), os.path.join(d, fn))
                parts.append(cap + f'\n[IMAGE {k}: view the file {fn} in this folder]')
            parts.append(P.IMAGES_NOTE)
        open(os.path.join(d, 'task.md'), 'w', encoding='utf-8').write('\n\n'.join(parts) + OUTFMT)
pids = [p for _, p in idmap]
blocks = {'img': [pids[i:i + 10] for i in range(0, 74, 10)],
          'text': [pids[i:i + 25] for i in range(0, 74, 25)]}
json.dump(blocks, open(os.path.join(OUT, 'blocks.json'), 'w'), indent=1)
pd.DataFrame(idmap, columns=['StudyInstanceUID', 'pid']).to_csv(os.path.join(HERE, 'cc_id_map.csv'), index=False)
print('tasks', len(ids), 'x 2 arms | img blocks', [len(b) for b in blocks['img']], '| text blocks', [len(b) for b in blocks['text']],
      '| prompt sha', P.prompt_sha()[:16])
