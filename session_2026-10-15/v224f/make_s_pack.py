#!/usr/bin/env python
# make_s_pack.py (2026-10-15 dir, 10-14, Claude subagent; R7-teacher line). NOT RUN YET (no student exists). BOX, CPU, read-only on
# the runs: stages the upload folder of the private S weights dataset for V224-F-S. It never creates a dataset and never pushes.
# Run by the main session only after the REAL student gate printed VERDICT PASS (PREREG_TEACHER sec 6 + A1).
#   python make_s_pack.py --gate ~/r7t_labels/gate/student_gate_<stamp>.txt
#        [--tpl 'f{F}_s1047_ttr_r7tb'] [--table ~/r7t_labels/real/labels_r7t_b.csv] [--ref ~/k11_test/a0/ds/rsna-knee-cnxt-fleet-k11w]
#        [--slug rsna-knee-cnxt-fleet-r7tb] [--out ~/r7t_labels/s_pack]
# Checks per fold F (0-4): exactly one raw-best checkpoint f{F}_s1047_raw_*_best.pt; ckpt args fold F / seed 1047 / 21 epochs /
# ttr_size 224 / model_mod fleet_model_ttr / model_sha f141aeed136dfc01 / amp fp16 / train_labels == sel_labels == the table name
# (A1 FX4); contract, views, backbone, dim, n_slices, n_views, n_contrast, store, crop_mm, src_slices, jpeg_fp, select_sha equal to the
# k11w D member (same recipe); decode_probe studies inside the k11w parity npz set (the fleet worker's re-encode parity);
# an OAI-clean scan in the K12-OAI-CLEAN spirit (top-level keys {model, args, cv, epoch}; tensors only in `model`; no 'oai_<digit>'
# string; no long id lists / arrays / blobs; decode_probe keys RSNA UIDs only, <= 8). Then hard-links each checkpoint under its
# staged name f{F}_s1047r7tb_<rest> (the build_v224fs.py name rule) into OUT/upload/<slug>/ with r7tb_manifest.json (the format of
# k12w_manifest.json: checkpoints + sources{original, run_dir, sha256[:16], sha256_full, model_mod, model_sha, tta AB}) and
# dataset-metadata.json (private by default when created with the CLI). Prints the next commands (main session).
# Fix 10-15 (F2/F3, v224f/RESULT.md): the gate file must name student --tpl and table basename(--table) on its first line and carry
# the all-pass REAL A1 verdict line; its sha256 goes into S_PACK.json. NEXT text = the FLIP-base plan (--base ff, v224ffs_check).
# Fix aofix 10-15 (F2, v224f/RESULT.md "AO fix 10-15"): NEXT text = the PREREG_10-15 A2 + A3 branch table: a FF-S (--base ff,
# v224ffs_check); b FLIP-AO + S (--base flip --start-budget-h 7.0, v224aos_check); c FLRM-AO + S (--base flrm, --ref-flrm).
import argparse, glob, hashlib, json, os, re, sys

H = os.path.expanduser('~')
ap = argparse.ArgumentParser()
ap.add_argument('--gate', required=True)
ap.add_argument('--tpl', default='f{F}_s1047_ttr_r7tb')
ap.add_argument('--runs', default=f'{H}/fleet_runs')
ap.add_argument('--table', default=f'{H}/r7t_labels/real/labels_r7t_b.csv')
ap.add_argument('--ref', default=f'{H}/k11_test/a0/ds/rsna-knee-cnxt-fleet-k11w')
ap.add_argument('--slug', default='rsna-knee-cnxt-fleet-r7tb')
ap.add_argument('--out', default=f'{H}/r7t_labels/s_pack')
A = ap.parse_args()
import torch  # noqa: E402

OAI_ID = re.compile(r'oai_\d', re.I); RSNA = re.compile(r'^1\.2\.826\.0\.1\.3680043\.')
S_RE = re.compile(r'^f([0-4])_s1047r7tb_raw_ttr224-320x24_cv\d{4}_best\.pt$')
KEYS = ('contract', 'views', 'backbone', 'dim', 'n_slices', 'n_views', 'n_contrast', 'store', 'crop_mm', 'src_slices', 'jpeg_fp', 'select_sha')
g = open(A.gate).read()
assert 'R7T_STUDENT_GATE REAL A1' in g and 'VERDICT PASS' in g and 'SIMULATION' not in g, 'needs a REAL A1 student gate PASS file'
TB = os.path.basename(A.table)
# fix F2 10-15: the gate file must be THIS student's (--tpl) on THIS table, and its own verdict line must be the all-pass one.
# student_gate.py (frozen) line 1: 'student_gate <ts> | student <tpl> | table <path as passed> | teacher <path> | REAL'.
gl = g.splitlines(); hd = gl[0].split(' | ') if gl else []
gf = dict(x.split(' ', 1) for x in hd[1:4] if ' ' in x) if len(hd) == 5 else {}
vl = [ln for ln in gl if ln.startswith('R7T_STUDENT_GATE ')]
assert (hd[0].startswith('student_gate ') and hd[4] == 'REAL' and gf.get('student') == A.tpl
        and os.path.basename(gf.get('table', '')) == TB and vl == ["R7T_STUDENT_GATE REAL A1 | G0 P G1 P G2 P G3 P G3' P G4 P | VERDICT PASS"]
        if len(hd) == 5 else False), f'student gate file is not a REAL all-pass gate of student {A.tpl} on table {TB}: {gl[:1]} {vl}'
GATE_SHA = hashlib.sha256(open(A.gate, 'rb').read()).hexdigest()
ref_ck = sorted(glob.glob(f'{A.ref}/f0_s1043f75_raw_ttr224-*_best.pt'))
assert len(ref_ck) == 1, ref_ck
ref = torch.load(ref_ck[0], map_location='cpu', weights_only=False)['args']
parity = {f[len('parity_'):-len('.npz')] for f in os.listdir(A.ref) if f.startswith('parity_')}
fsha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def walk(x, path, hits):
    if isinstance(x, str):
        if OAI_ID.search(x):
            hits.append(f'{path}: OAI id pattern')
    elif isinstance(x, (bytes, bytearray)):
        if len(x) > 4096:
            hits.append(f'{path}: blob {len(x)} B')
    elif isinstance(x, dict):
        if sum(isinstance(k, str) for k in x) > 64:
            hits.append(f'{path}: dict with {len(x)} keys')
        for k, v in x.items():
            walk(k, f'{path}.key', hits); walk(v, f'{path}.{k}', hits)
    elif isinstance(x, (list, tuple, set)):
        if sum(isinstance(v, str) for v in x) > 64:
            hits.append(f'{path}: list of {len(x)} strings')
        for i, v in enumerate(x):
            walk(v, f'{path}[{i}]', hits)
    elif hasattr(x, 'shape') and getattr(x, 'size', 0) > 64:
        hits.append(f'{path}: array {getattr(x, "shape", None)}')
    elif type(x).__module__.startswith('pandas'):
        hits.append(f'{path}: pandas object')


picks = []
for F in range(5):
    d = f'{A.runs}/{A.tpl.format(F=F)}'
    ck = glob.glob(f'{d}/f{F}_s1047_raw_*_best.pt'); assert len(ck) == 1, (d, ck)
    c = torch.load(ck[0], map_location='cpu', weights_only=False); a = c['args']
    want = dict(fold=F, seed=1047, epochs=21, ttr_size=224, model_mod='fleet_model_ttr', model_sha='f141aeed136dfc01', amp='fp16',
                train_labels=TB, sel_labels=TB)
    mism = {k: a.get(k) for k, v in want.items() if a.get(k) != v}
    assert not mism, f'f{F} args {mism}'
    for k in KEYS:
        assert a.get(k) == ref.get(k), f'f{F} args {k} {a.get(k)!r} != the k11w D member {ref.get(k)!r}'
    dp = a.get('decode_probe') or {}
    assert set(dp) <= parity and len(dp) <= 8 and all(RSNA.match(u) for u in dp), f'f{F} decode_probe {sorted(dp)[:3]}'
    hits = [] if set(c) == {'model', 'args', 'cv', 'epoch'} else [f'top-level keys {sorted(c)}']
    if not all(isinstance(v, torch.Tensor) for v in c['model'].values()):
        hits.append('model is not a pure tensor state dict')
    for k, v in c.items():
        if k != 'model':
            walk(v, k, hits)
    assert not hits, f'f{F} OAI-clean scan: {hits}'
    f = os.path.basename(ck[0]); staged = f'f{F}_s1047r7tb' + f[len(f'f{F}_s1047'):]
    assert S_RE.match(staged), staged
    picks.append(dict(fold=F, src=ck[0], file=f, staged=staged, sha256=fsha(ck[0]), cv=float(c['cv']), epoch=int(c['epoch']) + 1))
    print(f'f{F} OK {f} -> {staged} cv {float(c["cv"]):.4f} ep {int(c["epoch"]) + 1} sha {picks[-1]["sha256"][:16]}', flush=True)
up = f'{A.out}/upload/{A.slug}'
assert not os.path.exists(up), f'{up} exists: move it aside first'
os.makedirs(up)
src = {}
for p in picks:
    os.link(p['src'], f'{up}/{p["staged"]}')
    assert fsha(f'{up}/{p["staged"]}') == p['sha256']
    src[p['staged']] = {'original': p['staged'], 'run_dir': 'members', 'sha256': p['sha256'][:16], 'sha256_full': p['sha256'],
                        'model_mod': 'fleet_model_ttr', 'model_sha': 'f141aeed136dfc01', 'tta': 'AB'}
man = {'checkpoints': [p['staged'] for p in picks], 'sources': src}
json.dump(man, open(f'{up}/r7tb_manifest.json', 'w'), indent=1)
json.dump({'title': A.slug, 'id': f'eeshsaxena/{A.slug}', 'licenses': [{'name': 'CC0-1.0'}]}, open(f'{up}/dataset-metadata.json', 'w'), indent=1)
assert sorted(os.listdir(up)) == sorted(man['checkpoints'] + ['r7tb_manifest.json', 'dataset-metadata.json'])
json.dump(dict(gate=A.gate, gate_sha256=GATE_SHA, gate_header=gl[0], table=A.table, picks=picks,
               manifest_sha256=fsha(f'{up}/r7tb_manifest.json')), open(f'{A.out}/S_PACK.json', 'w'), indent=1)
print(f'S_PACK OK {up} | manifest sha256 {fsha(up + "/r7tb_manifest.json")[:16]} | gate {os.path.basename(A.gate)} sha256 {GATE_SHA[:16]}')
print(f'''NEXT (main session; nothing below was run; RUNBOOK_STUDENT_1015 steps 4b-13 are the reference). Pick ONE branch by the
10-15 API best-id reads (PREREG_10-15 Amendments 2 + 3, R1 read rule); only --wf 0.20 is pre-registered; --base is required.
  1 OAI-clean scan on the box (runbook step 4b), then copy to the laptop:  scp -r training@192.168.10.30:{up} <LAPTOP>/session_2026-10-15/v224f/s_upload/
  2 create the PRIVATE dataset:  kaggle datasets create -p <LAPTOP>/session_2026-10-15/v224f/s_upload/{A.slug}
  3 build, commit test (running.py 6 -> RUNNING_COUNT <= 1, push the kernel dir, wait_fetch.py -> OUT), gate, by branch:
  a V224-FF20 became the team best -> V224-FF-S (PREREG_10-14 A4; FF base: time-aware flip 4.9 h, fleet start 5.1 h), kernel_v224ffs20 = eeshsaxena/rsna-sub-v224ffs20:
    python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base ff --s-manifest <LAPTOP>/session_2026-10-15/v224f/s_upload/{A.slug}/r7tb_manifest.json
    python session_2026-10-15/v224f/v224ffs_check.py OUT --wf 0.20 --build session_2026-10-15/v224f/kernel_v224ffs20/BUILD_V224FS.json --ref-flip session_2026-10-14/v224/out_v224flip_v1 --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-fleet session_2026-10-15/v224ff/out_v224ff20_v1
    pass: V224FFS_CHECK PASS 14/14. REPLACE (--base ff --replace, kernel_v224ffsr20) only after a positive FF-S read (runbook step 13).
  b V224-FLIP stays the team best -> FLIP-AO + S (A3 R2; supersedes A2 b FFSR20, so never --base ff --replace here; REPLACE is implied), kernel_v224flips20 = eeshsaxena/rsna-sub-v224flips20:
    python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base flip --start-budget-h 7.0 --s-manifest <LAPTOP>/session_2026-10-15/v224f/s_upload/{A.slug}/r7tb_manifest.json
    python session_2026-10-15/v224f/v224aos_check.py OUT --wf 0.20 --build session_2026-10-15/v224f/kernel_v224flips20/BUILD_V224FS.json --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-flip session_2026-10-14/v224/out_v224flip_v1 [--ref-s OUT_WITH_THE_SAME_S]
    pass: V224AOS_CHECK PASS 28/28 (29/29 with --ref-s).
  c V224-FLIP-RM became the team best -> FLRM-AO + S (A3 R3), kernel_v224flrms20 = eeshsaxena/rsna-sub-v224flrms20:
    python session_2026-10-15/v224f/build_v224fs.py --wf 0.20 --base flrm --start-budget-h 7.0 --s-manifest <LAPTOP>/session_2026-10-15/v224f/s_upload/{A.slug}/r7tb_manifest.json
    python session_2026-10-15/v224f/v224aos_check.py OUT --wf 0.20 --build session_2026-10-15/v224f/kernel_v224flrms20/BUILD_V224FS.json --ref-exact session_2026-10-14/v224/out_v224_v1 --ref-flip session_2026-10-14/v224/out_v224flip_v1 --ref-flrm session_2026-10-15/v224rm/out_v224flrm_v1 [--ref-s OUT_WITH_THE_SAME_S]
    pass: V224AOS_CHECK PASS 28/28 (29/29 with --ref-s); --ref-flrm is required.
  d FF20 shows a canary -> no FF-S / FFSR (A2 d).  e student gate FAIL or the G3' skip -> no student entry.
  4 amendment BEFORE the submit (a: PREREG_10-14 A4; b / c: the A3 entry text naming start guard 7.0 h / MIN_START_MIN 78 and the HF-AO Fracture
    rule); runtime review (a: runbook step 10; b / c: sim_ffs_runtime.py section D + guard_paired_1015.py); submit only on the gate PASS.''')
