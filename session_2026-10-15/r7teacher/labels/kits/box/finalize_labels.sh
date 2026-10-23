#!/bin/bash
# finalize_labels.sh (r7teacher/labels, 10-14 16:10, PREREG_TEACHER Amendment A2; BOX, CPU, seconds). Run right after
# on_r7oof.sh finished the REAL build. It runs labels_gate.sh for BOTH tables (A2 per-table rule: REAL receipt, PREREG sha,
# L0 PASS/DROP, the table passed L1 on its own, md5 unchanged), then publishes the passing tables for the T_b rental agent and
# the box: ~/r7t_labels/final/{labels_r7t_b.csv, labels_r7t_a.csv, teacher_labels_receipt.json} (copies, md5 re-checked) and
# the marker ~/r7t_labels/final/LABELS_READY (written LAST, atomically) with the sha256 and md5 of every file, the PREREG
# sha256, the L0 verdict and the label gate deltas. T_b FAIL -> no LABELS_READY (LABELS_STOP with the reason instead); T_a only
# is listed as such. Never touches the queue or the GPU.
set -u
D=$HOME/r7t_labels; R=$D/real; F=$D/final; PY=$HOME/miniconda3/envs/rsna/bin/python
[ -s $R/teacher_labels_receipt.json ] || { echo "STOP: no REAL receipt (run on_r7oof.sh first)"; exit 1; }
[ -e $F/LABELS_READY ] && { echo "STOP: $F/LABELS_READY already exists (one teacher per PREREG)"; exit 1; }
VB=$(bash $D/labels_gate.sh labels_r7t_b.csv); VA=$(bash $D/labels_gate.sh labels_r7t_a.csv)
echo "labels_gate T_b: $VB"; echo "labels_gate T_a: $VA"
mkdir -p $F
ok=""
for t in b a; do V=$([ $t = b ] && echo "$VB" || echo "$VA")
  case "$V" in PASS*) cp -p $R/labels_r7t_$t.csv $F/labels_r7t_$t.csv.tmp && mv -f $F/labels_r7t_$t.csv.tmp $F/labels_r7t_$t.csv; ok="$ok $t";; esac
done
cp -p $R/teacher_labels_receipt.json $F/teacher_labels_receipt.json
$PY - "$R" "$F" "$ok" "$VB" "$VA" "$D/PREREG_TEACHER.sha256" <<'EOF'
import hashlib, json, os, sys, time
R, F, ok, vb, va, shaf = sys.argv[1:7]
ok = ok.split()
rec = json.load(open(f'{R}/teacher_labels_receipt.json'))
h = lambda p, a: hashlib.new(a, open(p, 'rb').read()).hexdigest()
lines = [f'# R7-teacher student label tables (PREREG_TEACHER + A1 + A2), written {time.strftime("%Y-%m-%d %H:%M:%S")} by finalize_labels.sh',
         f'prereg_sha256 {open(shaf).read().split()[0]}', f'teacher_tag {rec.get("teacher_tag")} real {rec.get("real")}',
         f'teacher_sha256 {rec["inputs"]["teacher"]["sha256"]}',
         f'L0 {rec["l0"]["verdict"]} spread {rec["l0"]["spread"]} auc_per_fold {rec["l0"]["auc_per_fold"]} outlier {rec["l0"].get("outlier_fold")}',
         f'label_gate delta_vs_frac75 {rec["label_gate"]["delta_vs_frac75"]} passes {rec["label_gate"]["passes"]} bar {rec["label_gate"]["bar"]}',
         f'labels_gate T_b: {vb}', f'labels_gate T_a: {va}']
for t in ok:
    p = f'{F}/labels_r7t_{t}.csv'
    assert h(p, 'md5') == rec[f'table_{t}']['md5'], p
    lines.append(f'table T_{t} {os.path.basename(p)} sha256 {h(p, "sha256")} md5 {h(p, "md5")} seed {1047 if t == "b" else 1048}')
lines.append(f'receipt teacher_labels_receipt.json sha256 {h(F + "/teacher_labels_receipt.json", "sha256")}')
name = 'LABELS_READY' if 'b' in ok else 'LABELS_STOP'
if name == 'LABELS_STOP':
    lines.append('T_b did not pass: no T_b student (the rental must not start); ' + ('T_a passed: box T_a only' if 'a' in ok else 'T_a failed too: STOP'))
open(f'{F}/{name}.tmp', 'w').write('\n'.join(lines) + '\n'); os.replace(f'{F}/{name}.tmp', f'{F}/{name}')
print('\n'.join(lines)); print(f'WROTE {F}/{name}')
EOF
ls -la $F
