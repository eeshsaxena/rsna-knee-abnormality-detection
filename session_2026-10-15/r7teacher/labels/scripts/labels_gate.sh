#!/bin/bash
# labels_gate.sh TABLE (r7teacher/labels, 2026-10-14): queue-runner gate adapter for the R7-teacher student lines
# (`needs PASS from ~/r7t_labels/real/labels_gate_<T>.verdict wait 120 via bash ~/r7t_labels/labels_gate.sh <T>`).
# CPU only, seconds. Writes ONE verdict line to ~/r7t_labels/real/labels_gate_<T>.verdict:
#   PASS  iff the REAL build receipt exists, is real (not SIM), carries the PREREG_TEACHER sha256 that is recorded in
#         ~/r7t_labels/PREREG_TEACHER.sha256, its L0 verdict is PASS or DROP, THIS table passed the label gate L1 bar on its
#         own (receipt label_gate.passes[a|b]; PREREG_TEACHER Amendment A2: T_b on the rental, T_a on the box), and the table
#         file on disk still has the md5 the receipt recorded for it (table_a / table_b);
#   FAIL  otherwise (with the reason). A SIM_ table can never pass.
# TABLE = labels_r7t_b.csv (primary) or labels_r7t_a.csv (fallback).
set -u
TB=${1:?usage: labels_gate.sh labels_r7t_b.csv|labels_r7t_a.csv}
D=$HOME/r7t_labels/real; V=$D/labels_gate_${TB%.csv}.verdict; RC=$D/teacher_labels_receipt.json
mkdir -p "$D"
PY=$HOME/miniconda3/envs/rsna/bin/python
out=$("$PY" - "$RC" "$TB" "$D" "$HOME/r7t_labels/PREREG_TEACHER.sha256" <<'EOF'
import json, sys, os, hashlib
rc, tb, d, shaf = sys.argv[1:5]
def fail(m): print('FAIL ' + m); sys.exit(0)
if tb.startswith('SIM_'): fail('SIM table')
if not os.path.isfile(rc): fail('no REAL receipt ' + rc)
r = json.load(open(rc))
if not r.get('real') or str(r.get('teacher_tag', '')).startswith('SIM'): fail('receipt is a simulation')
want = open(shaf).read().split()[0] if os.path.isfile(shaf) else None
if not want or r.get('prereg_sha256') != want: fail('PREREG sha256 missing or different')
g = r.get('label_gate', {})
k = tb[len('labels_r7t_'):-len('.csv')] if tb.startswith('labels_r7t_') and tb.endswith('.csv') else ''
if k not in ('a', 'b'): fail('unknown table ' + tb)
if (r.get('l0') or {}).get('verdict') not in ('PASS', 'DROP'): fail('L0 verdict %s' % (r.get('l0') or {}).get('verdict'))
if (g.get('passes') or {}).get(k) is not True: fail('%s did not pass the label gate L1 bar (passes %s)' % (tb, g.get('passes')))
p = os.path.join(d, tb)
if not os.path.isfile(p): fail('table missing ' + p)
m = hashlib.md5(open(p, 'rb').read()).hexdigest()
want_md5 = (r.get('table_' + k) or {}).get('md5')
if m != want_md5: fail('table md5 %s != receipt %s' % (m, want_md5))
print('PASS %s md5 %s teacher %s prereg %s' % (tb, m, r.get('teacher_tag'), want[:12]))
EOF
)
[ -n "$out" ] || out="FAIL gate script error"
printf '%s\n' "$out" > "$V.tmp" && mv -f "$V.tmp" "$V"
echo "$out"
