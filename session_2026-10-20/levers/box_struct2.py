# aggregates only: column names and shapes (no rows, no ids)
import json, os
import pandas as pd
base = os.path.expanduser('~/r5_oai/run_r6/worker0')
for f in ['state_stacker.csv', 'fluid_stacker.csv', 'predictions.csv']:
    t = pd.read_csv(os.path.join(base, f))
    print(f, t.shape, [c for c in t.columns][:80])
t = pd.read_csv(os.path.expanduser('~/r5_oai/s6_r5/out/predictions.csv'))
print('s6 pred', t.shape, list(t.columns)[:20])
l = json.load(open(os.path.expanduser('~/r5_oai/s6_r5/out/logits.json')))
print('s6 logits type', type(l).__name__, len(l))
if isinstance(l, dict):
    v = next(iter(l.values()))
    print('s6 entry type', type(v).__name__, (list(v.keys()) if isinstance(v, dict) else (len(v) if isinstance(v, list) else v)))
print(sorted(os.listdir(os.path.expanduser('~/oai_build')))[:60])
