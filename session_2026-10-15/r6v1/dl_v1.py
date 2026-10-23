# dl_v1.py: per-file download of heliosli/rsna-knee-r6-frozen-fe055406 VERSION 1 via the legacy endpoint, then sha-verify vs manifest.
import requests, hashlib, json, os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate(); cv = api.config_values; auth = (cv['username'], cv['key'])
FILES = ['manifest.json', 'fluid_stacker.json', 'state_stacker.json', 'state_spec.json', 'side_cls.pt',
         'E_lrprior_s0_f0.pt', 'E_lrprior_s0_f1.pt', 'E_lrprior_s0_f2.pt', 'E_lrprior_s0_f3.pt']
os.makedirs('raw', exist_ok=True)
for f in FILES:
    url = 'https://www.kaggle.com/api/v1/datasets/download/heliosli/rsna-knee-r6-frozen-fe055406/%s?datasetVersionNumber=1' % f
    for a in range(5):
        try:
            with requests.get(url, auth=auth, stream=True, timeout=120) as r:
                r.raise_for_status(); h = hashlib.sha256(); n = 0
                with open('raw/' + f + '.part', 'wb') as o:
                    for c in r.iter_content(1 << 20): o.write(c); h.update(c); n += len(c)
            os.replace('raw/' + f + '.part', 'raw/' + f); print('OK', f, n, h.hexdigest()[:16], flush=True); break
        except Exception as e:
            print('RETRY', f, a, type(e).__name__, str(e)[:100], flush=True)
m = json.load(open('raw/manifest.json'))
print('MANIFEST_SHA', hashlib.sha256(open('raw/manifest.json','rb').read()).hexdigest()[:16])
print(json.dumps(m)[:1500])
