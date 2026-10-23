# make_box_abc.py: embed public-LB posterior draws (G10, G20, gap960; lb_post.py) into the box ABC script. No OAI content.
import numpy as np, json, sys
d = np.load('lb_post_pooled.npz'); rng = np.random.default_rng(1); i = rng.choice(len(d['G10']), int(sys.argv[1]) if len(sys.argv) > 1 else 100, replace=False)
dr = np.stack([d['G10'][i], d['G20'][i], d['gap960'][i]], 1) * 1e-3
t = open('box_abc_template.py').read().replace('__LBDRAWS__', json.dumps(np.round(dr, 7).tolist()))
open('box_abc_run.py', 'w').write(t); print('embedded', len(dr), 'draws; mean', dr.mean(0))
