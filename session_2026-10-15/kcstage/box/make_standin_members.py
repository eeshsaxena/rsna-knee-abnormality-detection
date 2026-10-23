# make_standin_members.py (10-15, Claude subagent, kcstage): STAND-IN KC members for the S6F+KC kernel plumbing commit.
# Box CPU only (cores 28-39, CUDA hidden). NO TRAINING DATA: each member is a RANDOMLY INITIALISED KneeFleetNetTTR
# (ConvNeXt-S KC-TTR 256, the exact class / args of the real KC1 members) saved in the trainer's raw-ckpt format
# {model, args, cv, epoch}; args = the s7 smoke ckpt's args (metadata only) + standin markers. The head is rescaled once so the
# logits are not all ~0 (a parity test on p ~ 0.5 everywhere would be weak): head_w *= HEAD_GAIN, chosen from ONE forward on
# 2 RSNA train studies of the KC store (RSNA-only; no OAI anywhere). vbias = the A6 anatomical prior (trainer init).
#   member 1: fp32 state dict (= the trainer's raw ckpt form); member 2: fp16 state dict (= the planned dataset form).
#   python make_standin_members.py --out ~/kcstage_1015/standin
import os, sys, json, time, glob, hashlib, argparse
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser('~/kc/code'))
sys.path.insert(1, os.path.expanduser('~/fast_build/code2'))          # read only: fleet_model.py / fleet_model_ttr.py
import numpy as np
import torch
import kc_common as kc
import kc_worker as kw

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--s7', default=os.path.expanduser('~/kc/smoke/s7/epochs/f0_s2000_raw_kcttr256-256x12_ep1_best.pt'))
ap.add_argument('--store', default=os.path.expanduser('~/kc_store/store'))
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--target-std', type=float, default=1.5)
A = ap.parse_args()
torch.set_num_threads(A.threads)
os.makedirs(A.out, exist_ok=True)
from fleet_model_ttr import KneeFleetNetTTR

s7 = torch.load(A.s7, map_location='cpu', weights_only=False)
args0 = dict(s7['args'])
msha, mfiles = kw.model_sha()
assert args0['model_sha'] == msha, (args0['model_sha'], msha)
assert args0['kc_contract_hash'] == kc.CONTRACT_HASH and args0['model_mod'] == 'kc_ttr' and args0['mirror'] == 0
del s7
studies = sorted(glob.glob(A.store + '/*.npz'))[:2]
X = []
for p in studies:
    with np.load(p) as z:
        views = {n: (kc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']) if f'v{j}_blob' in z.files else None)
                 for j, n in enumerate(kc.VIEW_NAMES)}
    x, pr, ct = kc.views_to_tensor(views)
    X.append((torch.from_numpy(x)[None], torch.from_numpy(pr)[None], torch.from_numpy(ct)[None]))
rec = {'s7_args_from': A.s7, 'model_sha': msha, 'calib_studies': [os.path.basename(p)[:-4] for p in studies], 'members': []}
for k, (seed, dtype) in enumerate(((11, torch.float32), (12, torch.float16)), 1):
    torch.manual_seed(seed)
    m = KneeFleetNetTTR(ttr_size=args0['ttr_size'], backbone=args0['backbone'], dim=args0['dim'], contrast_emb=args0['contrast_emb'],
                        n_views=args0['n_views'], n_slices=args0['n_slices'], n_contrast=args0['n_contrast'], pretrained=False,
                        drop_path=0.0, dropout=0.0, grad_ckpt=False).eval()
    with torch.no_grad():
        m.vbias.copy_(torch.from_numpy(kc.vbias_prior_matrix()))
        t0 = time.time()
        z = torch.cat([m(*xx) for xx in X]).float()
        sec = (time.time() - t0) / len(X)
        gain = float(A.target_std / max(float(z.std()), 1e-6))
        m.head_w.mul_(gain)
        m.head_b.zero_()
        z2 = torch.cat([m(*xx) for xx in X]).float()
    sd = {kk: (v.to(dtype) if v.is_floating_point() else v) for kk, v in m.state_dict().items()}
    args = dict(args0, seed=seed, standin=True, standin_note='RANDOM-INIT stand-in (no training, no labels); head gain %.3f' % gain,
                max_steps=0, epochs=0, save_epochs='none', state_dtype=str(dtype).replace('torch.', ''))
    name = 'kc_standin_m%d_s%d_raw_%s.pt' % (k, seed, 'fp32' if dtype == torch.float32 else 'fp16')
    torch.save({'model': sd, 'args': args, 'cv': None, 'epoch': -1}, os.path.join(A.out, name))
    h = hashlib.sha256(open(os.path.join(A.out, name), 'rb').read()).hexdigest()
    rec['members'].append({'file': name, 'seed': seed, 'dtype': args['state_dtype'], 'head_gain': gain, 'sha256': h,
                           'logit_std_before': float(z.std()), 'logit_std_after': float(z2.std()),
                           'p_range_after': [float(torch.sigmoid(z2).min()), float(torch.sigmoid(z2).max())],
                           'cpu_s_per_study_fwd': round(sec, 2), 'bytes': os.path.getsize(os.path.join(A.out, name))})
    print('MEMBER', json.dumps(rec['members'][-1]), flush=True)
json.dump(rec, open(os.path.join(A.out, 'standin_receipt.json'), 'w'), indent=1)
print('STANDIN_DONE', A.out, flush=True)
