# FIX test updates (seed 1042 for --model ttr = BJ-5; TTR-6 buffer; FIX-1 HALF; R-2 A,B passes; FIX-5 manifest)
import os


def load(p):
    s = open(p, encoding='utf-8', newline='').read()
    crlf = '\r\n' in s
    return s.replace('\r\n', '\n'), crlf


def save(p, s, crlf):
    open(p, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n') if crlf else s)


class R:
    def __init__(self, p):
        self.p = p; self.s, self.crlf = load(p)

    def rep(self, a, b, n=1):
        assert self.s.count(a) == n, (self.p, a[:80], self.s.count(a))
        self.s = self.s.replace(a, b)

    def done(self):
        save(self.p, self.s, self.crlf); print('ok', self.p)


# ---------------- t_b2_ttr_train
t = R('tests/t_b2_ttr_train.py')
t.rep("#   guards   --model ttr --f2 1, --save_epochs 3-1 must refuse",
      "#   guards   --model ttr --f2 1, --save_epochs 3-1, --model ttr --seed 42 (BJ-5) must refuse\n"
      "# TTR runs use --seed 1042 (BJ-5: TTR seeds 1000-1999). TTR-6: both ckpts carry ttr_size_t == 224 (int64; the EMA\n"
      "# copy never averages it) and a TTR state_dict does NOT strict-load into a plain KneeFleetNet.")
t.rep("#   bigjump pack (--tag f0_s42ttr for the TTR fold-0 member, which shares f0_s42 with the family-1 member) ->",
      "#   bigjump pack (--tag f0_s1042ttr re-keys the TTR fold-0 member; the others keep their own keys) ->")
t.rep("TTR = ['--model', 'ttr', '--aug', 'heavy', '--train_slices', '16', '--ema', '0.9995']",
      "TTR = ['--model', 'ttr', '--aug', 'heavy', '--train_slices', '16', '--ema', '0.9995', '--seed', '1042']   # BJ-5")
t.rep("r'f0_s42_(raw|ema)_ttr224-64x24_cv\\d+_best\\.pt'", "r'f0_s1042_(raw|ema)_ttr224-64x24_cv\\d+_best\\.pt'")
t.rep("assert ep == sorted([f'f0_s42_{t}_ttr224-64x24_ep{e}_best.pt' for t in ('raw', 'ema') for e in (2, 3)]\n"
      "                    + [f'oof_f0_s42_{t}_ep{e}.csv' for t in ('raw', 'ema') for e in (2, 3)]), ep",
      "assert ep == sorted([f'f0_s1042_{t}_ttr224-64x24_ep{e}_best.pt' for t in ('raw', 'ema') for e in (2, 3)]\n"
      "                    + [f'oof_f0_s1042_{t}_ep{e}.csv' for t in ('raw', 'ema') for e in (2, 3)]), ep")
t.rep("        ck = torch.load(f'{d}/epochs/f0_s42_{t}_ttr224-64x24_ep{e}_best.pt', weights_only=False)",
      "        ck = torch.load(f'{d}/epochs/f0_s1042_{t}_ttr224-64x24_ep{e}_best.pt', weights_only=False)")
t.rep("    best = torch.load(glob.glob(f'{d}/f0_s42_{t}_ttr224-*_best.pt')[0], weights_only=False)",
      "    best = torch.load(glob.glob(f'{d}/f0_s1042_{t}_ttr224-*_best.pt')[0], weights_only=False)\n"
      "    tb = best['model']['ttr_size_t']                                          # TTR-6\n"
      "    assert tb.dtype == torch.int64 and int(tb) == 224, (t, tb)")
t.rep("        snap = torch.load(f'{d}/epochs/f0_s42_{t}_ttr224-64x24_ep{be}_best.pt', weights_only=False)",
      "        snap = torch.load(f'{d}/epochs/f0_s1042_{t}_ttr224-64x24_ep{be}_best.pt', weights_only=False)")
t.rep("        assert open(f'{d}/oof_f0_s42_{t}.csv', 'rb').read() == open(f'{d}/epochs/oof_f0_s42_{t}_ep{be}.csv', 'rb').read()",
      "        assert open(f'{d}/oof_f0_s1042_{t}.csv', 'rb').read() == open(f'{d}/epochs/oof_f0_s1042_{t}_ep{be}.csv', 'rb').read()")
t.rep("train_slices=16, model_sha=TTR_SHA, save_epochs='2-3', mirror_aug=0.5, epochs=3, fold=0, seed=42)",
      "train_slices=16, model_sha=TTR_SHA, save_epochs='2-3', mirror_aug=0.5, epochs=3, fold=0, seed=1042)")
t.rep("assert ep2 == sorted(f'f-1_s42_{t}_ttr224-64x24_ep{e}_best.pt' for t in ('raw', 'ema') for e in (1, 2, 3)), ep2",
      "assert ep2 == sorted(f'f-1_s1042_{t}_ttr224-64x24_ep{e}_best.pt' for t in ('raw', 'ema') for e in (1, 2, 3)), ep2")
t.rep("    a1 = torch.load(f'{d2}/f-1_s42_{t}_ttr224-64x24_ep3_best.pt', weights_only=False)\n"
      "    a2 = torch.load(f'{d2}/epochs/f-1_s42_{t}_ttr224-64x24_ep3_best.pt', weights_only=False)",
      "    a1 = torch.load(f'{d2}/f-1_s1042_{t}_ttr224-64x24_ep3_best.pt', weights_only=False)\n"
      "    a2 = torch.load(f'{d2}/epochs/f-1_s1042_{t}_ttr224-64x24_ep3_best.pt', weights_only=False)")
t.rep("    e1 = torch.load(f'{d2}/epochs/f-1_s42_{t}_ttr224-64x24_ep1_best.pt', weights_only=False)",
      "    e1 = torch.load(f'{d2}/epochs/f-1_s1042_{t}_ttr224-64x24_ep1_best.pt', weights_only=False)")
t.rep("""_, _, e2 = train('guard_save', F0 + ['--epochs', '1', '--save_epochs', '3-1'], ok=False)
assert 'not with --f2 1' in e1 and 'want a-b' in e2
print('[guards] --model ttr --f2 1 refused; --save_epochs 3-1 refused')""",
      """_, _, e2 = train('guard_save', F0 + ['--epochs', '1', '--save_epochs', '3-1'], ok=False)
_, _, e3 = train('guard_seed', ['--model', 'ttr'] + F0 + ['--epochs', '1'], ok=False)          # BJ-5: BASE seed 42
assert 'not with --f2 1' in e1 and 'want a-b' in e2 and 'seeds 1000-1999' in e3
print('[guards] --model ttr --f2 1 refused; --save_epochs 3-1 refused; --model ttr --seed 42 refused (BJ-5)')
# ---------------- TTR-6: a TTR state_dict does not strict-load into a plain KneeFleetNet
from fleet_model import KneeFleetNet
sd = torch.load(glob.glob(f'{d}/f0_s1042_ema_ttr224-*_best.pt')[0], weights_only=False)
ka = sd['args']
plain = KneeFleetNet(backbone=ka['backbone'], n_views=ka['n_views'], n_slices=ka['n_slices'], dim=ka['dim'],
                     n_contrast=ka['n_contrast'], contrast_emb=ka['contrast_emb'], pretrained=False, drop_path=0.0)
try:
    plain.load_state_dict(sd['model']); refused = False
except RuntimeError as ex:
    refused = 'ttr_size_t' in str(ex)
miss, unexp = plain.load_state_dict(sd['model'], strict=False)
assert refused and not miss and unexp == ['ttr_size_t'], (refused, miss, unexp)
print(f'[TTR-6] TTR ckpt into KneeFleetNet: strict load refused (unexpected {unexp}); raw/ema ttr_size_t int64 224')""")
t.rep("ttr_f0 = relabel_fp16(glob.glob(f'{d}/f0_s42_ema_ttr224-*_best.pt')[0], f'{MEM}/a/{os.path.basename(glob.glob(f\"{d}/f0_s42_ema_ttr224-*_best.pt\")[0])}')",
      "ttr_f0 = relabel_fp16(glob.glob(f'{d}/f0_s1042_ema_ttr224-*_best.pt')[0], f'{MEM}/a/{os.path.basename(glob.glob(f\"{d}/f0_s1042_ema_ttr224-*_best.pt\")[0])}')")
t.rep("ttr_all = relabel_fp16(f'{d2}/epochs/f-1_s42_ema_ttr224-64x24_ep2_best.pt', f'{MEM}/a/f-1_s42_ema_ttr224-64x24_ep2_best.pt')",
      "ttr_all = relabel_fp16(f'{d2}/epochs/f-1_s1042_ema_ttr224-64x24_ep2_best.pt', f'{MEM}/a/f-1_s1042_ema_ttr224-64x24_ep2_best.pt')")
t.rep("               '--tag', 'f0_s42ttr,f-1_s42ttr,=,=', ttr_f0, ttr_all, f1_f0, f1_all])",
      "               '--tag', 'f0_s1042ttr,=,=,=', ttr_f0, ttr_all, f1_f0, f1_all])")
t.rep("assert man['checkpoints'][0].startswith('f0_s42ttr_ema_ttr224-') and 'fleet_model_ttr.py' in os.listdir(DS_MIX)",
      "assert man['checkpoints'][0].startswith('f0_s1042ttr_ema_ttr224-') and 'fleet_model_ttr.py' in os.listdir(DS_MIX)\n"
      "assert man['checkpoints'][1].startswith('f-1_s1042_ema_ttr224-')\n"
      "assert all(set(v) == {'original', 'run_dir', 'sha256', 'model_mod', 'model_sha'} for v in man['sources'].values()), man['sources']   # FIX-5")
t.done()

# ---------------- t_gt2_saveep
t = R('tests/t_gt2_saveep.py')
print(t.s.count("'--seed', '42'"), t.s.count('s42'))
t.done()
