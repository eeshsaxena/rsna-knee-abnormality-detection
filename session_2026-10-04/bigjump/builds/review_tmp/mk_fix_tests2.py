# FIX test updates, part 2: t_b4 (FIX-5 manifest), t_gt2_saveep (BJ-5 seed), t_b5 (FIX-1 HALF, R-2, R-4, names),
# t_b1 (TTR-6 buffer), box_gp0 (BJ-5 seed)
import importlib.util, os
spec = importlib.util.spec_from_file_location('m', 'review_tmp/mk_fix_tests.py')
src = open('review_tmp/mk_fix_tests.py', encoding='utf-8').read().split('# ---------------- t_b2_ttr_train')[0]
ns = {}
exec(src, ns)
R = ns['R']

# ---------------- t_b4
t = R('tests/t_b4_pack_tag.py')
t.rep("assert man['sources'][nB]['original'] == os.path.basename(A) and man['sources'][nB]['src'].endswith('/runB/' + nA)",
      "assert man['sources'][nB]['original'] == os.path.basename(A) and man['sources'][nB]['run_dir'] == 'runB'   # FIX-5\n"
      "assert man['sources'][nB]['sha256'] == sha(B)[:16] and man['sources'][nA]['sha256'] == sha(A)[:16]\n"
      "assert ':' not in json.dumps(man['sources']).replace('\": ', '') and '/' not in json.dumps(man['sources']), man['sources']")
t.done()

# ---------------- t_gt2_saveep
t = R('tests/t_gt2_saveep.py')
t.rep("        '--device', 'cpu', '--seed', '42', '--backbone', 'convnext_atto.d2_in1k', '--model', 'ttr', '--aug', 'heavy',",
      "        '--device', 'cpu', '--seed', '1042', '--backbone', 'convnext_atto.d2_in1k', '--model', 'ttr', '--aug', 'heavy',")
t.done()

# ---------------- t_b5
t = R('tests/t_b5_tta_bench.py')
t.rep("#  (3) pass B of one study recomputed independently (mirror_views(sample_triplets(stack, 24, +.5))) == tta_bench's B",
      "#  (3) pass B of one study recomputed independently (mirror_views(sample_triplets(stack, 24, +HALF))) == tta_bench's B")
t.rep("#      --chunk 48 (the worker's) reported\n",
      "#      --chunk 48 (the worker's) reported\n"
      "#  (6) FIX-1 HALF = .5 - 1e-3: on a 24-slice stack P/N/B centres have no duplicate (exact +-.5 gives 0,2,2,4,..);\n"
      "#      for every other stack length 1..32 HALF == .5 except the N pass's last slot on even n (n-1, the tie gave n-2)\n"
      "#  (7) R-2 --passes A,B: per member none / A+B / A+B lesion-max, raw+EMA none / A+B, 2 G-TTA lines\n"
      "#  (8) R-4 --max_minutes cut before the first study: no crash, summary partial, gate lines say PARTIAL\n")
t.rep("mem = [f'{DS}/{n}' for n in man if n.startswith(('f0_s42ttr_', 'f0_s42_'))]",
      "mem = [f'{DS}/{n}' for n in man if n.startswith(('f0_s1042ttr_', 'f0_s42_'))]")
t.rep("x = torch.from_numpy(np.stack([fc.sample_triplets(g[0], fc.N_SLICES, 0.5) if g else np.zeros((24, 3, 64, 64), np.uint8)",
      "assert tb.HALF == 0.5 - 1e-3 and tb.PASSES['B'] == (tb.HALF, True) and tb.PASSES['N'] == (-tb.HALF, False)\n"
      "x = torch.from_numpy(np.stack([fc.sample_triplets(g[0], fc.N_SLICES, tb.HALF) if g else np.zeros((24, 3, 64, 64), np.uint8)")
t.rep("print(f'(3) pass B recomputed independently (mirror_views(sample_triplets(+.5))) vs tta_bench B: max|diff| {dB}')",
      "print(f'(3) pass B recomputed independently (mirror_views(sample_triplets(+HALF))) vs tta_bench B: max|diff| {dB}')")
t.rep("ck = glob.glob(f'{OUT}/ttr_members/a/f0_s42_ema_ttr224-*_best.pt')[0]\noof = f'{OUT}/ttr_f0/oof_f0_s42_ema.csv'",
      "ck = glob.glob(f'{OUT}/ttr_members/a/f0_s1042_ema_ttr224-*_best.pt')[0]\noof = f'{OUT}/ttr_f0/oof_f0_s1042_ema.csv'")
t.rep("""assert par[32]['n'] == 2 and par[32]['max_abs'] == 0.0
res.update(passB=dB, oof_parity=par)""", """assert par[32]['n'] == 2 and par[32]['max_abs'] == 0.0
# (6) FIX-1 HALF vs the exact .5 tie
st24 = np.arange(24, dtype=np.uint8)[:, None, None].repeat(4, 1).repeat(4, 2)
for off in (tb.HALF, -tb.HALF):
    c = fc.sample_triplets(st24, 24, off)[:, 1, 0, 0].astype(int)
    assert (np.diff(c) >= 1).all() and (c == np.arange(24)).all(), (off, c)
c5 = fc.sample_triplets(st24, 24, 0.5)[:, 1, 0, 0].astype(int)
assert (np.diff(c5) == 0).any(), c5                                    # the bug the fix avoids
dif = {}
for n in range(1, 33):
    stn = np.arange(n, dtype=np.uint8)[:, None, None].repeat(2, 1).repeat(2, 2)
    for sg in (1, -1):
        a = fc.sample_triplets(stn, 24, sg * 0.5)[:, 1, 0, 0].astype(int); b = fc.sample_triplets(stn, 24, sg * tb.HALF)[:, 1, 0, 0].astype(int)
        if (a != b).any():
            dif[(n, sg)] = [(int(k), int(a[k]), int(b[k])) for k in np.nonzero(a != b)[0]]
assert all(n == 24 or (sg == -1 and n % 2 == 0 and v == [(23, n - 2, n - 1)]) for (n, sg), v in dif.items()), dif
print(f'(6) HALF {tb.HALF}: n=24 P/N centres 0..23 (exact .5 gave {c5[:6].tolist()}..); other n: == .5 except the N '
      f'pass last slot on even n ({sum(1 for k in dif if k[0] != 24)} cases, n-1 instead of n-2)')
# (7) R-2: --passes A,B gives A+B and the gate
oAB = bench(f'{OUT}/b5_AB', ['--passes', 'A,B'] + mem)
sAB = json.load(open(f'{OUT}/b5_AB/tta_summary.json'))
for p in mem:
    assert set(sAB['results'][os.path.basename(p)]) == {'none', 'A+B', 'A+B lesion-max'}, sAB['results'].keys()
assert set(sAB['results']['raw+EMA']) == {'none', 'A+B'} and len(sAB['gate']) == 2 and not sAB['partial']
for p in mem:                                                           # same numbers as the 5-pass run (A and B)
    n = os.path.basename(p)
    assert sAB['results'][n]['A+B']['val'] == s24['results'][n]['A+B']['val'], n
    assert sAB['gate'][n] == s24['gate'][n], n
gl = [l for l in oAB.splitlines() if l.startswith('G-TTA')]
assert len(gl) == 2, oAB[-800:]
print(f'(7) --passes A,B: variants {sorted(sAB["results"][os.path.basename(mem[0])])}, raw+EMA A+B, gate == 5-pass run: ' + ' | '.join(gl)[:300])
# (8) R-4: a cut before the first study
oC = bench(f'{OUT}/b5_cut', ['--passes', 'A,B', '--max_minutes', '1e-9'] + mem)
sC = json.load(open(f'{OUT}/b5_cut/tta_summary.json'))
assert sC['partial'] and sC['n_val'] == 0 and 'max_minutes' in oC and 'PARTIAL' in oC, oC[-600:]
print(f'(8) --max_minutes cut: partial {sC["partial"]}, n_val {sC["n_val"]}, gate lines flagged PARTIAL; no crash')
res.update(passB=dB, oof_parity=par, half_diff={f'{k[0]},{k[1]}': v for k, v in dif.items()})""")
t.done()

# ---------------- t_b1
t = R('tests/t_b1_model.py')
t.rep("#      drop_path / dropout) and every gradient, max |diff| 0; state_dict keys identical (no new parameter/buffer)",
      "#      drop_path / dropout) and every gradient, max |diff| 0; state_dict keys identical except TTR-6's persistent\n"
      "#      int64 buffer ttr_size_t (no new parameter); (e) TTR-6: the buffer survives channels_last / .to / the\n"
      "#      trainer's EMA (AveragedModel + get_ema_multi_avg_fn, 5 updates) as int64 == ttr_size, and a TTR state_dict\n"
      "#      does NOT strict-load into a plain KneeFleetNet (unexpected key ttr_size_t)")
t.rep("assert list(a[3]) == list(b[3]), 'TTR adds/removes state_dict keys'",
      "assert [k for k in b[3] if k != 'ttr_size_t'] == list(a[3]) and int(b[3]['ttr_size_t']) == 64, 'TTR state_dict keys'")
t.rep("      f'{len(a[3])} state_dict keys identical')",
      "      f'{len(a[3])} state_dict keys identical + ttr_size_t')")
t.rep("""res['resize'] = dict(encode_manual=d_enc, backbone_in=seen[0])""",
      """res['resize'] = dict(encode_manual=d_enc, backbone_in=seen[0])
# (e) TTR-6
torch.manual_seed(0)
mt = KneeFleetNetTTR(ttr_size=224, **kw).to(memory_format=torch.channels_last)
ema = torch.optim.swa_utils.AveragedModel(mt, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9995))
opt = torch.optim.AdamW(mt.parameters(), lr=1e-2)
for _ in range(5):
    lo = mt(x, pr, ct); lo.float().pow(2).mean().backward(); opt.step(); opt.zero_grad(); ema.update_parameters(mt)
for mm in (mt, ema.module):
    sd = mm.state_dict()
    assert sd['ttr_size_t'].dtype == torch.int64 and int(sd['ttr_size_t']) == 224 == mm.ttr_size, sd['ttr_size_t']
moved = any(not torch.equal(p, q) for p, q in zip(mt.parameters(), ema.module.parameters()))
plain = KneeFleetNet(**kw)
try:
    plain.load_state_dict(ema.module.state_dict()); refused = False
except RuntimeError as ex:
    refused = 'ttr_size_t' in str(ex)
back = KneeFleetNetTTR(ttr_size=224, **kw); back.load_state_dict(ema.module.state_dict())
assert refused and moved and int(back.ttr_size_t) == 224
print(f'(e) TTR-6: ttr_size_t int64 224 in raw and EMA after 5 EMA updates (params moved {moved}); strict load into '
      f'KneeFleetNet refused; TTR->TTR strict load ok')
res['ttr6'] = dict(refused=refused)""")
t.done()

# ---------------- box_gp0
t = R('box/box_gp0.py')
t.rep("              '--workers', '0', '--device', 'cpu', '--seed', '42', '--min_cover', '0', '--out', f'{OUT}/run'], f'{OUT}/train.log')",
      "              '--workers', '0', '--device', 'cpu', '--seed', '1042', '--min_cover', '0', '--out', f'{OUT}/run'], f'{OUT}/train.log')")
t.rep("    p = glob.glob(f'{OUT}/run/f0_s42_{tag}_ttr224-320x24_*_best.pt')",
      "    p = glob.glob(f'{OUT}/run/f0_s1042_{tag}_ttr224-320x24_*_best.pt')")
t.rep("    a = ck['args']\n", "    a = ck['args']\n    assert int(ck['model']['ttr_size_t']) == 224 == a['ttr_size']                 # TTR-6\n")
t.rep("""                '--threads', '2', '--out', f'{OUT}/tb_c{chunk}', '--oof', f'{OUT}/run/oof_f0_s42_ema.csv',
                f'{OUT}/run/oof_f0_s42_raw.csv', '--'] + mem, f'{OUT}/tb_c{chunk}.log')""",
      """                '--threads', '2', '--out', f'{OUT}/tb_c{chunk}', '--oof', f'{OUT}/run/oof_f0_s1042_ema.csv',
                f'{OUT}/run/oof_f0_s1042_raw.csv', '--'] + mem, f'{OUT}/tb_c{chunk}.log')""")
t.done()
