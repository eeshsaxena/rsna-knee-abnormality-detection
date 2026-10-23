p = 'code/tta_bench.py'
s = open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)
rep("# slice offset +0.5) beat pass A alone, measured through the WORKER path on fleet checkpoints?",
    "# slice offset +HALF) beat pass A alone, measured through the WORKER path on fleet checkpoints?")
rep("# Passes per member (one decode per study): A off 0 | M mirror | P off +.5 | N off -.5 | B mirror + off +.5\n",
    "# Passes per member (one decode per study): A off 0 | M mirror | P off +HALF | N off -HALF | B mirror + off +HALF\n"
    "#   HALF = 0.5 - 1e-3, not 0.5 (FIX-1): fleet_common.sample_triplets rounds half to even (ndarray.round), so an EXACT\n"
    "#   +-0.5 on a 24-slice stack (linspace(0, 23, 24) = integers; ~5.5 % of stored views, ~12 % of studies) gives centres\n"
    "#   0,2,2,4,4,.. (odd slices dropped, even ones doubled), a view training never produces. HALF rounds exactly like 0.5\n"
    "#   wherever 0.5 is not a tie (fractional parts are multiples of 1/(n_slices-1) = 1/23 >> 1e-3) and leaves n = 24 at\n"
    "#   offset 0; the only other change is the N pass's LAST slot on even n (n-1 instead of the tie's n-2). B6 (the\n"
    "#   worker's pass B) must use the same HALF. sample_triplets itself is contract code (tensor_sha): unchanged.\n")
rep("PASSES = {'A': (0.0, False), 'M': (0.0, True), 'P': (0.5, False), 'N': (-0.5, False), 'B': (0.5, True)}",
    "HALF = 0.5 - 1e-3                                    # FIX-1 (see the header): never an exact .5 tie in sample_triplets\n"
    "PASSES = {'A': (0.0, False), 'M': (0.0, True), 'P': (HALF, False), 'N': (-HALF, False), 'B': (HALF, True)}")
rep("""    miss, unexp = m.load_state_dict(ck['model'], strict=False)
    assert not miss and not unexp, (p, miss[:3], unexp[:3])
    m = m.eval().to(dev)""", """    miss, unexp = m.load_state_dict(ck['model'], strict=False)
    assert not miss and not unexp, (p, miss[:3], unexp[:3])
    if mod == 'fleet_model_ttr':                                             # == fleet_worker W-TTR / TTR-6
        assert int(m.ttr_size_t) == int(a['ttr_size']) == m.ttr_size, (p, 'ttr_size buffer != args', int(m.ttr_size_t))
    m = m.eval().to(dev)""")
rep("""#   then the same without --n (867 val + 58 gold; ~0.3-0.5 s per study-pass per member on a shared V100: GUESS)
""", """#   then the gate run, cheapest form: --passes A,B --max_minutes 20 (A+B, A+B lesion-max, raw+EMA A+B and the G-TTA line
#   need only A and B; the full 5-pass run = 925 studies x 5 passes x 2 members ~ 45-75 min of 3rd-job GPU: GUESS)
#   R-3 GPU guard: refuses below --min_free_gb (4) free and caps THIS process at --max_gpu_gb (3) GB, so an over-budget
#   bench OOMs itself, never a running trainer. R-4 --max_minutes N stops the pass loop and scores the studies done
#   (val and gold58 interleaved, so a cut keeps both sets; the gate line then says PARTIAL)
""")
rep("""    summ = dict(n_val=len(val), n_gold58=len(g58), passes=passes, members=[i for _, i in members], results={}, gate={})""",
    """    summ = dict(n_val=len(val), n_gold58=len(g58), passes=passes, members=[i for _, i in members], results={}, gate={},
                partial=n_run < len(sids), half=HALF)""")
rep("""        print(f'G-TTA {cname}: A+B - none = {g["val_delta"]:+.4f} val (P>0 {g["p_gt0"]:.2f}), gold58 {g["gold58_delta"]:+.4f} -> {g["verdict"]}')""",
    """        print(f'G-TTA {cname}: A+B - none = {g["val_delta"]:+.4f} val (P>0 {g["p_gt0"]:.2f}), gold58 {g["gold58_delta"]:+.4f} -> {g["verdict"]}'
              + (f' [PARTIAL: {summ["n_val"]} val + {summ["n_gold58"]} gold58 studies, --max_minutes cut]' if summ['partial'] else ''))""")
open(p, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
print('ok')
