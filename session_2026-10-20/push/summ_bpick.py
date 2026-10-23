# summ_bpick.py (10-20 push): sensitivity matrix from bpick_<tag>.json (MODEL). Prints, per row and per A (eoa20 now / A21 R1 winner),
# P(silver) and P(r<=167) of every B, the best B, and R6e20 minus R6 / minus the best non-R6e20 B (pts).
import json, glob, os
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push'
TAGS = ['base', 'p_more', 'p_ladder', 'p_orig', 'p_null', 'p_neg', 'noise1.5', 'k_skep', 'k_pess', 'k_opt', 'k_pess_noise1.5', 'rho0.6', 'rho0.9',
        'o_skepS6', 'o_strongS6', 'o_tabReal', 'o_tabOver', 'o_r6better', 'o_noise1.5', 'o_bigKnobs', 'o_smallKnobs',
        'f_fieldBad', 'f_fieldGood', 'f_Wbig', 'g_low', 'g_high', 'u_flat', 'd30_opt', 'd30_pess']
BS = ['R6u', 'R6e20', 'R6e20g', 'c4', 'eoa10', 'HW', 'S6F', 'a25']
for Aname in ('A_eoa20', 'A_A21'):
    print(f'\n== {Aname}: P(silver) by B (pts x100); last cols: best B, R6e20 - R6, R6e20 - best other (excl. S6F), R6e20g - R6, P(R6e20 shows .957)')
    print('%-16s %6s %6s ' % ('row', 'd20', 'k') + ' '.join('%7s' % b for b in ['alone'] + BS) + '   best     e-R6  e-oth  g-R6  P957')
    for t in TAGS:
        p = f'{D}/bpick_{t}.json'
        if not os.path.isfile(p): continue
        J = json.load(open(p)); S = J[Aname]
        sil = {b: S[b]['silver'] for b in BS if b in S}
        oth = max(v for b, v in sil.items() if b not in ('R6e20', 'R6e20g', 'S6F'))
        best = max(sil, key=sil.get)
        print('%-16s %+6.2f %6.2f ' % (t, J['post']['d20'][0] * 1e4, J['post']['k'][0]) + ' '.join('%7.1f' % (100 * S[b]['silver']) for b in ['A alone'] + BS) +
              '   %-8s %+5.1f %+6.1f %+5.1f %.3f' % (best, 100 * (sil['R6e20'] - sil['R6u']), 100 * (sil['R6e20'] - oth), 100 * (sil['R6e20g'] - sil['R6u']),
                                              J['post']['R6e20_disp']['957']))
    print(f'-- {Aname}: P(private rank <= 167, the projected .960+ band) by B')
    for t in TAGS:
        p = f'{D}/bpick_{t}.json'
        if not os.path.isfile(p): continue
        J = json.load(open(p)); S = J[Aname]
        r = {b: S[b]['r167'] for b in BS if b in S}
        print('%-16s ' % t + ' '.join('%7.1f' % (100 * S[b]['r167']) for b in ['A alone'] + BS) + '   best %-7s e-R6 %+5.1f' % (max(r, key=r.get), 100 * (r['R6e20'] - r['R6u'])))
