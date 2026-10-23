# local_fluid.py (10-20 levers / rubric lens, Claude; research only, CPU, competition data only, no OAI)
# Lever T-VOL: the rubric grades Effusion and Baker's positive only when MODERATE or LARGE; every member's training labels
# count small ones (census_lang.log: 55 % of Effusion and 44 % of Baker's EN training positives are small-only). R6 measures a
# fluid volume per study (fluid_stacker.csv: eff_sag_p99_open / bak_sag_p99_open, mL) and its stacker adds it in logit space
# with coefficients fitted on REPORT labels. Transform: out = pct(score) + b * pct(volume) (missing volume -> .5), b in B.
# Rulers here (all on competition data):
#   (1) report regime: v4 / jev labels >= .5 (small counted) on R7 OOF + S6 seed OOF (n ~4.1k)   -> mechanism control
#   (2) rubric-mapped report text (EN only, regex from census_lang.py): 1 = moderate / large stated, 0 = none or small only
#   (3) gold58 expert labels with the exact R6 run_b fluid csv and the c4 gold58 reconstruction (consistency only, NOT selection)
import re, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, s6seed
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers')
import importlib.util
BS = (0., .1, .2, .3, .5, 1.)
rng = np.random.default_rng(1020)


def tvol(s, v, b):
    pv = np.full(len(v), .5); ok = np.isfinite(v); pv[ok] = pct(v[ok])
    return pct(s) + b * pv


def boot_delta(s, v, y, b, B=1000):
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        d.append(auc(tvol(s[i], v[i], b), y[i]) - auc(s[i], y[i]))
    d = np.array(d); return d.mean(), d.std(), (d > 0).mean()

# ---------------- train-side data ----------------
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
mem = {'R7': r7[['p|' + c for c in L]].set_axis(L, axis=1)}
for sd in (1044, 1045):
    try:
        d, fs = s6seed(sd); mem['s%d' % sd] = d
    except Exception as e:
        print('seed', sd, 'missing', repr(e)[:80])
ids = r7.index
for k in list(mem):
    ids = ids.intersection(mem[k].dropna().index)
print('train OOF ids common', len(ids), 'members', list(mem))
mem['R7m'] = r7[['fluid|Effusion|p_model']].rename(columns={'fluid|Effusion|p_model': 'Effusion'}).assign(**{"Baker's": r7["stk|Baker's|p_model"]})
VOL = {'Effusion': r7['fluid|eff_sag_p99_open'], "Baker's": r7['fluid|bak_sag_p99_open']}


def lab(src, c):
    f = {'v4': R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
         'jev': R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv'}[src]
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    return (d[c] >= .5).astype(float).where(d[c].notna())

# rubric-mapped report-text labels (EN), reusing census_lang regex helpers
spec = importlib.util.spec_from_file_location('cl', 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/census_lang.py')
src_txt = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/census_lang.py', encoding='utf-8').read()
helpers = src_txt.split('CATS = []')[0]
ns = {}
exec(helpers, ns)   # defines lang, en, sents, pos_hit, SS, EFF etc. (prints the EN count line)
SS, pos_hit = ns['SS'], ns['pos_hit']
EFF = r"(effusion|joint fluid|intra-?articular fluid|fluid in the joint)"
SMALL = r"(small|trace|minimal|mild|slight|physiolog|little|minor|tiny)"
BIG = r"(moderate|large|massive|marked|extensive|considerable|significant)"
BAK = r"(baker|popliteal cyst|semimembranosus[^.]{0,30}(bursa|cyst)|gastrocnemi[^.]{0,30}(bursa|cyst))"


def rub_eff(ss):
    big = any(re.search(EFF, s) and pos_hit(s, BIG) for s in ss)
    small = any(re.search(EFF, s) and pos_hit(s, SMALL) for s in ss)
    anyeff = any(pos_hit(s, r"\beffusion") for s in ss)
    none_ = any(re.search(r"\b(no|without)\b[^.]{0,30}(joint )?effusion", s) for s in ss)
    if big:
        return 1.
    if small or (none_ and not anyeff):
        return 0.
    if not anyeff:
        return 0.   # effusion never mentioned positively -> none
    return np.nan  # positive but size unstated


def rub_bak(ss):
    big = any(re.search(BAK, s) and pos_hit(s, r"(moderate|large|big|sizable|sizeable|giant)") for s in ss)
    small = any(re.search(BAK, s) and pos_hit(s, r"(small|tiny|minimal|minute|little|mild)") for s in ss)
    pres = any(re.search(BAK, s) and pos_hit(s, BAK, 30) for s in ss)
    if big:
        return 1.
    if small:
        return 0.
    if not pres:
        return 0.
    return np.nan

RUB = {'Effusion': pd.Series({u: rub_eff(ss) for u, ss in SS.items()}),
       "Baker's": pd.Series({u: rub_bak(ss) for u, ss in SS.items()})}
for c in RUB:
    v = RUB[c]; print(f'rubric-mapped EN {c}: n {int(v.notna().sum())} pos {int((v == 1).sum())} neg {int((v == 0).sum())} unstated {int(v.isna().sum())}')

print('\n== TRAIN (OOF) rulers: single-label AUC delta of out = pct(s) + b pct(vol); rows = member x ruler; cols = b', BS[1:])
for c in ['Effusion', "Baker's"]:
    for ruler in ['v4', 'jev', 'RUBRIC_EN', 'v4_EN']:
        if ruler == 'RUBRIC_EN':
            y_all = RUB[c]
        elif ruler == 'v4_EN':
            y_all = lab('v4', c).reindex(RUB[c].index)
        else:
            y_all = lab(ruler, c)
        for m in ['R7', 'R7m', 's1044', 's1045', 'mean3']:
            if m == 'mean3':
                s_all = sum(pd.Series(pct(mem[k].loc[ids, c].values), index=ids) for k in ('R7', 's1044', 's1045')) / 3
            else:
                if m not in mem:
                    continue
                s_all = mem[m][c]
            ii = ids.intersection(y_all.dropna().index).intersection(s_all.dropna().index)
            y = y_all.loc[ii].values.astype(int); s = s_all.loc[ii].values; v = VOL[c].reindex(ii).values.astype(float)
            base = auc(s, y)
            ds = [auc(tvol(s, v, b), y) - base for b in BS[1:]]
            line = f'{c:9s} {ruler:9s} {m:6s} n {len(y):5d} prev {y.mean():.3f} base {base:.4f} | ' + ' '.join(f'{d:+.4f}' for d in ds)
            if m in ('R7', 'mean3') and ruler in ('RUBRIC_EN', 'v4_EN', 'v4'):
                bm, bsd, bp = boot_delta(s, v, y, .2, 500)
                line += f' | b.2 boot {bm:+.4f} sd {bsd:.4f} P {bp:.2f}'
            print(line)
        vv = VOL[c].reindex(y_all.dropna().index).values.astype(float); yy = y_all.dropna().values.astype(int); ok = np.isfinite(vv)
        print(f'   volume alone vs {ruler}: AUC {auc(vv[ok], yy[ok]):.4f} (n {ok.sum()}, missing {int((~ok).sum())})')

# ---------------- gold58 consistency (expert labels; NOT a selection ruler) ----------------
g, g58 = gold(); gids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int)
fl = pd.read_csv(G + '/r6g58_box/run_b/worker0/fluid_stacker.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(gids)
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}


def rdg(path):
    d = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    return d.reindex(gids)[L].astype(float).values
R6g = rdg(G + '/R6_gold58_serve.csv'); S6g = rdg(G + '/n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4P = np.zeros((58, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j])
    b = w * rr + (1 - w) * pct(R6g[:, j]); C4P[:, j] = .8 * pct(b) + .2 * pct(S6g[:, j])
print('\n== gold58 (expert; consistency only) c4 macro %.5f (expect .94102)' % np.mean([auc(C4P[:, j], Y[:, j]) for j in range(12)]))
for c, vc in (('Effusion', 'eff_sag_p99_open'), ("Baker's", 'bak_sag_p99_open')):
    j = L.index(c); v = fl[vc].values.astype(float); y = Y[:, j]
    print(f'{c}: pos {y.sum()} | volume alone AUC {auc(np.nan_to_num(v, nan=np.nanmedian(v)), y):.4f} | R6 {auc(R6g[:, j], y):.4f} c4 {auc(C4P[:, j], y):.4f}')
    for nm, s in (('c4', C4P[:, j]), ('R6', R6g[:, j]), ('S6fall', S6g[:, j]), ('r384', R384[:, j])):
        print(f'   {nm:6s} ' + ' '.join(f'b{b}: {auc(tvol(s, v, b), y) - auc(s, y):+.4f}' for b in BS[1:]))
