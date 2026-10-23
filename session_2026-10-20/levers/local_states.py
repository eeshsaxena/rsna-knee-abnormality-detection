# local_states.py (10-20 levers / rubric lens, Claude; research only, CPU, competition data only, no OAI)
# Lever T-STATE: rubric-mapped heliosli state read-out (weights W pre-registered 10-19 in trick/regime_work/m2_states.py, copied
# unchanged) vs the binary head, on a NEW ruler: report text re-thresholded at the RUBRIC (EN reports, regex, census_lang.py).
# Prediction if the regime mechanism is real AND the state heads carry it: rubric-mapped / mix gains on the RUBRIC ruler and
# loses (or gains less) on the report-threshold ruler (v4 >= .5) for the SAME studies. R7 OOF only (R6's states are not dumped
# anywhere: not in the c4 kernel, not on OAI).
import re, sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, G, R
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/regime_work/m2_states.py', encoding='utf-8').read()
W = eval(src.split('W = ', 1)[1].split('\nSTRICT', 1)[0])
ns = {}
exec(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/census_lang.py', encoding='utf-8').read().split('CATS = []')[0], ns)
SS, pos_hit, side_sent, men = ns['SS'], ns['pos_hit'], ns['side_sent'], ns['men']
TEAR = ns['TEAR']; ACL_S = ns['ACL_S']; MCL_S = ns['MCL_S']; any_s = ns['any_s']
rng = np.random.default_rng(1020)
d = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str})
gold_ids = set(tr.loc[tr['ACL'].notna(), 'StudyInstanceUID'])

# ---- rubric-mapped text labels (1 / 0 / NaN = ambiguous, excluded) ----
EFF = r"(effusion|joint fluid|intra-?articular fluid|fluid in the joint)"
SMALL = r"(small|trace|minimal|mild|slight|physiolog|little|minor|tiny)"
BIG = r"(moderate|large|massive|marked|extensive|considerable|significant)"
BAK = r"(baker|popliteal cyst|semimembranosus[^.]{0,30}(bursa|cyst)|gastrocnemi[^.]{0,30}(bursa|cyst))"


def r_eff(ss):
    if any(re.search(EFF, s) and pos_hit(s, BIG) for s in ss): return 1.
    if any(re.search(EFF, s) and pos_hit(s, SMALL) for s in ss): return 0.
    if not any(pos_hit(s, r"\beffusion") for s in ss): return 0.
    return np.nan


def r_bak(ss):
    if any(re.search(BAK, s) and pos_hit(s, r"(moderate|large|big|sizable|sizeable|giant)") for s in ss): return 1.
    if any(re.search(BAK, s) and pos_hit(s, r"(small|tiny|minimal|minute|little|mild)") for s in ss): return 0.
    if not any(re.search(BAK, s) and pos_hit(s, BAK, 30) for s in ss): return 0.
    return np.nan


def r_acl(ss):
    if any_s(ss, ACL_S, r"(complete|full[- ]thickness|rupture|discontinu|non-?visuali|high[- ]grade)"): return 1.
    if any_s(ss, ACL_S, r"(chronic|old|remote)") and any_s(ss, ACL_S, TEAR): return 1.
    if any_s(ss, ACL_S, r"partial") and not any_s(ss, ACL_S, r"low[- ]grade"): return np.nan
    if any_s(ss, ACL_S, r"\btear|torn") and not any_s(ss, ACL_S, r"(low[- ]grade|sprain|partial)"): return np.nan  # tear, grade unstated
    return 0.


def r_mcl(ss):
    if any_s(ss, MCL_S, r"(grade (3|iii)\b|complete|full[- ]thickness|high[- ]grade|rupture)"): return 1.
    if any_s(ss, MCL_S, r"(grade (2|ii)\b|partial|tear|torn)"): return np.nan
    return 0.


def r_men(side):
    tear, degen, postop, extr = men(side)
    return lambda ss: 1. if (tear(ss) or postop(ss)) else 0.


def r_cont(ss):
    if any(pos_hit(s, r"(contusion|bruis)") for s in ss): return 1.
    BME = r"(bone marrow (edema|oedema)|marrow (edema|oedema)|bone (edema|oedema)|edema-like|oedema-like)"
    if any(pos_hit(s, BME) and not re.search(r"(degenerat|subchondral|reactive|stress|arthr|osteoarth|chondr|cyst|insufficien)", s) for s in ss): return np.nan
    return 0.


def r_frac(ss):
    if any(pos_hit(s, r"(acute|impact|depress|displac|plateau|comminut|nondisplac|non-displac|undisplac|avulsion|segond)[^.]{0,30}fractur|fractur[^.]{0,30}(acute|impact|depress|displac|plateau)|segond") for s in ss): return 1.
    if any(pos_hit(s, r"(insufficien|sifk|subchondral fractur|stress fractur)") for s in ss): return np.nan
    if any(pos_hit(s, r"(?<!micro)(?<!micro-)fractur") for s in ss): return np.nan
    return 0.

RUB = {'Effusion': r_eff, "Baker's": r_bak, 'ACL': r_acl, 'MCL': r_mcl, 'Medial Meniscus': r_men('medial'),
       'Lateral Meniscus': r_men('lateral'), 'Contusion': r_cont, 'Fracture': r_frac}
YR = pd.DataFrame({c: pd.Series({u: f(ss) for u, ss in SS.items()}) for c, f in RUB.items()})
YR = YR.drop(index=[u for u in YR.index if u in gold_ids])
v4 = pd.read_csv(R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
jev = pd.read_csv(R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')


def sc(df, c):
    b = df['p|' + c].values
    r = sum(df['state|%s|%s' % (c, s)].values * w for s, w in W[c].items())
    return b, r


def bootd(b, r, y, B=1000):
    n = len(y); o = {'rub': [], 'mix': []}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        a0 = auc(b[i], y[i]); o['rub'].append(auc(r[i], y[i]) - a0); o['mix'].append(auc(.5 * pct(b[i]) + .5 * pct(r[i]), y[i]) - a0)
    return {k: (np.mean(v), np.std(v), np.mean(np.array(v) > 0)) for k, v in o.items()}

print('R7 OOF, non-gold EN train studies. rub = rubric-mapped state score (W of m2, 10-19); mix = .5 pct(bin) + .5 pct(rub)')
print('ruler RUBRIC = report text re-thresholded at the rubric (regex); ruler REPORT = v4 >= .5 on the SAME studies; JEV = jev >= .5')
for c in RUB:
    y_all = YR[c].dropna()
    ii = y_all.index.intersection(d.index)
    b, r = sc(d.loc[ii], c)
    for nm, yy in (('RUBRIC', y_all.loc[ii].values.astype(int)), ('REPORT', (v4.reindex(ii)[c] >= .5).astype(int).values),
                   ('JEV', (jev.reindex(ii)[c] >= .5).astype(int).values)):
        if yy.min() == yy.max(): continue
        a0 = auc(b, yy); bt = bootd(b, r, yy)
        print(f'{c:16s} {nm:6s} n {len(yy):5d} pos {yy.sum():4d} bin {a0:.4f} | rub {auc(r, yy) - a0:+.4f} (P {bt["rub"][2]:.2f}) '
              f'| mix {auc(.5 * pct(b) + .5 * pct(r), yy) - a0:+.4f} (sd {bt["mix"][1]:.4f} P {bt["mix"][2]:.2f})')
