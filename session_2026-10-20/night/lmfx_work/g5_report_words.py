# g5_report_words.py (10-20 night, Track 3, Claude). RESEARCH ONLY, local CPU, competition train reports only (no OAI).
# Rough multilingual keyword census of train reports: what fracture TYPES the report-positive Fracture studies are, and how
# LM wording (degeneration vs tear) splits; then the R7 OOF Fracture / LM output by type (pct within train), and the expert
# label on the gold58 studies with that wording. Regexes are GUESS-level (no negation handling except 'no/without fracture').
import sys, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G, R, labels
t = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rep = t['Report'].fillna('').astype(object).str.lower().astype(object)
K = {
 'insufficiency/SIFK/SONK/stress': r'insuffic|insuficien|insuffizien|insuffici\u00ebn|yetmezlik|\u03b1\u03bd\u03b5\u03c0\u03ac\u03c1\u03ba\u03b5\u03b9|\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u044a\u0447\u043d|insuficijenc|sifk|sonk|stress fract|fractura de estr|stres k\u0131r\u0131\u011f|stressfrakt|osteonecros|necrosis avascular|spontaneous osteonec|\u03bf\u03c3\u03c4\u03b5\u03bf\u03bd\u03ad\u03ba\u03c1\u03c9\u03c3',
 'microtrabecular/trabecular': r'trabecul|microfract|microfrakt|\u03bc\u03b9\u03ba\u03c1\u03bf\u03b4\u03bf\u03ba\u03b9\u03b4|mikrofrakt|trabek',
 'avulsion/Segond/spine': r'avuls|avulz|segond|ausriss|eminenti|eminence|espinas tibial|\u03b1\u03c0\u03bf\u03c3\u03c0\u03b1\u03c3\u03c4\u03b9\u03ba|\u0430\u0432\u0443\u043b\u0441|kopma k\u0131r',
 'impaction/osteochondral/notch': r'impact|impaksiyon|impresij|impression|osteochond|osteocondral|osteohondral|\u043e\u0441\u0442\u0435\u043e\u0445\u043e\u043d\u0434\u0440|\u03bf\u03c3\u03c4\u03b5\u03bf\u03c7\u03bf\u03bd\u03b4\u03c1|notch sign|depresi|depresy|\u03b5\u03bd\u03c3\u03c6\u03ae\u03bd\u03c9\u03c3',
 'plateau / condyle fracture': r'(plat|plateau|kondil|condil|condyl|\u03ba\u03cc\u03bd\u03b4\u03c5\u03bb|\u043a\u043e\u043d\u0434\u0438\u043b|\u043f\u043b\u0430\u0442\u043e).{0,60}(fract|frakt|k\u0131r\u0131k|prijelom|\u03ba\u03ac\u03c4\u03b1\u03b3\u03bc|\u0444\u0440\u0430\u043a\u0442\u0443\u0440|\u0441\u0447\u0443\u043f)|(fract|frakt|k\u0131r\u0131k|prijelom|\u03ba\u03ac\u03c4\u03b1\u03b3\u03bc|\u0444\u0440\u0430\u043a\u0442\u0443\u0440).{0,60}(plat|plateau|kondil|condil|condyl|\u03ba\u03cc\u03bd\u03b4\u03c5\u03bb|\u043a\u043e\u043d\u0434\u0438\u043b|\u043f\u043b\u0430\u0442\u043e)',
 'old/healed/chronic fracture': r'old fract|healed|healing fract|chronic fract|consolidad|konsolid|eski k\u0131r\u0131k|\u03c0\u03b1\u03bb\u03b1\u03b9|\u0441\u0442\u0430\u0440\u0430 \u0444\u0440\u0430\u043a\u0442|posttraumat|post-traumat|alte fraktur|zarasl|knochenheil',
}
y2 = labels('v2'); yF = y2['Fracture']; yL = y2['Lateral Meniscus']
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
g, g58 = gold(); gid = set(g.StudyInstanceUID)
tr = [u for u in t.index if u not in gid and u in r7.index]
pF = pd.Series(pct(r7.loc[tr, 'p|Fracture'].values), index=tr); pL = pd.Series(pct(r7.loc[tr, 'p|Lateral Meniscus'].values), index=tr)
pos = yF.reindex(tr) == 1
print('train (non-gold, R7 OOF) n %d | report v2 Fracture YES %d, NO %d, UNK %d' % (len(tr), pos.sum(), (yF.reindex(tr) == 0).sum(), yF.reindex(tr).isna().sum()))
print('Fracture-type keyword share among report-YES fractures, and R7 OOF Fracture pct (mean) per type:')
for k, rx in K.items():
    m = rep.reindex(tr).astype(object).str.contains(rx, regex=True)
    print('  %-32s YES-fracture share %.3f (n %3d) | mean R7 pct when YES %.3f | when keyword but report NO/UNK: n %3d mean pct %.3f' % (
        k, (m & pos).sum() / max(pos.sum(), 1), (m & pos).sum(), pF[m & pos].mean() if (m & pos).any() else np.nan, (m & ~pos).sum(), pF[m & ~pos].mean() if (m & ~pos).any() else np.nan))
print('  report-YES fracture, none of the keywords: n %d mean pct %.3f' % ((pos & ~np.any([rep.reindex(tr).astype(object).str.contains(rx, regex=True) for rx in K.values()], 0)).sum(),
      pF[pos & ~np.any([rep.reindex(tr).astype(object).str.contains(rx, regex=True) for rx in K.values()], 0)].mean()))
# gold58: expert Fracture by keyword
g58i = g58.set_index('StudyInstanceUID'); rg = rep.reindex(g58i.index).astype(object)
print('gold58: expert Fracture prevalence by report keyword (n, expert pos)')
for k, rx in K.items():
    m = rg.str.contains(rx, regex=True).values
    print('  %-32s n %2d expert Fracture+ %2d | expert Contusion+ %2d' % (k, m.sum(), g58i['Fracture'].values[m].sum(), g58i['Contusion'].values[m].sum()))
# LM wording: degeneration-only vs tear words near 'lateral' meniscus (very rough; English + Spanish + Turkish only)
LMX = {'en': (r'lateral menisc', r'tear', r'degenerat|signal'), 'es': (r'menisco (externo|lateral)', r'rotura', r'degenera|se\u00f1al'),
       'tr': (r'lateral menisk', r'y\u0131rt\u0131k', r'dejenera|grade (i|ii|1|2)\b')}
print('LM wording (en/es/tr reports, sentence-level co-occurrence, GUESS-level): R7 OOF LM pct by report v2 LM verdict x wording')
for lang, (lm, tw, dw) in LMX.items():
    sub = [u for u in tr if t.loc[u, 'lang'] == lang]
    sents = rep.reindex(sub).str.split(r'[.\n]')
    hasT = sents.apply(lambda ss: any(re.search(lm, s) and re.search(tw, s) for s in ss))
    hasD = sents.apply(lambda ss: any(re.search(lm, s) and re.search(dw, s) and not re.search(tw, s) for s in ss))
    v = yL.reindex(sub)
    for nm, m in (('tear-word', hasT & ~hasD), ('degeneration-word only', hasD & ~hasT), ('both', hasT & hasD), ('neither', ~hasT & ~hasD)):
        print('  %s %-24s n %4d | v2 YES %.2f | R7 LM pct mean %.3f' % (lang, nm, m.sum(), (v[m] == 1).mean(), pL.reindex(sub)[m].mean()))
