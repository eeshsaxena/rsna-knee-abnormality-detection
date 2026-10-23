# census_lang.py (10-20 levers / rubric lens, Claude; research only, CPU, competition data only, no OAI)
# Report-language census: for each label, rubric-edge phrases in the ENGLISH train reports (regex, sentence level, simple
# negation window), and how the LLM training-label sets that the c4 members learned from treat each phrase class.
# Output: per category n studies (EN), share labelled >= .5 by each label source, and the rubric's expected call.
# Also: share of each source's positives that sit in a rubric-NEGATIVE category ("regime contamination").
import re, sys, numpy as np, pandas as pd
R = 'C:/Users/eeshs/Downloads/rsna-knee'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str})
tr = pd.read_csv(R + '/_data/train.csv', dtype={'StudyInstanceUID': str})
gold_ids = set(tr.loc[tr['ACL'].notna(), 'StudyInstanceUID'])
en = lang[lang.lang == 'en'].copy()
print('EN reports', len(en), 'of', len(lang), '| gold58 among EN', len(gold_ids & set(en.StudyInstanceUID)))


def rd(path, col_ok=True):
    d = pd.read_csv(path, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    return d[L].astype(float)

LB = R + '/_labels/'
SRC = {
    'v4': rd(LB + 'rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'),
    'jev': rd(R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv'),
    'sol56': rd(LB + 'rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'),
    'v2': rd(LB + 'rsna-knee-llm-labels/report_labels_v2.csv'),
}
SRC['s6t'] = (SRC['v4'] + SRC['jev']) / 2   # approx of S6fall's JEV-blend target (Gemini half not available locally)
try:
    SRC['bjs'] = rd(R + '/session_2026-10-13/raptor/vast/remote/labels_raptor_rsna_blendjev_sol.csv')
except Exception as e:
    print('bjs missing', e)
# R7 OOF state heads (heliosli state taxonomy, R6's sibling) as a view of the state-target regime
r7 = pd.read_csv(R + '/session_2026-10-18/next/g58/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')

NEG = r"\b(no|not|without|negative for|absence of|absent|free of|nor|unremarkable|rule out|ruled out)\b"


def sents(t):
    t = re.sub(r'\s+', ' ', str(t).replace('>', '. ').replace('\n', '. '))
    return [s.strip().lower() for s in re.split(r'(?<=[.;:!?])\s+|\.\s', t) if s.strip()]


def pos_hit(s, pat, win=45):
    """True if pat occurs in s with no negation cue in the `win` chars before it (same sentence)."""
    for m in re.finditer(pat, s):
        pre = s[max(0, m.start() - win):m.start()]
        if not re.search(NEG, pre):
            return True
    return False


def any_s(ss, subj, pat, win=45):
    return any(re.search(subj, s) and pos_hit(s, pat, win) for s in ss)

SS = {u: sents(t) for u, t in zip(en.StudyInstanceUID, en.Report)}
U = list(SS)

# ---- category definitions: (label, category, function(ss) -> bool, rubric call)
ACL_S = r"\b(acl|anterior cruciate)"
MCL_S = r"\b(mcl|medial collateral|tibial collateral)"
TEAR = r"(tear|torn|rupture|disrupt|discontinu|fissur|bucket|flap tear|radial|horizontal cleavage|cleavage|avuls)"


def side_sent(s, side):
    other = 'lateral' if side == 'medial' else 'medial'
    if re.search(side + r"\s+(and\s+" + other + r"\s+)?menisc", s) or re.search(other + r"\s+and\s+" + side + r"\s+menisc", s) \
            or re.search(r"both menisc|menisci", s) or re.search(r"menisc\w*[^.]{0,30}" + side, s):
        return True
    return False


def men(side):
    def tear(ss):
        return any(side_sent(s, side) and pos_hit(s, TEAR) for s in ss)
    def degen(ss):
        return any(side_sent(s, side) and pos_hit(s, r"(degenerat|intrasubstance|intra-substance|mucoid|myxoid|grade (1|2|i|ii)\b|increased (intra)?signal)") for s in ss)
    def postop(ss):
        return any(side_sent(s, side) and pos_hit(s, r"(menisce?ctomy|resect|post-?surgical|post-?operative|truncat|blunt|diminutive|repair)") for s in ss)
    def extr(ss):
        return any(side_sent(s, side) and pos_hit(s, r"extru") for s in ss)
    return tear, degen, postop, extr

CATS = []
# ACL
CATS += [
    ('ACL', 'complete / full / rupture', lambda ss: any_s(ss, ACL_S, r"(complete|full[- ]thickness|rupture|torn|discontinu|non-?visuali)"), '+'),
    ('ACL', 'high-grade partial', lambda ss: any_s(ss, ACL_S, r"high[- ]grade"), '+'),
    ('ACL', 'low-grade / sprain / grade 1', lambda ss: any_s(ss, ACL_S, r"(low[- ]grade|sprain|grade (1|i)\b|strain|interstitial)"), '-'),
    ('ACL', 'partial (grade unstated)', lambda ss: any_s(ss, ACL_S, r"partial") and not any_s(ss, ACL_S, r"(high|low)[- ]grade"), '?'),
    ('ACL', 'mucoid / degeneration / thickening, no tear', lambda ss: any_s(ss, ACL_S, r"(mucoid|degenerat|thicken|intrasubstance|ganglion)") and not any_s(ss, ACL_S, TEAR), '-'),
    ('ACL', 'reconstruction / graft', lambda ss: any_s(ss, ACL_S + r"|graft", r"(reconstruct|graft|aclr|tunnel)"), '? (graft intact -)'),
    ('ACL', 'chronic / old tear', lambda ss: any_s(ss, ACL_S, r"(chronic|old|remote)"), '+ if discontinuous'),
]
# MCL
CATS += [
    ('MCL', 'grade 1 / sprain / periligamentous edema only', lambda ss: any_s(ss, MCL_S, r"(grade (1|i)\b|sprain|periligament|edema|strain|low[- ]grade)") and not any_s(ss, MCL_S, r"(grade (2|3|ii|iii)\b|partial|complete|full[- ]thickness|rupture|torn|tear|high[- ]grade)"), '-'),
    ('MCL', 'grade 2 / partial tear', lambda ss: any_s(ss, MCL_S, r"(grade (2|ii)\b|partial)") and not any_s(ss, MCL_S, r"(grade (3|iii)\b|complete|full[- ]thickness|high[- ]grade)"), '? (+ if high-grade)'),
    ('MCL', 'grade 3 / complete / high-grade', lambda ss: any_s(ss, MCL_S, r"(grade (3|iii)\b|complete|full[- ]thickness|high[- ]grade|rupture)"), '+ if acute'),
    ('MCL', 'chronic / old / thickened / Pellegrini', lambda ss: any_s(ss, MCL_S + r"|pellegrini", r"(chronic|old|remote|thicken|pellegrini|ossific|healed|prior)"), '-'),
    ('MCL', 'MCL bursitis', lambda ss: any_s(ss, MCL_S, r"burs"), '-'),
]
# Menisci
for side, lab in (('medial', 'Medial Meniscus'), ('lateral', 'Lateral Meniscus')):
    tear, degen, postop, extr = men(side)
    CATS += [
        (lab, 'tear (stated)', tear, '+'),
        (lab, 'degeneration / intrasubstance, no tear stated', lambda ss, d=degen, t=tear: d(ss) and not t(ss), '-'),
        (lab, 'post-op / truncated / blunted, no tear stated', lambda ss, p=postop, t=tear: p(ss) and not t(ss), '+ (truncated / diminutive)'),
        (lab, 'extrusion, no tear stated', lambda ss, e=extr, t=tear: e(ss) and not t(ss), '- (unless morphologic)'),
    ]
# OA / cartilage (study level; compartment attribution approximate)
CH = r"(chondr|cartilag)"
LATF = r"lateral (patellar )?facet|lateral patell|lateral trochle"
LATTF = r"lateral (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial|condyle|plateau)"
CATS += [
    ('Lateral OA', 'lateral PATELLAR facet / lateral trochlea chondral, no lateral TF chondral sentence',
     lambda ss: any((re.search(LATF, s) and re.search(CH, s)) for s in ss) and not any((re.search(LATTF, s) and pos_hit(s, CH)) for s in ss), '- for Lateral OA'),
    ('Lateral OA', 'lateral TF chondral sentence', lambda ss: any((re.search(LATTF, s) and pos_hit(s, CH)) for s in ss), '+ if high-grade >= 1 cm'),
    ('Medial OA', 'low-grade / grade 1-2 / mild chondral only (no high-grade, full or grade 3-4 anywhere)',
     lambda ss: any(pos_hit(s, r"(low[- ]grade|grade (1|2|i|ii)\b|mild(ly)? (chondr|cartilage)|superficial|fibrillat|softening)") and re.search(CH, s) for s in ss)
     and not any(pos_hit(s, r"(high[- ]grade|full[- ]thickness|grade (3|4|iii|iv)\b|denud|exposed bone|bone[- ]on[- ]bone|severe)") for s in ss), '-'),
    ('Medial OA', 'high-grade / full-thickness / grade 3-4 anywhere', lambda ss: any(pos_hit(s, r"(high[- ]grade|full[- ]thickness|grade (3|4|iii|iv)\b|denud|exposed bone|bone[- ]on[- ]bone)") and re.search(CH + r"|compartment|condyle|plateau", s) for s in ss), '+ if >= 1 cm in compartment'),
]
# Effusion
EFF = r"(effusion|joint fluid|intra-?articular fluid|fluid in the joint|suprapatellar (recess|bursa)[^.]{0,20}fluid)"
CATS += [
    ('Effusion', 'small / trace / minimal / mild only', lambda ss: any(re.search(EFF, s) and pos_hit(s, r"(small|trace|minimal|mild|slight|physiolog|little|minor|tiny)") for s in ss)
     and not any(re.search(EFF, s) and pos_hit(s, r"(moderate|large|massive|extensive|significant|marked|considerable|big)") for s in ss), '-'),
    ('Effusion', 'moderate', lambda ss: any(re.search(EFF, s) and pos_hit(s, r"moderate") for s in ss) and not any(re.search(EFF, s) and pos_hit(s, r"(large|massive|marked)") for s in ss), '+'),
    ('Effusion', 'large / massive / marked', lambda ss: any(re.search(EFF, s) and pos_hit(s, r"(large|massive|marked|extensive|considerable)") for s in ss), '+'),
    ('Effusion', 'effusion, size unstated', lambda ss: any(pos_hit(s, r"\beffusion") for s in ss) and not any(re.search(EFF, s) and re.search(r"(small|trace|minimal|mild|slight|physiolog|little|minor|tiny|moderate|large|massive|marked|extensive|significant|considerable)", s) for s in ss), '?'),
]
# Synovitis
CATS += [
    ('Synovitis', "explicit 'synovitis'", lambda ss: any(pos_hit(s, r"synovitis") and not re.search(r"effusion[- ]synovitis", s) for s in ss), '+'),
    ('Synovitis', 'synovial thickening / proliferation / hypertrophy, no explicit synovitis', lambda ss: any(pos_hit(s, r"synovi\w* (thicken|prolifer|hypertroph)|thicken\w* (of the )?synovi|(thickened|hypertrophic|proliferative) synovi") for s in ss) and not any(pos_hit(s, r"synovitis") for s in ss), '+'),
    ('Synovitis', 'Hoffa / infrapatellar fat pad edema or inflammation, no synovitis word', lambda ss: any(re.search(r"(hoffa|infrapatellar fat|fat pad)", s) and pos_hit(s, r"(edema|oedema|inflam|signal|impinge)") for s in ss) and not any(pos_hit(s, r"synovi") for s in ss), '? (Hoffa-synovitis)'),
    ('Synovitis', 'debris / loose bodies / complex effusion, no synovitis word', lambda ss: any(pos_hit(s, r"(debris|complex effusion|hemarthros|haemarthros|lipohemarthros|fluid-fluid|rice bod)") for s in ss) and not any(pos_hit(s, r"synovi") for s in ss), '?'),
    ('Synovitis', 'moderate / large effusion, no synovial word', lambda ss: any(re.search(EFF, s) and pos_hit(s, r"(moderate|large|massive|marked)") for s in ss) and not any(pos_hit(s, r"synovi") for s in ss), '? (effusion-synovitis)'),
]
# Baker's
BAK = r"(baker|popliteal cyst|semimembranosus[^.]{0,30}(bursa|cyst)|gastrocnemi[^.]{0,30}(bursa|cyst))"


def bak_size(ss):
    best = None
    for s in ss:
        if re.search(BAK, s):
            for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:x\s*(\d+(?:[.,]\d+)?)\s*)?(?:x\s*(\d+(?:[.,]\d+)?)\s*)?(mm|cm)", s):
                v = [float(x.replace(',', '.')) for x in m.groups()[:3] if x]
                mx = max(v) * (10 if m.group(4) == 'cm' else 1)
                best = mx if best is None else max(best, mx)
    return best
CATS += [
    ('Baker\'s', 'small / tiny / minimal only', lambda ss: any(re.search(BAK, s) and pos_hit(s, r"(small|tiny|minimal|minute|little|mild)") for s in ss) and not any(re.search(BAK, s) and pos_hit(s, r"(moderate|large|big|sizable|sizeable|giant)") for s in ss), '-'),
    ('Baker\'s', 'moderate / large', lambda ss: any(re.search(BAK, s) and pos_hit(s, r"(moderate|large|big|sizable|sizeable|giant)") for s in ss), '+'),
    ('Baker\'s', 'size given, max < 30 mm (no size word)', lambda ss: (bak_size(ss) or 1e9) < 30 and not any(re.search(BAK, s) and re.search(r"(small|tiny|minimal|moderate|large|big)", s) for s in ss), '- (likely small)'),
    ('Baker\'s', 'size given, max >= 50 mm (no size word)', lambda ss: (bak_size(ss) or 0) >= 50 and not any(re.search(BAK, s) and re.search(r"(small|tiny|minimal|moderate|large|big)", s) for s in ss), '+ (likely moderate / large)'),
    ('Baker\'s', 'ruptured / leaking', lambda ss: any(re.search(BAK, s) and pos_hit(s, r"(ruptur|leak|dissect)") for s in ss), '+ if moderate+'),
]
# Contusion
BME = r"(bone marrow (edema|oedema)|marrow (edema|oedema)|bone (edema|oedema)|edema-like|oedema-like|marrow signal)"
CATS += [
    ('Contusion', "explicit contusion / bone bruise", lambda ss: any(pos_hit(s, r"(contusion|bruis)") for s in ss), '+'),
    ('Contusion', 'BME with degenerative / subchondral / reactive / stress context, no contusion word',
     lambda ss: any(pos_hit(s, BME) and re.search(r"(degenerat|subchondral|reactive|stress|arthr|osteoarth|chondr|cyst|insufficien)", s) for s in ss) and not any(pos_hit(s, r"(contusion|bruis)") for s in ss), '-'),
    ('Contusion', 'BME non-specific, no contusion word, no degenerative context',
     lambda ss: any(pos_hit(s, BME) and not re.search(r"(degenerat|subchondral|reactive|stress|arthr|osteoarth|chondr|cyst|insufficien)", s) for s in ss) and not any(pos_hit(s, r"(contusion|bruis)") for s in ss), '?'),
    ('Contusion', 'BME around a fracture only (fracture + BME, no contusion word)', lambda ss: any(pos_hit(s, r"fractur") for s in ss) and any(pos_hit(s, BME) for s in ss) and not any(pos_hit(s, r"(contusion|bruis)") for s in ss), '? (+ if impact edema apart from the line)'),
]
# Fracture
CATS += [
    ('Fracture', 'insufficiency / SIFK / subchondral fracture / SONK', lambda ss: any(pos_hit(s, r"(insufficien|sifk|sonk|subchondral (insufficiency )?fractur|osteonecros|spontaneous osteonecrosis)") for s in ss), '? (fracture line, not traumatic)'),
    ('Fracture', 'stress fracture / stress reaction', lambda ss: any(pos_hit(s, r"stress (fractur|reaction|response)") for s in ss), '?'),
    ('Fracture', 'old / healed / remote fracture', lambda ss: any(pos_hit(s, r"(old|healed|healing|remote|prior|previous|chronic|consolidated|united)\b[^.]{0,25}fractur|fractur[^.]{0,25}(healed|old|remote|consolidat)") for s in ss), '-'),
    ('Fracture', 'avulsion / Segond / arcuate', lambda ss: any(pos_hit(s, r"(avulsion|segond|arcuate)") for s in ss), '+ if acute cortical'),
    ('Fracture', 'osteochondral lesion / OCD (no fracture word)', lambda ss: any(pos_hit(s, r"(osteochondral|osteochondritis|\bocd\b|\bocl\b)") for s in ss) and not any(pos_hit(s, r"fractur") for s in ss), '-'),
    ('Fracture', 'trabecular / microfracture (no other fracture word)', lambda ss: any(pos_hit(s, r"(trabecular|micro-?fractur|microtrabecular|micro-?trabecular)") for s in ss) and not any(pos_hit(s, r"(?<!micro)(?<!micro-)fractur") for s in ss), '-'),
    ('Fracture', 'acute / impaction / plateau / displaced fracture', lambda ss: any(pos_hit(s, r"(acute|impact|depress|displac|plateau|comminut|nondisplac|non-displac|undisplac)[^.]{0,30}fractur|fractur[^.]{0,30}(acute|impact|depress|displac|plateau)") for s in ss), '+'),
]

rows = []
for lab, cat, fn, rub in CATS:
    m = np.array([bool(fn(SS[u])) for u in U])
    ids = [u for u, k in zip(U, m) if k]
    r = {'label': lab, 'category': cat, 'rubric': rub, 'n_en': int(m.sum()), 'frac_en': round(m.mean(), 3)}
    for k, d in SRC.items():
        v = d.reindex(ids)[lab]
        r[k] = round(float((v >= .5).mean()), 2) if len(ids) else None
    g = [u for u in ids if u in gold_ids]
    r['n_g58'] = len(g)
    if g:
        r['g58_exp+'] = round(float(tr.set_index('StudyInstanceUID').loc[g, lab].mean()), 2)
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 200)
print(T.to_string(index=False))
T.to_csv(R + '/session_2026-10-20/levers/census_lang.csv', index=False)

# contamination: share of each source's EN positives that sit in a rubric-negative category
print('\nREGIME CONTAMINATION (EN): share of a source\'s positives (>= .5) that fall in rubric-NEGATIVE categories')
negcats = {}
for lab, cat, fn, rub in CATS:
    if rub.startswith('-'):
        negcats.setdefault(lab, []).append(fn)
crow = []
for lab, fns in negcats.items():
    m = np.array([any(bool(f(SS[u])) for f in fns) for u in U])
    poscat = {lab2: fn for (lab2, cat, fn, rub) in CATS if lab2 == lab and rub.startswith('+')}
    r = {'label': lab, 'n_negcat': int(m.sum())}
    for k, d in SRC.items():
        v = d.reindex(U)[lab].values
        pos = v >= .5
        r[k + '_pos'] = int(pos.sum()); r[k + '_contam'] = round(float((pos & m).sum() / max(pos.sum(), 1)), 3)
    crow.append(r)
print(pd.DataFrame(crow).to_string(index=False))
