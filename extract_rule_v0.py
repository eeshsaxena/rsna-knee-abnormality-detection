#!/usr/bin/env python3
"""Rule-based v0 report -> 12-label extractor (multilingual: EN/ES/NL + common radiology
terms). Clause-scoped positive-finding detection with negation. Measured on the 58 gold
(report + true label) studies, then applied to all 4407 -> pseudo-labels for later training.

Not a model. A floor + a reusable measurement harness for the label pipeline (lever #1).
"""
import re
import unicodedata
import pandas as pd
import numpy as np

DATA = r"C:\Users\eeshs\Downloads\rsna-knee\_data\train.csv"
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# Negation / normalcy cues (EN/ES/NL). Clause-scoped.
NEG = ['intact', 'normal', 'preserved', 'unremarkable', 'no evidence', 'without', 'no ',
       'not torn', 'no tear', 'sin ', 'no hay', 'no se ', 'conservad', 'integr',
       'geen ', 'zonder', 'niet ', 'onopvallend', 'ongestoord', 'behouden',
       'no acute', 'is intact', 'are intact', 'well preserved', 'no significant']

# Positive finding cues shared by tear-type structures.
TEAR = ['tear', 'torn', 'ruptur', 'rotura', 'roto', 'rota', 'desgarr', 'scheur',
        'gescheurd', 'disrupt', 'discontinu', 'macerat', 'avuls']
OA = ['osteoarthr', 'arthros', 'artros', 'osteophyt', 'osteofit', 'chondral loss',
      'cartilage loss', 'chondropat', 'condropat', 'chondromalac', 'condromalac',
      'joint space narrow', 'kraakbeen', 'artrose', 'degenerative', 'degenerat']

# label -> (anatomy terms, positive-finding terms). Accent-stripped, lowercase.
RULES = {
    'ACL': (['acl', 'anterior cruciate', 'cruzado anterior', 'voorste kruisband', 'lca',
             'vkb'], TEAR + ['sprain', 'esguince']),
    'MCL': (['mcl', 'medial collateral', 'colateral medial', 'lateral interno',
             'mediale collaterale', 'binnenband'], TEAR + ['sprain', 'esguince']),
    'Medial Meniscus': (['medial meniscus', 'menisco medial', 'menisco interno',
                         'mediale meniscus', 'binnenmeniscus'], TEAR),
    'Lateral Meniscus': (['lateral meniscus', 'menisco lateral', 'menisco externo',
                          'laterale meniscus', 'buitenmeniscus'], TEAR),
    'Medial OA': (['medial compartment', 'femorotibial medial', 'compartimento medial',
                   'mediale compartiment', 'medial femorotibial'], OA),
    'Lateral OA': (['lateral compartment', 'femorotibial lateral', 'compartimento lateral',
                    'laterale compartiment', 'lateral femorotibial'], OA),
    'PF OA': (['patellofemoral', 'patelofemoral', 'femoropatellar', 'retropatellar',
               'trochlea', 'troclea', 'patellar cartilage', 'rotula', 'patela',
               'patellofemoraal'], OA + ['chondral', 'condral']),
    'Effusion': (['effusion', 'derrame', 'joint fluid', 'vocht', 'erguss', 'epanch'],
                 ['effusion', 'derrame', 'fluid', 'vocht', 'epanch']),
    'Synovitis': (['synovitis', 'sinovitis', 'synovial thick', 'synoviaal', 'synovitis',
                   'sinovial'], ['synovit', 'sinovit', 'thick', 'inflam', 'proliferat']),
    "Baker's": (['baker', 'popliteal cyst', 'quiste poplite', 'bakercyste', 'poplit cyst',
                 'quiste de baker'], ['cyst', 'quiste', 'cyste']),
    'Contusion': (['contusion', 'bone bruise', 'bone marrow edema', 'bone marrow oedema',
                   'bone edema', 'subchondral edema', 'edema oseo', 'contusion osea',
                   'beenmergoedeem', 'botcontusie', 'medular edema', 'marrow oedema'],
                  ['contusion', 'bruise', 'edema', 'oedema', 'oedeem']),
    'Fracture': (['fracture', 'fractura', 'fractuur', 'avulsion fract'],
                 ['fracture', 'fractura', 'fractuur', 'avuls']),
}


def norm(t):
    t = str(t).replace('\\n', ' ').replace('\\t', ' ')
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
    return t.lower()


def clauses(t):
    t = re.sub(r'[\n\r;]+', '.', t)
    return [c.strip() for c in re.split(r'[.]', t) if c.strip()]


def has(clause, terms):
    return any(term in clause for term in terms)


def predict_one(report):
    text = norm(report)
    cls = clauses(text)
    out = {}
    for lab, (anat, pos) in RULES.items():
        vote = 0
        for c in cls:
            if has(c, anat) and has(c, pos) and not has(c, NEG):
                vote = 1
                break
        out[lab] = vote
    return out


def main():
    df = pd.read_csv(DATA)
    preds = df['Report'].map(predict_one).apply(pd.Series)
    preds.columns = list(RULES.keys())

    gold_mask = df['ACL'].notna()
    gold = df[gold_mask].reset_index(drop=True)
    gp = preds[gold_mask.values].reset_index(drop=True)

    print(f"gold N = {len(gold)}\n")
    print(f"{'label':18s} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'prec':>5} {'rec':>5} "
          f"{'f1':>5}  {'AUC':>5}")
    aucs = []
    for lab in LABELS:
        y = (gold[lab].values == 1.0).astype(int)
        p = gp[lab].values.astype(int)
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        auc = (tpr + tnr) / 2  # AUC of a binary predictor
        aucs.append(auc)
        print(f"{lab:18s} {tp:3d} {fp:3d} {fn:3d} {tn:3d}  {prec:5.2f} {rec:5.2f} "
              f"{f1:5.2f}  {auc:5.3f}")
    print(f"\nMACRO AUC on gold-58 (binary-pred): {np.mean(aucs):.4f}  "
          f"(baseline all-0.5 = 0.500)")

    # pseudo-labels for the 4349 report-only studies (+ keep gold as-is where present)
    out = df[['StudyInstanceUID']].copy()
    for lab in LABELS:
        out[lab] = np.where(gold_mask.values, df[lab].values, preds[lab].values.astype(float))
    outp = r"C:\Users\eeshs\Downloads\rsna-knee\_data\pseudo_labels_rulev0.csv"
    out.to_csv(outp, index=False)
    pos_rate = preds[~gold_mask.values].mean().round(3).to_dict()
    print(f"\npseudo-label positive rate on the 4349 report-only:\n  {pos_rate}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
