# pairs_table.py (10-21 levers, Claude). LOCAL ONLY. One row per label pair (66): every implied expert coupling (tetrachoric) the regime work
# uses, written to regime_pairs.csv; prints a compact view. Inputs: regime_cal.json, regime_dlt.json, regime_dlt_sweep.json (rel, meanD, gamma).
import json, csv
import numpy as np
AB = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
      "PF OA": "PF", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Cont", "Fracture": "Frac"}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
cal = json.load(open('regime_cal.json')); dl = json.load(open('regime_dlt.json')); sw = json.load(open('regime_dlt_sweep.json'))
lam = cal['lam']; REL, mD, gam = sw['REL'], sw['meanD'], sw['GAM']['3mean']
rows = []
for k, e in dl['pairs'].items():
    c = cal['pairs'][k]; a, b = k.split(' | '); tr = a in OA and b in OA
    eb = e['delta'] if tr else mD + REL * (e['delta'] - mD)
    rows.append(dict(pair=AB[a] + '-' + AB[b], trio=tr, rep_train=round(e['rep_tr'], 3), gold74=round(c['tet74'], 3), gold58_expert=round(e['exp_g58'], 3),
                     report_v2_on_gold58=round(e['rep_g58'], 3), same_knee_D_v2=round(e['delta'], 3), same_knee_D_v4=round(e['delta4'], 3),
                     oai_B1=c['tetoai'], c4_rho_gold58=round(e['c4_g58'], 3), r7_rho_train=round(e['r7_tr'], 3), r7_rho_gold58=round(e['r7_g58'], 3),
                     implied_G74_lamFT=round(lam['FT'] * c['tet74'], 3), implied_G74_lamAREA=round(lam['AREA'] * c['tet74'], 3),
                     implied_DLT_EB=round(e['rep_tr'] + eb, 3), implied_DLT_EBg_3mean=round(e['rep_tr'] + gam * eb, 3), implied_DLT_RAW=round(e['rep_tr'] + e['delta'], 3)))
with open('regime_pairs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('%-9s %6s %6s %6s %6s | %7s %7s %7s %7s' % ('pair', 'rep', 'g74', 'Dv2', 'Dv4', 'G74xFT', 'DLT-EB', 'DLT-RAW', 'c4g58'))
for r in sorted(rows, key=lambda r: r['implied_DLT_EB'] - r['rep_train']):
    print('%-9s %+6.2f %+6.2f %+6.2f %+6.2f | %+7.2f %+7.2f %+7.2f %+7.2f' % (r['pair'], r['rep_train'], r['gold74'], r['same_knee_D_v2'], r['same_knee_D_v4'],
          r['implied_G74_lamFT'], r['implied_DLT_EB'], r['implied_DLT_RAW'], r['c4_rho_gold58']))
