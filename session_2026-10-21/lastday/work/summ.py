# summ.py: table of P(pub > swide) / E population / R1 increment per candidate across surface_mc variants (reads surface_*.json). MODEL.
import json, glob, os
H = os.path.dirname(os.path.abspath(__file__))
tags = ['base', 'seed2', 'unif', 'noswr', 'nolink', 'lskep', 'mstack', 'mzero', 'mneg', 'kaphi', 'kaplo', 'linear', 'tau6', 'eoaflat',
        'eoasteep', 'wsd2', 'wsd4', 'rsd8', 'lsd13', 'moaregime', 'moaoptim', 'moapess', 'moaflip', 'rho0', 'qsd15']
R = {t: json.load(open(H + '/surface_%s.json' % t)) for t in tags if os.path.exists(H + '/surface_%s.json' % t)}
cands = list(R['base']['res'].keys())
out = open(H + '/summ.out', 'w')
def P(s=''):
    print(s); out.write(s + '\n')
P('ESS: ' + ' '.join('%s %.0f' % (t, R[t]['ess']) for t in R))
for c in cands:
    P('\n%s' % c)
    P('  %-10s %8s %8s %8s %8s %8s' % ('variant', 'pub', 'P(pub>0)', 'popE', 'P(pop>0)', 'R1inc'))
    for t in R:
        x = R[t]['res'][c]
        flag = ' <== PASS' if (x['Ppub'] >= .6 and x['pop'] >= 0) else ''
        P('  %-10s %+8.2f %8.2f %+8.2f %8.2f %+8.2f%s' % (t, x['pub'], x['Ppub'], x['pop'], x['Ppop'], x['R1inc'], flag))
    ps = [R[t]['res'][c]['Ppub'] for t in R]; es = [R[t]['res'][c]['pop'] for t in R]; ri = [R[t]['res'][c]['R1inc'] for t in R]
    npass = sum(1 for t in R if R[t]['res'][c]['Ppub'] >= .6 and R[t]['res'][c]['pop'] >= 0)
    P('  RANGE P(pub>0) %.2f-%.2f | popE %+.2f..%+.2f | R1inc %+.2f..%+.2f | PASS in %d / %d variants' % (min(ps), max(ps), min(es), max(es), min(ri), max(ri), npass, len(R)))
