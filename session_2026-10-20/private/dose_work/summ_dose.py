# summ_dose.py: summarise dose_<tag>.json (dose_mc.py). MIX = scenario-weighted average (GUESS weights below); metrics are linear in the
# scenario probabilities, so the MIX of each metric is the weighted mean of the per-scenario metric (paired draws, same N).
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
WMIX = {'S0 same': .60, 'C+ .5 more': .10, 'C- .5 less': .10, 'SITE held-out': .20}
SCN = ['S0 same', 'C+ .5 more', 'C- .5 less', 'C++ 1.0 more', 'SITE held-out', 'MIX']
AS = ['eoa20', 'pw20', 'pw25', 'pw30', 'R1{pw30,pw25}']
KEYP = ['pw30 + R6pw20', 'R6pw20 + R1{pw30,pw25}', 'pw25 + R6pw20', 'pw20 + R6pw20', 'eoa20 + R6pw20', 'R6 + pw30', 'R6 + pw20', 'HW + pw30',
        'HW + pw20', 'pw20 + pw30', 'eoa20 + pw30', 'c4 + pw30', 'c4 + pw20']


def norm(PR, p):
    if p in PR: return p
    q = ' + '.join(p.split(' + ')[::-1]); return q


def add_mix(R):
    S = R['scen']; M = {'singles': {}, 'vs_pw30': {}, 'pairs': {}}
    for sec in ('singles', 'pairs'):
        for k in S['S0 same'][sec]:
            M[sec][k] = {m: sum(w * S[s][sec][k][m] for s, w in WMIX.items()) for m in ('E', 'silver', 'r150', 'r130', 'erank', 'pub', 'bronze', 'r200')}
    for k in S['S0 same']['vs_pw30']:
        M['vs_pw30'][k] = {m: sum(w * S[s]['vs_pw30'][k][m] for s, w in WMIX.items()) for m in ('E', 'P')}
    S['MIX'] = M


def main():
    tags = sys.argv[1:] or [os.path.basename(f)[5:-5] for f in sorted(glob.glob(HERE + '/dose_*.json'))]
    RS = {}
    for t in tags:
        R = json.load(open(HERE + '/dose_%s.json' % t)); add_mix(R); RS[t] = R
    B = RS['base'] if 'base' in RS else RS[tags[0]]
    print('MIX weights (GUESS):', WMIX)
    print('\n## 1. A alone, primary (base): E margin e-4 | private macro (pub-scale) | sd | [p10 p90] | vs pw30 E (P>) | P(silver) P(r<=150) E[rank] (median rank)')
    for sn in SCN:
        S = B['scen'][sn]; print('-- %s' % sn)
        for a in AS:
            m, v = S['singles'][a], S['vs_pw30'][a]
            extra = ' sd %5.2f [%+6.2f %+6.2f] med rank %5.1f' % (m['sd'] * 1e4, m['p10'] * 1e4, m['p90'] * 1e4, m['mrank']) if sn != 'MIX' else ''
            print('   %-14s E %+6.2f  priv %.5f%s | vs pw30 %+5.2f (P %.2f) | silver %.3f r150 %.3f E[rank] %5.1f' % (
                a, m['E'] * 1e4, m['pub'], extra, v['E'] * 1e4, v['P'], m['silver'], m['r150'], m['erank']))
    print('\n## 2. Key pairs, primary: silver / r150 / E[rank] / E[max] e-4')
    print('   %-24s ' % 'pair' + ' | '.join('%-24s' % s for s in SCN))
    for p in KEYP:
        cells = []
        for sn in SCN:
            PR = B['scen'][sn]['pairs']; v = PR[norm(PR, p)]
            cells.append('%.3f/%.3f/%5.1f/%+5.1f' % (v['silver'], v['r150'], v['erank'], v['E'] * 1e4))
        print('   %-24s ' % p + ' | '.join('%-24s' % c for c in cells))
    print('\n## 3. Best pair per metric, primary (top 4)')
    for sn in SCN:
        PR = B['scen'][sn]['pairs']; print('-- %s' % sn)
        for key in ('silver', 'r150', 'erank'):
            o = sorted(PR.items(), key=lambda kv: kv[1][key] if key != 'erank' else -kv[1][key], reverse=True)[:4]
            print('   by %-6s ' % key + ' | '.join('%s %.3f/%.3f/%.1f' % (k, v['silver'], v['r150'], v['erank']) for k, v in o))
    print('\n## 4. Best B for each A (MIX, primary)')
    PR = B['scen']['MIX']['pairs']
    for a in AS + ['R6pw20']:
        cand = {k: v for k, v in PR.items() if a in k.split(' + ')}
        for key in ('silver', 'r150', 'erank'):
            k, v = sorted(cand.items(), key=lambda kv: kv[1][key] if key != 'erank' else -kv[1][key], reverse=True)[0]
            print('   A %-14s best B by %-6s: %-28s %.3f/%.3f/%.1f' % (a, key, k, v['silver'], v['r150'], v['erank']))
    print('\n## 5. Variants: pairs with B = R6pw20, A in {pw30, R1, pw25, pw20, eoa20}; delta vs A = pw30 (silver pts / r150 pts / E[rank])')
    for t, R in RS.items():
        for sn in ('S0 same', 'C+ .5 more', 'C- .5 less', 'SITE held-out', 'MIX'):
            PR = R['scen'][sn]['pairs']; ref = PR[norm(PR, 'pw30 + R6pw20')]
            cells = []
            for a in ('R1{pw30,pw25}', 'pw25', 'pw20', 'eoa20'):
                v = PR[norm(PR, a + ' + R6pw20')]
                cells.append('%s %+5.1f/%+5.1f/%+6.1f' % (a.replace('R1{pw30,pw25}', 'R1'), (v['silver'] - ref['silver']) * 100, (v['r150'] - ref['r150']) * 100, v['erank'] - ref['erank']))
            print('   %-10s %-13s ref pw30+R6pw20 %.3f/%.3f/%5.1f | ' % (t, sn, ref['silver'], ref['r150'], ref['erank']) + ' | '.join(cells))
    print('\n## 6. Variants: A alone E vs pw30 (e-4) and P(silver) / r150 / E[rank] in MIX')
    for t, R in RS.items():
        S = R['scen']['MIX']
        print('   %-10s ' % t + ' | '.join('%s %+.2f %.3f/%.3f/%.0f' % (a.replace('R1{pw30,pw25}', 'R1'), S['vs_pw30'][a]['E'] * 1e4, S['singles'][a]['silver'], S['singles'][a]['r150'], S['singles'][a]['erank']) for a in AS))
    print('\n## 7. Variants: population argmax share and pop vs pw30 per scenario')
    for t, R in RS.items():
        for sn in ('S0 same', 'C+ .5 more', 'C- .5 less', 'SITE held-out'):
            S = R['scen'][sn]
            print('   %-10s %-13s argmax %s | pop vs pw30 %s' % (t, sn, ' '.join('%s %.2f' % kv for kv in S['argmax_dose'].items()),
                                                           ' '.join('%s %+.2f' % (k, v['E'] * 1e4) for k, v in S['pop_vs_pw30'].items() if k in ('eoa20', 'pw20', 'pw25', 'pw35'))))
    # count over variants x scenarios (excluding MIX and C++) where pw20+R6pw20 beats pw30+R6pw20
    print('\n## 7b. Variants: MIX silver / r150 / E[rank] of key pairs')
    for t, R in RS.items():
        PR = R['scen']['MIX']['pairs']
        print('   %-10s ' % t + ' | '.join('%s %.3f/%.3f/%.0f' % (p.replace('R6pw20 + R1{pw30,pw25}', 'R1 + R6pw20'), PR[norm(PR, p)]['silver'], PR[norm(PR, p)]['r150'], PR[norm(PR, p)]['erank'])
                                         for p in ('pw30 + R6pw20', 'R6pw20 + R1{pw30,pw25}', 'pw25 + R6pw20', 'pw20 + R6pw20', 'eoa20 + R6pw20', 'R6 + pw30', 'R6 + pw20', 'eoa10 + pw30')))
    print('\n## 7c. Median private rank of key pairs (primary)')
    for sn in ('S0 same', 'C+ .5 more', 'C- .5 less', 'SITE held-out'):
        PR = B['scen'][sn]['pairs']
        print('   %-13s ' % sn + ' | '.join('%s %.0f' % (p, PR[norm(PR, p)]['mrank']) for p in ('pw30 + R6pw20', 'R6pw20 + R1{pw30,pw25}', 'pw25 + R6pw20', 'pw20 + R6pw20', 'eoa20 + R6pw20', 'R6 + pw30', 'R6 + pw20')))
    print('\n## 8. Win counts over variants x {S0, C+, C-, SITE} (and MIX): A + R6pw20 vs pw30 + R6pw20')
    for a in ('pw20', 'pw25', 'eoa20', 'R1{pw30,pw25}'):
        for sset, nm in ((('S0 same', 'C+ .5 more', 'C- .5 less', 'SITE held-out'), '4 scen'), (('MIX',), 'MIX')):
            c = {'silver': 0, 'r150': 0, 'erank': 0}; n = 0; dl = {'silver': [], 'r150': [], 'erank': []}
            for t, R in RS.items():
                for sn in sset:
                    PR = R['scen'][sn]['pairs']; ref = PR[norm(PR, 'pw30 + R6pw20')]; v = PR[norm(PR, a + ' + R6pw20')]; n += 1
                    for k in c:
                        better = v[k] > ref[k] if k != 'erank' else v[k] < ref[k]
                        c[k] += better; dl[k].append(v[k] - ref[k])
            print('   %-14s %-6s n %2d | silver %2d (%+.1f..%+.1f) | r150 %2d (%+.1f..%+.1f) | E[rank] %2d (%+.0f..%+.0f)' % (
                a, nm, n, c['silver'], min(dl['silver']) * 100, max(dl['silver']) * 100, c['r150'], min(dl['r150']) * 100, max(dl['r150']) * 100,
                c['erank'], min(dl['erank']), max(dl['erank'])))


main()
