# private_designs.py (10-22, Claude, private-entries track). MODEL: P(>= 1 clean B) and private value of the 3 private slots.
# Exact enumeration, no randomness. Inputs are tagged in private.md (cause.md scenario mixture, ops.md refusal GUESS, bpick values).
import itertools, sys
sys.stdout.reconfigure(encoding='utf-8')
SCEN = {'cause.md mixture (persist .57 / clear .05 / worse .80; w .5/.25/.25)': [(.5, .57), (.25, .05), (.25, .80)],
        'LASTDAY indep, P(clean) .55': [(1.0, .45)],
        'LASTDAY indep, P(clean) .64': [(1.0, .36)]}
ERR = .02          # GUESS: ERROR / timeout per session on top of the canary
REFUSE = .05       # GUESS (ops.md): same-version resubmit refused; hits every r6pw20 v1 copy together, seen at c1's submit
RESCUE = .13       # MODEL cause.md sec 6: share of would-be canaries the g85j guard variant rescues
# value of B (silver pts vs B = R6, step / smooth). bpick 10-21 base (k .86) and skeptic k ~ N(.6, .3); r6e20 = r6pw20 - 0.5 (VAL picks2)
VAL = {'base k .86': {'pw': (0.6, 1.3), 'e20': (0.1, 0.8)}, 'skeptic k N(.6,.3)': {'pw': (0.2, 0.6), 'e20': (-0.3, 0.1)}}
BASE = (87.6, 76.3)  # P(silver) x 100 with B = R6 (step / smooth), bpick 10-21

def design(slots, commit_fail):
    """slots: list of kinds in submission (= id) order: 'pw' (r6pw20 v1 resubmit), 'g85' (guard variant, commit), 'e20' (r6e20, commit).
    -> per scenario: P(any clean), P(B kind = pw-class), P(B = e20)."""
    out = {}
    for name, sc in SCEN.items():
        pany = ppw = pe = 0.0
        for w, pc in sc:
            for refused in (0, 1):
                pr = REFUSE if refused else 1 - REFUSE
                if pr == 0: continue
                # per slot: P(clean)
                ps = []
                for k in slots:
                    if k == 'pw':
                        q = 0.0 if refused else (1 - pc) * (1 - ERR)
                    else:
                        fail_c = pc * (1 - RESCUE) if k == 'g85' else pc
                        q = (1 - commit_fail) * (1 - fail_c) * (1 - ERR)
                    ps.append(q)
                # B = earliest clean slot
                none = 1.0
                for k, q in zip(slots, ps):
                    hit = none * q
                    if k == 'e20': pe += w * pr * hit
                    else: ppw += w * pr * hit
                    none *= 1 - q
                pany += w * pr * (1 - none)
        out[name] = (pany, ppw, pe)
    return out

DES = [('D0 none (B = R6)', [], 0), ('D5 1 x r6pw20', ['pw'], 0), ('D4 2 x r6pw20', ['pw', 'pw'], 0),
       ('D1 3 x r6pw20 (RECOMMENDED)', ['pw', 'pw', 'pw'], 0),
       ('D2 2 x r6pw20 + r6e20, commit fail .10', ['pw', 'pw', 'e20'], .10), ('D2 ... commit fail .25', ['pw', 'pw', 'e20'], .25),
       ('D3 2 x r6pw20 + g85j, commit fail .10', ['pw', 'pw', 'g85'], .10), ('D3 ... commit fail .25', ['pw', 'pw', 'g85'], .25),
       ('D6 r6pw20 + r6e20 + g85j (2 commits), fail .10', ['pw', 'e20', 'g85'], .10)]
for vname, v in VAL.items():
    print('== value rows: %s ==' % vname)
    for sname in SCEN:
        print('  -- %s' % sname)
        print('  %-48s %8s %8s %8s | %16s | %14s' % ('design', 'P(any)', 'P(B=pw)', 'P(B=e20)', 'E gain step/smth', 'P(silver) s/sm'))
        for dn, sl, cf in DES:
            pany, ppw, pe = design(sl, cf)[sname]
            gs = ppw * v['pw'][0] + pe * v['e20'][0]; gm = ppw * v['pw'][1] + pe * v['e20'][1]
            print('  %-48s %8.3f %8.3f %8.3f | %+7.3f / %+6.3f | %5.2f / %5.2f' % (dn, pany, ppw, pe, gs, gm, BASE[0] + gs, BASE[1] + gm))
