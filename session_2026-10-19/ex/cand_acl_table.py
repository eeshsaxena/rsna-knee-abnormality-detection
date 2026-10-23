# cand_acl_table.py: compacts cand_acl_box.log (aggregates) into per-group x scheme rows, 3 models side by side. Local only.
import re, collections
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
rx = re.compile(r'\[(ACL B\d) \| (\S+)\s*\| (\S+)\s*\] base ([\d.]+) \| ACL <- (\S+)\s+rho ([+-][\d.]+) q ([+-][\d.]+) \| a -0.1 ([+-][\d.]+) \(P>0 ([\d.]+)\) \| a \+0.1 ([+-][\d.]+) \(P>0 ([\d.]+)\) \| a \+0.2 ([+-][\d.]+) \(P>0 ([\d.]+)\)')
T = collections.OrderedDict()
for line in open(E + 'cand_acl_box.log', encoding='utf-8'):
    m = rx.search(line)
    if not m: continue
    lab, mk, sn, base, g, rho, q, dn, pn, d1, p1, d2, p2 = m.groups()
    T.setdefault((lab, g, sn), []).append(dict(mk=mk, base=float(base), rho=float(rho), q=float(q), dn=float(dn), pn=float(pn),
                                                d1=float(d1), p1=float(p1), d2=float(d2), p2=float(p2)))
print('| label | ACL <- | scheme | base AUC (3) | rho - q (3) | a -.1 (3) | a .1 (3) [P>0 min] | a .2 (3) [P>0 min] | crit |')
print('|---|---|---|---|---|---|---|---|---|')
for (lab, g, sn), v in T.items():
    j = lambda k, f='%+.4f': '/'.join(f % x[k] for x in v)
    c1 = all(x['d1'] > 0 for x in v) and min(x['p1'] for x in v) >= .9 and all(x['dn'] < 0 for x in v)
    c2 = all(x['d2'] > 0 for x in v) and min(x['p2'] for x in v) >= .9 and all(x['dn'] < 0 for x in v)
    crit = ('a.1 PASS ' if c1 else '') + ('a.2 PASS' if c2 else '')
    print('| %s | %s | %s | %s | %s | %s | %s [%.2f] | %s [%.2f] | %s |' % (lab, g, sn, j('base', '%.3f'),
          '/'.join('%+.2f' % (x['rho'] - x['q']) for x in v), j('dn'), j('d1'), min(x['p1'] for x in v), j('d2'), min(x['p2'] for x in v), crit or '-'))
