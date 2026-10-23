# skep_apick_kband.py (skeptic of track apick; MODEL). Runs apick_mc.py UNCHANGED except:
#  (1) the swide in-band ruler: rivals iid with density ~exp(-k x) on [0,1) (x from the band bottom; surface.md's band model,
#      k from the .959/.960/.961 count ratio 80/51/28), so weight = G(t)^4 (1-G(t))^46, G(t) = (1-exp(-k t))/(1-exp(-k));
#  (2) outputs go to skep_apick_<tag>.log/.json (never overwrites apick_*).
# Args as apick_mc.py plus kband=<k>.
import sys
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday'
kb = [a for a in sys.argv[1:] if a.startswith('kband=')]
K = float(kb[0].split('=')[1]) if kb else 0.5
sys.argv = ['apick_mc.py'] + [a for a in sys.argv[1:] if not a.startswith('kband=')]
src = open(HERE + '/apick_mc.py', encoding='utf-8').read()
old = "    wt = wt * np.where(t < 1, t ** 4 * (1 - t) ** 46, 0.)"
new = "    Gt = (1 - np.exp(-%r * t)) / (1 - np.exp(-%r)); wt = wt * np.where(t < 1, Gt ** 4 * (1 - Gt) ** 46, 0.)" % (K, K)
assert src.count(old) == 1; src = src.replace(old, new)
for a, b in (("HERE + '/apick_%s.log' % TAG", "HERE + '/skep_apick_%s.log' % TAG"), ("HERE + '/apick_%s.json' % TAG", "HERE + '/skep_apick_%s.json' % TAG")):
    assert src.count(a) == 1; src = src.replace(a, b)
exec(compile(src, HERE + '/apick_mc.py[kband]', 'exec'), {'__name__': '__main__'})
