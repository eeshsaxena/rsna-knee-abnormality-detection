# stack_mc_sens.py (10-21, Claude, STACK). MODEL only: stack_mc.py slates re-run with harsher MCLmm priors (sensitivity).
import numpy as np, runpy, io, contextlib
src = open('stack_mc.py', encoding='utf-8').read()
pre, post = src.split("Ipl = rng.normal(.16, .12, N)")
VARS = {'stack_mc prior (S1/S2/S3 .40/.35/.25, x.75)': None,
        'pessimistic (.25/.50/.25, x.60)': "sc2 = rng.choice(3, N, p=[.25, .50, .25]); mu['MCLmm'] = .60 * (np.array([2.3, 0., 1.3])[sc2] + np.array([1.0, .8, 1.2])[sc2] * rng.standard_normal(N))",
        'skeptic null N(0, 1.0)': "mu['MCLmm'] = rng.normal(0., 1.0, N)",
        'recal prior N(-0.2, 1.0)': "mu['MCLmm'] = rng.normal(-.2, 1.0, N)"}
for nm, line in VARS.items():
    code = pre + ('' if line is None else line + '\n') + "Ipl = rng.normal(.16, .12, N)" + post
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, 'stack_mc_variant', 'exec'), {})
    out = buf.getvalue().splitlines()
    keep = [l for l in out if l.startswith(('MCLmm alone', 'pw25+MCLmm', 'pw25+lomm05+MCLmm', 'NIGHT', '{S_safe}', '{S_wide}', '{S_alt}', '{S_alt, S_wide}', '{S_safe, S_wide}', '{pw25, lomm05, S_wide}', 'component'))]
    print('== MCLmm prior: %s' % nm)
    for l in keep: print('   ' + l)
