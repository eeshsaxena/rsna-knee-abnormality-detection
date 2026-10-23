# sweep_width.py (10-20 r5): how tight must the E-OA prior be before "A = eoa10 after a NON-move" pays? (MODEL, uses decision_mc.py)
import subprocess, json, sys
R5 = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/r5'
rows = []
for nz in (1.0, 1.5):
    for s10 in (1.0e-4, 1.5e-4, 2.0e-4, 2.5e-4, 3.0e-4, 4.5e-4):
        s20 = 2.2 * s10; tag = f'w_s{s10*1e4:.1f}_n{nz}'
        subprocess.run([sys.executable, R5 + '/decision_mc.py', 'rep=2', 'prior=custom', 'm10=3e-4', f's10={s10}', 'm20=1e-4', f's20={s20}', f'noise={nz}', f'tag={tag}'],
                       check=True, capture_output=True)
        J = json.load(open(f'{R5}/dec_{tag}.json'))
        for o in ('O0 no move', 'P10- only eoa10 scored, no move', 'P20- only eoa20 scored, no move', 'O10 eoa10 best-id', 'O20 eoa20 best-id'):
            O = J['outcomes'][o]; P = O['pairs']
            d = P['eoa10+R6']['silver'] - P['c4+R6']['silver']; d2 = P['eoa20+R6']['silver'] - P['eoa10+R6']['silver']
            rows.append((nz, s10, o, O['P'], O['d10'][0], O['d10'][2], d))
            print(f'noise x{nz} prior d10 N(+3e-4, {s10*1e4:.1f}e-4) d20 N(+1e-4, {s20*1e4:.1f}e-4) | {o:34s} P {O["P"]:.2f} | post d10 {O["d10"][0]*1e4:+.2f}e-4 P>0 {O["d10"][2]:.2f} | silver eoa10+R6 - c4+R6 {100*d:+.1f} pts | eoa20+R6 - eoa10+R6 {100*d2:+.1f}')
