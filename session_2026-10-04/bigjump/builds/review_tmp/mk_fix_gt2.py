p = 'code/gt2_read.py'
s = open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("#   TTR-A fold-0 vs f0_s42_v4 (same fold 0, seed 42, steven_v4 targets and selection labels, gold74 held out).",
    "#   TTR-A fold-0 (seed 1042, trainer BJ-5) vs f0_s42_v4 (same fold 0, steven_v4 targets and selection labels, gold74\n"
    "#   held out). Files are found by glob (oof_f0_s*_{raw,ema}.csv), so the seed does not matter here.")
rep("#   ~/miniconda3/envs/rsna/bin/python ~/bigjump_build/code/gt2_read.py --ttr ~/fleet_runs/f0_s42_ttr \\",
    "#   ~/miniconda3/envs/rsna/bin/python ~/bigjump_build/code/gt2_read.py --ttr ~/fleet_runs/f0_s1042_ttr \\")
open(p, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
print('ok')
