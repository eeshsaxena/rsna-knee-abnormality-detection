# Member-passes that fit after the parent: stage = 7 (spawn/load/write) + max(18 decode, n*g) + crop(10) <= (8.6 - P)*60
for P in (6.0,6.3,6.65,7.0,7.4,7.7,8.0):
    avail=(8.6-P)*60
    row=[]
    for g in (3.0,3.75,4.5):
        n=int(max(0,(avail-7-10))/g)
        row.append(f'g{g}: {n:2d}')
    print(f'P {P:.2f} h  window {avail:5.0f} min | '+' | '.join(row))
# nominal plan: F1 x7 (6 all-data + fold-0) + F2 x4 (0.91x), mirror TTA x2
for g in (3.0,3.75,4.5):
    t=2*(7*g+4*0.91*g)
    print(f'plan 11 members x mirror: g{g} -> GPU {t:.0f} min, stage {7+max(18,t)+10:.0f} min, fits to P <= {8.6-(7+max(18,t)+10)/60:.2f} h; no-mirror stage {7+max(18,t/2)+10:.0f} min fits to P <= {8.6-(7+max(18,t/2)+10)/60:.2f} h')
