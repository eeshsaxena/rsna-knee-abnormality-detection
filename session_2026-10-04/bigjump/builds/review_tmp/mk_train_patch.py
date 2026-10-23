s = open('code/fleet_train.py', encoding='utf-8', newline='').read()
assert '\r\n' not in s
a = """assert not (A.f2 and A.model == 'ttr'), '--model ttr is a ConvNeXt (family-1 style) path: not with --f2 1'   # BJ-1
"""
b = a + """if A.model == 'ttr':                                                                                         # R-5
    assert 1000 <= A.seed <= 1999, (f'--model ttr runs use seeds 1000-1999 (fold-0 1042, all-data 1001..): the '
                                    f'f{{fold}}_s{{seed}} member key and the save() prefix must never equal family 1 (42, 43..) or '
                                    f'family 2 (142, 242..): got --seed {A.seed}')
"""
assert s.count(a) == 1
s = s.replace(a, b)
open('review_tmp/fleet_train_new.py', 'w', encoding='utf-8', newline='').write(s)
print('ok')
