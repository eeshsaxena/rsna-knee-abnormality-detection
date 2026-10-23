import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build_k8.py')
s = open(p, encoding='utf-8').read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old)); s = s.replace(old, new)
rep("""    interim = not (nA == 5 and nC == 5)
""", """    interim = not (nA == 5 and nC == 5)
    sim = any('/k8_test/sim/' in str(r.get('run', '')) for r in mem)   # collected from the SIMULATION root (test only)
""")
rep("""    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k8=True, interim=interim, do_not_submit=interim,""",
    """    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k8=True, interim=interim, simulated=sim,
                do_not_submit=interim or sim,""")
rep("""    if interim else f'FINAL ({len(mem)} members: A 5/5, C 5/5: {keys8})')
""", """    if interim else f'FINAL ({len(mem)} members: A 5/5, C 5/5: {keys8})')
    if sim:
        tag = 'SIMULATION (members from ~/k8_test/sim: TEST ONLY, NEVER VERSION / PUSH / SUBMIT) ' + tag
""")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
