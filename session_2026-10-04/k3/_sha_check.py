# extract fleet_worker._model_sha verbatim from the shipped worker source and run it on k3/code
import ast, sys, os
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
sys.path.insert(0, CODE)
src = open(os.path.join(CODE, 'fleet_worker.py'), encoding='utf-8').read()
tree = ast.parse(src)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_model_sha'][0]
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'fleet_worker._model_sha', 'exec'), ns)
for m in ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr'):
    print(m, ns['_model_sha'](m))
