# t_b1_model: B1 fleet_model_ttr.py and its model_sha registry.
#  (a) the contract files shipped with the bigjump code (fleet_common.py, fleet_model.py, fleet_model_f2.py) are
#      byte-identical to round3/fleet (~/fleet), fleet2/code (~/fleet2_code) and k2/code: every running / packed
#      member's CONTRACT and model_sha still match
#  (b) model_sha registry: the SAME function body in fleet_worker.py, pack_fleet_ds.py and tta_bench.py gives family 1
#      fa35880fca963652 (the running family-1 checkpoints), family 2 == the fleet2 tools' value, TTR distinct
#  (c) KneeFleetNetTTR with ttr_size == input size (no resize) == KneeFleetNet: eval logits, train logits (same RNG:
#      drop_path / dropout) and every gradient, max |diff| 0; state_dict keys identical (no new parameter/buffer)
#  (d) with a resize: encode == cnn(ImageNet-norm(interpolate(x/255, ttr_size, bilinear, antialias))) exactly, the
#      backbone sees ttr_size x ttr_size (320 -> 224 checked with a hook), fp32 resize, antialias active
import os, sys, ast, json, hashlib, importlib.util
from _paths import CODE, ORIG, F2, K2C, OUT
import torch, torch.nn.functional as F
torch.set_num_threads(4)
res = {}
for f in ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py'):
    srcs = {'bigjump': CODE, 'fleet2': F2, 'k2': K2C} | ({'round3': ORIG} if f != 'fleet_model_f2.py' else {})
    h = {k: hashlib.sha256(open(f'{d}/{f}', 'rb').read()).hexdigest() for k, d in srcs.items()}
    print(f, {k: v[:16] for k, v in h.items()})
    assert len(set(h.values())) == 1, f
    res[f'identical_{f}'] = True
sys.path.insert(0, CODE)


def fn_from(path, name):
    """exec ONE top-level function of a script (the worker / trainer run on import) in a fresh namespace."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][0]
    import inspect; ns = {"hashlib": hashlib, "inspect": inspect}   # the scripts import these at module level
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
    return ns[name]


w_sha = fn_from(f'{CODE}/fleet_worker.py', '_model_sha')
p_sha = fn_from(f'{CODE}/pack_fleet_ds.py', '_model_sha')
t_sha = fn_from(f'{CODE}/tta_bench.py', 'model_sha')
shas = {m: {k: f(m) for k, f in (('worker', w_sha), ('pack', p_sha), ('tta_bench', t_sha))}
        for m in ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr')}
print('model_sha', json.dumps(shas))
for m, d in shas.items():
    assert len(set(d.values())) == 1, (m, d)
assert shas['fleet_model']['worker'] == 'fa35880fca963652', shas['fleet_model']
f2_ref = fn_from(f'{F2}/pack_fleet_ds.py', '_model_sha')
sys.path.insert(0, F2)
assert shas['fleet_model_f2']['worker'] == f2_ref('fleet_model_f2'), 'family-2 sha moved'
assert len({shas[m]['worker'] for m in shas}) == 3, 'TTR sha must be distinct'
res['model_sha'] = {m: d['worker'] for m, d in shas.items()}

from fleet_model import KneeFleetNet
from fleet_model_ttr import KneeFleetNetTTR, mirror_views as mv_ttr
from fleet_model import mirror_views
assert mv_ttr is mirror_views
BB = 'convnext_atto.d2_in1k'
kw = dict(backbone=BB, n_views=2, n_slices=2, pretrained=False, grad_ckpt=True, drop_path=0.2, dropout=0.3)
g = torch.Generator().manual_seed(7)
x = torch.randint(0, 256, (1, 2, 2, 3, 64, 64), dtype=torch.uint8, generator=g)
x[:, 1] = 0                                               # absent view, as the Dataset feeds it
pr = torch.tensor([[True, False]]); ct = torch.tensor([[1, 3]])
y = torch.rand(1, 12, generator=g); y[0, 3] = float('nan')
out = {}
for name, ctor in (('f1', lambda: KneeFleetNet(**kw)), ('ttr_noresize', lambda: KneeFleetNetTTR(ttr_size=64, **kw))):
    torch.manual_seed(0)
    m = ctor().to(memory_format=torch.channels_last)
    m.eval()
    with torch.no_grad():
        le = m(x, pr, ct)
    m.train(); m.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    lt = m(x, pr, ct)
    msk = torch.isfinite(y).float(); yy = torch.nan_to_num(y)
    loss = (F.binary_cross_entropy_with_logits(lt.float(), yy, reduction='none') * msk).sum() / msk.sum().clamp(min=1)
    loss.backward()
    out[name] = (le, lt, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}, m.state_dict())
a, b = out['f1'], out['ttr_noresize']
assert list(a[3]) == list(b[3]), 'TTR adds/removes state_dict keys'
d_eval = (a[0] - b[0]).abs().max().item(); d_train = (a[1] - b[1]).abs().max().item()
d_grad = max((a[2][n] - b[2][n]).abs().max().item() for n in a[2])
print(f'(c) TTR(ttr_size 64) vs KneeFleetNet on 64 px: eval {d_eval} train {d_train} grads ({len(a[2])}) {d_grad}; '
      f'{len(a[3])} state_dict keys identical')
assert d_eval == 0 and d_train == 0 and d_grad == 0
res['noresize_parity'] = dict(eval=d_eval, train=d_train, grad=d_grad)

torch.manual_seed(0)
m = KneeFleetNetTTR(ttr_size=96, **kw).eval()
seen = []
h = m.cnn.register_forward_pre_hook(lambda mod, inp: seen.append((tuple(inp[0].shape), inp[0].dtype)))
xi = x.flatten(0, 2)
with torch.no_grad():
    e = m.encode(xi)
    xr = F.interpolate(xi.float() / 255.0, size=(96, 96), mode='bilinear', align_corners=False, antialias=True)
    man = m.cnn(((xr - m.m) / m.s).contiguous(memory_format=torch.channels_last))
h.remove()
d_enc = (e - man).abs().max().item()
print(f'(d) encode vs manual resize+norm+cnn: max|diff| {d_enc}; backbone input {seen[0]}')
assert d_enc == 0 and seen[0][0][-2:] == (96, 96) and seen[0][1] == torch.float32
m2 = KneeFleetNetTTR(ttr_size=224, **kw).eval(); seen.clear()
h = m2.cnn.register_forward_pre_hook(lambda mod, inp: seen.append(tuple(inp[0].shape)))
x320 = torch.randint(0, 256, (1, 2, 2, 3, 320, 320), dtype=torch.uint8, generator=g)
with torch.no_grad():
    lo = m2(x320, pr, ct)
h.remove()
xa = F.interpolate(x320[0, 0, :1].float() / 255, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
xn = F.interpolate(x320[0, 0, :1].float() / 255, size=(224, 224), mode='bilinear', align_corners=False, antialias=False)
print(f'(d) 320 px input -> backbone sees {seen[0]} (chunk {m2.chunk}); logits {tuple(lo.shape)} finite '
      f'{bool(torch.isfinite(lo).all())}; antialias changes the image: max|aa - no-aa| {(xa - xn).abs().max().item():.4f}')
assert seen[0][-2:] == (224, 224) and torch.isfinite(lo).all() and (xa - xn).abs().max() > 0
res['resize'] = dict(encode_manual=d_enc, backbone_in=seen[0])
json.dump(res, open(f'{OUT}/t_b1_model.json', 'w'), indent=1, default=str)
print('T_B1_MODEL OK')
