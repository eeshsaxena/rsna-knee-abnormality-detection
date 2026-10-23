#!/usr/bin/env python3
"""_test_run_pilot.py (2026-10-08): offline test of run_pilot.py. The anthropic SDK's HTTP transport is replaced by an
httpx.MockTransport (no network), a FAKE key is used, batches report in_progress once then ended, the first batch has
one errored request, every succeeded request bills 4,000 input + 1,500 output tokens ($0.023 at Opus batch price).
Scenarios: cap 8 -> gold74 in one round; cap 3.5 -> gold58 only; cap 2.0 -> STOP after stage A; chunk_above 1.0 ->
chunks of 20; bad key (401) -> exits before any POST. Writes only under ./_test_run_pilot_out.
  python _test_run_pilot.py
"""
import os, sys, io, json, glob, shutil, runpy, random, contextlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '_test_run_pilot_out')
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
FAKE = 'sk-ant-FAKE-RUNPILOT-0000-not-a-real-key'
os.environ['ANTHROPIC_API_KEY'] = FAKE
os.environ.pop('ANTHROPIC_BASE_URL', None)
import httpx, anthropic                                     # noqa: E402
sys.path.insert(0, HERE)
import prompt_v1 as P                                       # noqa: E402

CAP, BATCH, POLLS, MODE = [], {}, {}, {'auth_ok': True}


def fake_msg(model, k):
    rnd = random.Random(k)
    obj = {key: dict(report_evidence=rnd.choice(P.REPORT_EVIDENCE), image_evidence=rnd.choice(P.IMAGE_EVIDENCE),
                     call=rnd.choice(P.CALLS), probability=rnd.randint(0, 100), note='synthetic') for key, _ in P.LABELS}
    return {'id': f'msg_{k}', 'type': 'message', 'role': 'assistant', 'model': model,
            'content': [{'type': 'text', 'text': json.dumps(obj)}], 'stop_reason': 'end_turn', 'stop_sequence': None,
            'usage': {'input_tokens': 4000, 'output_tokens': 1500, 'cache_creation_input_tokens': 0,
                      'cache_read_input_tokens': 0}}


def bobj(bid, status):
    return {'id': bid, 'type': 'message_batch', 'processing_status': status,
            'request_counts': {'processing': 0, 'succeeded': 1, 'errored': 0, 'canceled': 0, 'expired': 0},
            'ended_at': None, 'created_at': '2026-10-08T00:00:00Z', 'expires_at': '2026-10-09T00:00:00Z',
            'cancel_initiated_at': None, 'archived_at': None,
            'results_url': f'https://api.anthropic.com/v1/messages/batches/{bid}/results' if status == 'ended' else None}


def handler(req):
    body = json.loads(req.content) if req.content else None
    CAP.append((req.method, str(req.url), body))
    path = req.url.path
    if req.method == 'GET' and path.startswith('/v1/models/'):
        if not MODE['auth_ok']:
            return httpx.Response(401, json={'type': 'error', 'error': {'type': 'authentication_error',
                                                                        'message': 'API key is invalid.'}})
        return httpx.Response(200, json={'id': path.split('/')[-1], 'type': 'model', 'display_name': 'x',
                                         'created_at': '2026-01-01T00:00:00Z'})
    if req.method == 'POST' and path == '/v1/messages/batches':
        bid = f'msgbatch_t{len(BATCH)}'
        BATCH[bid] = body['requests']
        return httpx.Response(200, json=bobj(bid, 'in_progress'))
    pre = '/v1/messages/batches/'
    if req.method == 'GET' and path.startswith(pre):
        bid = path[len(pre):].split('/')[0]
        if not path.endswith('/results'):
            POLLS[bid] = POLLS.get(bid, 0) + 1
            return httpx.Response(200, json=bobj(bid, 'ended' if POLLS[bid] > 1 else 'in_progress'))
        lines = []
        for i, r in enumerate(BATCH[bid]):
            if i == 0 and bid == 'msgbatch_t0':
                lines.append({'custom_id': r['custom_id'], 'result': {'type': 'errored', 'error': {
                    'type': 'error', 'error': {'type': 'api_error', 'message': 'synthetic'}}}})
            else:
                lines.append({'custom_id': r['custom_id'], 'result': {'type': 'succeeded',
                                                                      'message': fake_msg(r['params']['model'], i)}})
        return httpx.Response(200, content='\n'.join(json.dumps(x) for x in lines).encode(),
                              headers={'content-type': 'application/binary'})
    return httpx.Response(404, json={'type': 'error', 'error': {'type': 'not_found_error', 'message': path}})


_Real = anthropic.Anthropic


class MockAnthropic(_Real):
    def __init__(self, *a, **kw):
        kw['http_client'] = httpx.Client(transport=httpx.MockTransport(handler))
        super().__init__(*a, **kw)


anthropic.Anthropic = MockAnthropic


def run(name, extra):
    od = os.path.join(OUT, name)
    BATCH.clear(); POLLS.clear()
    n0 = len(CAP)
    argv = ['run_pilot.py', '--allow_agent_shell', '--poll_s', '0', '--out_dir', od, '--no_save'] + extra
    out, code = io.StringIO(), 0
    old = sys.argv; sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            runpy.run_path(os.path.join(HERE, 'run_pilot.py'), run_name='__main__')
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old
    posts = [c for c in CAP[n0:] if c[0] == 'POST']
    sizes = [len(p[2]['requests']) for p in posts]
    sp = json.load(open(os.path.join(od, 'spend.json'))) if os.path.exists(os.path.join(od, 'spend.json')) else None
    st = {}
    for arm in ('text', 'img'):
        f = os.path.join(od, f'soft_{arm}_opus-5-5.csv')
        if os.path.exists(f):
            import pandas as pd
            st[arm] = pd.read_csv(f).status.value_counts().to_dict()
    return dict(code=code, sizes=sizes, spend=sp, status=st, out=out.getvalue(),
                hosts=sorted({httpx.URL(c[1]).host for c in CAP[n0:]}))


res = {}


def check(name, fn):
    try:
        fn(); res[name] = 'PASS'
    except Exception as e:
        res[name] = f'FAIL {type(e).__name__}: {e}'; traceback.print_exc()


def s1():
    r = run('cap8', [])
    assert r['hosts'] == ['api.anthropic.com'], r['hosts']
    assert r['sizes'][:2] == [10, 10], r['sizes']                     # stage A, both arms
    assert r['sizes'][2:] == [65, 64], r['sizes']                     # rest of gold74 (+1 errored text retry)
    assert r['status'] == {'text': {'ok': 74}, 'img': {'ok': 74}}, r['status']
    assert abs(r['spend']['actual_usd'] - 148 * 0.023) < 1e-6, r['spend']['actual_usd']
    assert 'scope: gold74' in r['out'] and 'DECISION v2' in r['out'], r['out'][-1500:]


def s2():
    r = run('cap3.5', ['--cap', '3.5'])
    assert 'scope: gold58' in r['out'], r['out'][-1500:]
    assert r['sizes'] == [10, 10, 49, 48], r['sizes']
    assert r['status'] == {'text': {'ok': 58}, 'img': {'ok': 58}}, r['status']
    assert r['spend']['actual_usd'] <= 3.5


def s3():
    r = run('cap2', ['--cap', '2.0'])
    assert 'STOP: even the rest of gold58' in r['out'], r['out'][-1500:]
    assert r['sizes'] == [10, 10], r['sizes']


def s4():
    r = run('chunk', ['--chunk_above', '1.0'])
    assert r['sizes'][:2] == [10, 10] and all(s <= 20 for s in r['sizes'][2:]), r['sizes']
    assert r['status'] == {'text': {'ok': 74}, 'img': {'ok': 74}}, r['status']
    assert r['out'].count('chunked') >= 3, r['out'][-1500:]


def s5():
    MODE['auth_ok'] = False
    r = run('badkey', [])
    MODE['auth_ok'] = True
    assert r['sizes'] == [] and 'auth check FAILED' in str(r['code']), (r['sizes'], r['code'])


def s6():
    blob = ''
    for f in glob.glob(os.path.join(OUT, '**', '*'), recursive=True):
        if os.path.isfile(f):
            blob += open(f, 'rb').read().decode('utf-8', 'ignore')
    assert FAKE not in blob, 'fake key leaked into an output file'


def s7():
    r = run('cap9', ['--cap', '9'])
    assert 'above the registered cap' in r['out'] or r['code'] not in (0, None), r['code']


for n, f in (('S1 cap 8 -> gold74', s1), ('S2 cap 3.5 -> gold58 only', s2), ('S3 cap 2 -> stop after stage A', s3),
             ('S4 chunking', s4), ('S5 bad key -> no POST', s5), ('S7 cap above prereg refused', s7),
             ('S6 key never in outputs', s6)):
    check(n, f)
print(json.dumps(res, indent=1))
