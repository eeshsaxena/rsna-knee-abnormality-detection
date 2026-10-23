#!/usr/bin/env python3
"""selftest_offline.py (gold_prep/relabel, 2026-10-06): exercises relabel_call.py end to end WITHOUT any network.

The real anthropic SDK is used, but its HTTP transport is replaced by httpx.MockTransport, so every request is captured
locally and answered with a synthetic message. A FAKE key is used. Checks:
  T1 dry run builds every request and sends nothing
  T2 sync text + img arms: host == api.anthropic.com only, model / output_config / effort / fallbacks / cache_control
     / image count per manifest, CSV written, a synthetic refusal is recorded as status refusal
  T3 batch submit -> poll -> fetch round trip; the batch body keeps output_config and has no fallbacks; the errored
     request is resubmitted once, a second submit before that batch is fetched queues nothing, the final fetch is all ok
  T4 the fake key never appears in stdout/stderr or in any output file
  T5 guards: ANTHROPIC_BASE_URL elsewhere -> refused before any request; inside a Claude Code shell (CLAUDECODE) ->
     refused; no key -> refused
  T6 pilot_eval.py runs on SYNTHETIC soft labels (numbers meaningless) and its gate reacts (img == gold-like -> PASS,
     img == text -> FAIL, 4 studies answered by a fallback model -> 54 paired -> INVALID)
Writes only under ./_selftest_out.  Usage: python selftest_offline.py
"""
import os, sys, io, json, glob, shutil, runpy, random, contextlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, '_selftest_out')
DATA = os.path.join(HERE, '_selftest_data')
FAKE = 'sk-ant-FAKE-SELFTEST-0000-not-a-real-key'
shutil.rmtree(OUTD, ignore_errors=True); os.makedirs(OUTD)
import httpx, anthropic                                     # noqa: E402
sys.path.insert(0, HERE)
import prompt_v1 as P                                       # noqa: E402

CAP = []                                                    # (method, url, headers, body)
REFUSE_PID = {'n': 0}
BATCH = {}


def fake_msg(model, k):
    rnd = random.Random(k)
    obj = {key: dict(report_evidence=rnd.choice(P.REPORT_EVIDENCE), image_evidence=rnd.choice(P.IMAGE_EVIDENCE),
                     call=rnd.choice(P.CALLS), probability=rnd.randint(0, 100), note='synthetic')
           for key, _ in P.LABELS}
    return {'id': f'msg_{k}', 'type': 'message', 'role': 'assistant', 'model': model,
            'content': [{'type': 'text', 'text': json.dumps(obj)}], 'stop_reason': 'end_turn', 'stop_sequence': None,
            'usage': {'input_tokens': 4000, 'output_tokens': 1500}}


def handler(req: httpx.Request):
    body = json.loads(req.content) if req.content else None
    CAP.append((req.method, str(req.url), dict(req.headers), body))
    path = req.url.path
    if req.method == 'POST' and path == '/v1/messages':
        REFUSE_PID['n'] += 1
        if REFUSE_PID['n'] == 2:                            # one synthetic refusal
            m = fake_msg(body['model'], 'r'); m['stop_reason'] = 'refusal'; m['content'] = []
            m['stop_details'] = {'type': 'refusal', 'category': None, 'explanation': 'synthetic'}
            return httpx.Response(200, json=m)
        return httpx.Response(200, json=fake_msg(body['model'], len(CAP)))
    if req.method == 'POST' and path == '/v1/messages/batches':      # unique id per batch, like the real API
        bid = f'msgbatch_selftest{len(BATCH)}'
        BATCH[bid] = body['requests']
        return httpx.Response(200, json=batch_obj(bid, 'in_progress'))
    pre = '/v1/messages/batches/'
    if req.method == 'GET' and path.startswith(pre) and path[len(pre):].split('/')[0] in BATCH:
        bid = path[len(pre):].split('/')[0]
        if not path.endswith('/results'):
            return httpx.Response(200, json=batch_obj(bid, 'ended'))
        lines = []
        for i, r in enumerate(BATCH[bid]):
            if i == 0 and bid == 'msgbatch_selftest0':              # one synthetic error, in the first batch only
                lines.append({'custom_id': r['custom_id'], 'result': {'type': 'errored', 'error': {
                    'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'synthetic'}}}})
            else:
                lines.append({'custom_id': r['custom_id'], 'result': {'type': 'succeeded',
                                                                      'message': fake_msg(r['params']['model'], i)}})
        return httpx.Response(200, content='\n'.join(json.dumps(x) for x in lines).encode(),
                              headers={'content-type': 'application/binary'})
    return httpx.Response(404, json={'type': 'error', 'error': {'type': 'not_found_error', 'message': path}})


def batch_obj(bid, status):
    return {'id': bid, 'type': 'message_batch', 'processing_status': status,
            'request_counts': {'processing': 0, 'succeeded': 3, 'errored': 1, 'canceled': 0, 'expired': 0},
            'ended_at': '2026-10-06T00:10:00Z' if status == 'ended' else None, 'created_at': '2026-10-06T00:00:00Z',
            'expires_at': '2026-10-07T00:00:00Z', 'cancel_initiated_at': None, 'archived_at': None,
            'results_url': f'https://api.anthropic.com/v1/messages/batches/{bid}/results' if status == 'ended' else None}


_Real = anthropic.Anthropic


class MockAnthropic(_Real):
    def __init__(self, *a, **kw):
        kw['http_client'] = httpx.Client(transport=httpx.MockTransport(handler))
        super().__init__(*a, **kw)


anthropic.Anthropic = MockAnthropic


def run(args, expect_exit=False):
    argv = ['relabel_call.py'] + args
    out, err = io.StringIO(), io.StringIO()
    code = 0
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(os.path.join(HERE, 'relabel_call.py'), run_name='__main__')
    except SystemExit as e:
        code = e.code if e.code is not None else 0
    finally:
        sys.argv = old
    if not expect_exit:
        assert code in (0, None), (args, code, err.getvalue()[-500:], out.getvalue()[-500:])
    return out.getvalue() + err.getvalue(), code


COMMON = ['--ids', os.path.join(OUTD, 'ids4.csv'), '--manifest', os.path.join(DATA, 'manifest_rule4.csv'),
          '--png_root', DATA, '--out_dir', OUTD]
results, logs = {}, []
import pandas as pd                                          # noqa: E402
man = pd.read_csv(os.path.join(DATA, 'manifest_rule4.csv'), dtype={'StudyInstanceUID': str})
ids = man.StudyInstanceUID.drop_duplicates().tolist()
pd.DataFrame({'StudyInstanceUID': ids}).to_csv(os.path.join(OUTD, 'ids4.csv'), index=False)
nimg = man.groupby('StudyInstanceUID').size().to_dict()
os.environ.pop('ANTHROPIC_BASE_URL', None)
os.environ.pop('CLAUDECODE', None)                          # the HTTP layer is mocked here; T5 re-checks the guard


def check(name, fn):
    try:
        fn(); results[name] = 'PASS'
    except Exception as e:
        results[name] = f'FAIL {type(e).__name__}: {e}'
        traceback.print_exc()


def t1():
    os.environ.pop('ANTHROPIC_API_KEY', None)
    n0 = len(CAP)
    o, _ = run(['--arm', 'img', '--dry_run'] + COMMON); logs.append(o)
    assert len(CAP) == n0, 'dry run made a request'
    assert 'Nothing was sent' in o and 'COST' in o, o[-400:]


def t2():
    os.environ['ANTHROPIC_API_KEY'] = FAKE
    n0 = len(CAP)
    o1, _ = run(['--arm', 'text', '--mode', 'sync', '--concurrency', '1'] + COMMON); logs.append(o1)
    o2, _ = run(['--arm', 'img', '--mode', 'sync', '--concurrency', '2'] + COMMON); logs.append(o2)
    reqs = CAP[n0:]
    assert len(reqs) == 2 * len(ids), len(reqs)
    for meth, url, hd, body in reqs:
        assert httpx.URL(url).host == 'api.anthropic.com', url
        assert hd.get('x-api-key') == FAKE                  # the key goes to the provider only
        assert 'server-side-fallback-2026-07-01' in hd.get('anthropic-beta', ''), hd.get('anthropic-beta')
        assert body['model'] == 'claude-opus-5-5' and body['fallbacks'] == 'default'
        assert body['output_config']['effort'] == 'medium'
        assert body['output_config']['format']['type'] == 'json_schema'
        assert body['system'][0]['cache_control'] == {'type': 'ephemeral'}
        c = body['messages'][0]['content']
        n_im = sum(b['type'] == 'image' for b in c)
        txt = ' '.join(b.get('text', '') for b in c if b['type'] == 'text')
        assert '1.2.826.0.1.3680043' not in json.dumps(body), 'a StudyInstanceUID was sent'
        if 'No images are provided' in txt:
            assert n_im == 0
        else:
            pid = txt.split('STUDY ')[1].split()[0]
            sid = [s for s in ids if 'S' + __import__('hashlib').sha256(s.encode()).hexdigest()[:12] == pid][0]
            assert n_im == nimg[sid], (n_im, nimg[sid])
    t = pd.read_csv(os.path.join(OUTD, 'soft_text_opus-5-5.csv')); i = pd.read_csv(os.path.join(OUTD, 'soft_img_opus-5-5.csv'))
    assert len(t) == len(ids) and len(i) == len(ids)
    for d in (t, i):                                         # per-row provenance = what the request was sent with
        assert set(d.effort) == {'medium'} and set(d.prompt_sha) == {P.prompt_sha()[:16]}, (set(d.effort), set(d.prompt_sha))
    st = pd.concat([t.status, i.status]).value_counts().to_dict()
    assert st.get('refusal') == 1 and st.get('ok') == 2 * len(ids) - 1, st
    assert t.Effusion.between(0, 1).sum() >= len(ids) - 1


def t3():
    os.environ['ANTHROPIC_API_KEY'] = FAKE
    for f in glob.glob(os.path.join(OUTD, 'raw_img_*')) + glob.glob(os.path.join(OUTD, 'soft_img_*')):
        os.remove(f)
    n0 = len(CAP)
    for mode in ('batch_submit', 'batch_poll', 'batch_fetch'):
        o, _ = run(['--arm', 'img', '--mode', mode] + COMMON); logs.append(o)
    posts = [c for c in CAP[n0:] if c[0] == 'POST']
    assert len(posts) == 1 and posts[0][1].endswith('/v1/messages/batches')
    for r in posts[0][3]['requests']:
        assert 'output_config' in r['params'] and 'fallbacks' not in r['params'], list(r['params'])
        assert sum(b['type'] == 'image' for b in r['params']['messages'][0]['content']) > 0
    i = pd.read_csv(os.path.join(OUTD, 'soft_img_opus-5-5.csv'))
    st = i.status.value_counts().to_dict()
    assert st.get('ok') == len(ids) - 1 and st.get('batch_errored') == 1, st
    n1 = len(CAP)                                          # resubmission picks up exactly the 1 errored request
    o, _ = run(['--arm', 'img', '--mode', 'batch_submit'] + COMMON); logs.append(o)
    posts = [c for c in CAP[n1:] if c[0] == 'POST']
    assert len(posts) == 1 and len(posts[0][3]['requests']) == 1, [len(p[3]['requests']) for p in posts]
    n2 = len(CAP)                                          # submitting again BEFORE that batch is fetched: no duplicate
    o, _ = run(['--arm', 'img', '--mode', 'batch_submit'] + COMMON); logs.append(o)
    assert not [c for c in CAP[n2:] if c[0] == 'POST'], 'a pending resubmission was queued twice'
    o, _ = run(['--arm', 'img', '--mode', 'batch_fetch'] + COMMON); logs.append(o)
    i = pd.read_csv(os.path.join(OUTD, 'soft_img_opus-5-5.csv'))
    assert i.status.value_counts().to_dict() == {'ok': len(ids)}, i.status.value_counts().to_dict()
    assert set(i.effort) == {'medium'} and set(i.model_requested) == {'claude-opus-5-5'}, (set(i.effort), set(i.model_requested))


def t4():
    blob = '\n'.join(logs)
    for f in glob.glob(os.path.join(OUTD, '**', '*'), recursive=True):
        if os.path.isfile(f):
            blob += open(f, 'rb').read().decode('utf-8', 'ignore')
    assert FAKE not in blob, 'fake key leaked into output'


def t5():
    os.environ['ANTHROPIC_API_KEY'] = FAKE
    os.environ['ANTHROPIC_BASE_URL'] = 'https://collector.example.com'
    n0 = len(CAP)
    o, code = run(['--arm', 'text', '--mode', 'sync'] + COMMON, expect_exit=True)
    os.environ.pop('ANTHROPIC_BASE_URL')
    assert code not in (0, None) and len(CAP) == n0 and 'refusing' in str(code), (code, o[-300:])
    os.environ['CLAUDECODE'] = '1'                           # an agent's shell: its key is not the pilot key
    o, code = run(['--arm', 'text', '--mode', 'sync'] + COMMON, expect_exit=True)
    os.environ.pop('CLAUDECODE')
    assert code not in (0, None) and len(CAP) == n0 and 'Claude Code session' in str(code), (code, o[-300:])
    os.environ.pop('ANTHROPIC_API_KEY')
    o, code = run(['--arm', 'text', '--mode', 'sync'] + COMMON, expect_exit=True)
    assert code not in (0, None) and len(CAP) == n0 and 'not set' in str(code), (code, o[-300:])


def t6():
    import numpy as np
    sys.path.insert(0, HERE)
    DS = os.path.expanduser('~/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train')
    g = pd.read_csv(f'{DS}/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
    g58 = g[g.source == 'original_gold58']
    rng = np.random.default_rng(1)
    names = [n for _, n in P.LABELS]

    def mk(arm, noise, base=None):
        d = pd.DataFrame({'StudyInstanceUID': g58.StudyInstanceUID, 'arm': arm, 'status': 'ok',
                          'model_requested': 'claude-opus-5-5', 'served_model': 'claude-opus-5-5', 'prompt_sha': 'x'})
        for n in names:
            src = g58[n].to_numpy(float) if base is None else base[n].to_numpy(float)
            d[n] = np.clip(src * 0.6 + 0.2 + rng.normal(0, noise, len(d)), 0, 1)
        return d
    tx = mk('text', 0.45)
    good, same = mk('img', 0.12), tx.copy()
    same['arm'] = 'img'
    fb4 = good.copy()                                       # 4 studies answered by a fallback model -> 54 paired
    fb4.loc[fb4.index[:4], 'served_model'] = 'claude-opus-5'
    verdicts = {}
    for nm, im in (('good', good), ('same', same), ('fb4', fb4)):
        pt, pi = os.path.join(OUTD, f'syn_text_{nm}.csv'), os.path.join(OUTD, f'syn_img_{nm}.csv')
        tx.to_csv(pt, index=False); im.to_csv(pi, index=False)
        out = io.StringIO(); old = sys.argv
        sys.argv = ['pilot_eval.py', '--text', pt, '--img', pi, '--gold', f'{DS}/gold74_eval_set.csv',
                    '--mix50', f'{DS}/labels_mix50_logit.csv', '--v4', f'{DS}/llm_labels_v4_blend.csv', '--boot', '200',
                    '--out', os.path.join(OUTD, f'syn_eval_{nm}.json')]
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(os.path.join(HERE, 'pilot_eval.py'), run_name='__main__')
        finally:
            sys.argv = old
        ev = json.load(open(os.path.join(OUTD, f'syn_eval_{nm}.json')))
        verdicts[nm] = ev['verdict']
        if nm == 'fb4':
            assert ev['n_paired'] == 54 and ev['n_dropped_fallback'] == 4, ev
        logs.append(out.getvalue())
    assert verdicts == {'good': 'PASS', 'same': 'FAIL', 'fb4': 'INVALID'}, verdicts


for name, fn in (('T1 dry run, no network', t1), ('T2 sync arms', t2), ('T3 batch round trip', t3),
                 ('T5 guards', t5), ('T6 pilot_eval synthetic', t6), ('T4 key never in outputs', t4)):
    check(name, fn)
print(json.dumps(results, indent=1))
print(f'requests captured by the mock transport: {len(CAP)}; hosts: {sorted({httpx.URL(c[1]).host for c in CAP})}')
sys.exit(0 if all(v == 'PASS' for v in results.values()) else 1)
