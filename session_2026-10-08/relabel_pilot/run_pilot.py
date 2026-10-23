#!/usr/bin/env python3
"""run_pilot.py (gold_prep/relabel, 2026-10-08): ONE command for the whole relabel pilot under prereg_v2.json, with the
$8.00 hard cap and the pre-registered staging enforced by code. Run it in a shell where YOUR ANTHROPIC_API_KEY is
exported (the key is read from the environment by the SDK only; this script never prints, logs or stores it).

  python run_pilot.py                      # Eesh's own terminal (CLAUDECODE not set)
  python run_pilot.py --allow_agent_shell  # only if that shell is a Claude Code child AND holds Eesh's real key

What it does (prereg_v2.json "spend" + "gate"; resumable: rerun after Ctrl-C or a crash and it picks up where it was)
  0. integrity: prereg_v2.json sha == prereg_v2.sha256, prompt sha == prereg, relabel_call.py / pilot_eval.py sha ==
     the ones registered in prereg_v2.json; then ONE free auth check (models.retrieve) so a bad key fails before any spend
  1. Stage A: the first 10 gold58 studies, both arms, Message Batches (half price); poll every --poll_s; fetch
  2. actual spend from the fetched usage (spend_tally.py logic); projection = actual + 1.3 x mean cost/request x
     requests still to submit; rest of gold74 if <= cap, else rest of gold58 only if <= cap, else STOP (no model switch);
     projection > $6.00 -> chunks of <= 20 per arm with a re-check before each; a failed request is retried at most once
  3. pilot_eval.py --prereg prereg_v2.json -> results/pilot_eval_v2.log + results/pilot_eval_opus-5-5_v2.json
  4. copies prereg, CSVs, raw responses, batch ids, logs and spend to --save_to
"""
import os, sys, io, json, time, glob, shutil, hashlib, runpy, argparse, contextlib, subprocess
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd                                         # noqa: E402
import prompt_v1 as P                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--allow_agent_shell', action='store_true')
ap.add_argument('--poll_s', type=float, default=300.0, help='seconds between batch polls (default 5 min)')
ap.add_argument('--max_wait_h', type=float, default=26.0)
ap.add_argument('--cap', type=float, default=None, help='may only LOWER the prereg cap')
ap.add_argument('--chunk_above', type=float, default=None, help='testing only: lower the $6.00 chunking threshold')
ap.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
ap.add_argument('--save_to', default=os.path.expanduser('~/Downloads/rsna-knee/session_2026-10-08/relabel_pilot'))
ap.add_argument('--no_save', action='store_true')
A = ap.parse_args()

PRF = os.path.join(HERE, 'prereg_v2.json')
PR = json.load(open(PRF, encoding='utf-8'))
CAP_PR = float(PR['spend']['cap_usd'])
CAP = CAP_PR if A.cap is None else float(A.cap)
if CAP > CAP_PR:
    sys.exit(f'--cap {CAP} above the registered cap {CAP_PR}: refused')
CHUNK_ABOVE = 6.0 if A.chunk_above is None else min(6.0, A.chunk_above)
MARGIN, CHUNK, STAGE_A, MAX_ATTEMPTS = 1.3, 20, 10, 2
PX = PR['spend']['prices_usd_per_mtok']
MODEL = PR['model']
TAG = MODEL.replace('claude-', '')
ARMS = ('text', 'img')
os.makedirs(A.out_dir, exist_ok=True)
LOG = os.path.join(A.out_dir, 'run_pilot.log')
DS = '~/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
GOLD = os.path.expanduser(f'{DS}/gold74_eval_set.csv')


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def pid_of(sid):
    return 'S' + hashlib.sha256(sid.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ 0. integrity + auth
def integrity():
    reg = open(os.path.join(HERE, 'prereg_v2.sha256')).read().split()[0]
    assert sha(PRF) == reg, 'prereg_v2.json changed after registration'
    assert PR['prompt_sha256'] == P.prompt_sha(), 'prompt_v1.py differs from the registered prompt sha'
    assert sha(os.path.join(HERE, 'relabel_call.py')) == PR['code']['relabel_call_py_sha256'], 'relabel_call.py changed'
    assert sha(os.path.join(HERE, 'pilot_eval.py')) == PR['code']['pilot_eval_py_sha256'], 'pilot_eval.py changed'
    log(f'integrity ok: prereg_v2 {reg[:16]}, prompt {P.prompt_sha()[:12]}, model {MODEL}, cap ${CAP:.2f}')


def auth_check():
    if os.environ.get('CLAUDECODE') and not A.allow_agent_shell:
        sys.exit('refusing: CLAUDECODE is set (a Claude Code shell). Run in your own terminal, or pass --allow_agent_shell '
                 'if this shell really holds your key.')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('ANTHROPIC_API_KEY is not set in this shell.')
    base = os.environ.get('ANTHROPIC_BASE_URL', '')
    if base and urlparse(base).hostname != 'api.anthropic.com':
        sys.exit('ANTHROPIC_BASE_URL is not api.anthropic.com: refusing.')
    import anthropic
    try:
        m = anthropic.Anthropic(max_retries=2, timeout=60.0).models.retrieve(MODEL)     # free; verifies the key
    except anthropic.AuthenticationError:
        sys.exit('auth check FAILED (401): the ANTHROPIC_API_KEY in this shell is not a valid API key. Nothing was sent.')
    log(f'auth check ok (models.retrieve {getattr(m, "id", MODEL)}; free call, no tokens)')


# ------------------------------------------------------------------ relabel_call in-process (same as the selftest)
def call(arm, mode, ids_csv=None):
    argv = ['relabel_call.py', '--prereg', PRF, '--arm', arm, '--mode', mode, '--out_dir', A.out_dir,
            '--ids', ids_csv or 'gold74']
    if A.allow_agent_shell:
        argv.append('--allow_agent_shell')
    out = io.StringIO()
    old = sys.argv
    sys.argv = argv
    code = 0
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(os.path.join(HERE, 'relabel_call.py'), run_name='__main__')
    except SystemExit as e:
        code = e.code if e.code is not None else 0
    finally:
        sys.argv = old
    text = out.getvalue()
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text)
    if code not in (0, None):
        log(f'relabel_call {arm} {mode} exited: {code}')
        sys.exit(f'stopped: relabel_call {arm} {mode} failed ({code})')
    return text


def batches(arm):
    p = os.path.join(A.out_dir, f'batches_{arm}_{TAG}.json')
    return json.load(open(p)) if os.path.exists(p) else []


def attempts(arm):
    n = {}
    for r in batches(arm):
        for sid in r.get('sids', {}).values():
            n[sid] = n.get(sid, 0) + 1
    return n


def ok_set(arm):
    p = os.path.join(A.out_dir, f'raw_{arm}_{TAG}.jsonl')
    s = set()
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r.get('status') == 'ok':
                s.add(r['sid'])
    return s


def fetched_batches(arm):
    p = os.path.join(A.out_dir, f'raw_{arm}_{TAG}.jsonl')
    s = set()
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            b = json.loads(line).get('batch')
            if b:
                s.add(b)
    return s


def wait_and_fetch():
    t0 = time.time()
    while True:
        pending = []
        for arm in ARMS:
            if not batches(arm):
                continue
            txt = call(arm, 'batch_poll')
            for line in txt.splitlines():
                if ' counts ' in line and ': ended ' not in line:
                    pending.append(line.strip())
        if not pending:
            break
        if time.time() - t0 > A.max_wait_h * 3600:
            sys.exit(f'stopped: batches still pending after {A.max_wait_h} h; rerun later (resumable)')
        log(f'{len(pending)} batch(es) not ended yet; next poll in {A.poll_s:.0f}s ({pending[0][:90]})')
        time.sleep(A.poll_s)
    for arm in ARMS:
        if batches(arm) and {r['id'] for r in batches(arm)} - fetched_batches(arm):
            call(arm, 'batch_fetch')


# ------------------------------------------------------------------ spend
def spend():
    tot, per = 0.0, {}
    for arm in ARMS:
        p = os.path.join(A.out_dir, f'raw_{arm}_{TAG}.jsonl')
        seen, n, c = set(), 0, 0.0
        tok = dict(input=0, cache_creation=0, cache_read=0, output=0)
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                key = (r.get('batch') or 'sync:' + r.get('ts', ''), r['custom_id'])
                if key in seen:
                    continue
                seen.add(key)
                u = ((r.get('message') or {}).get('usage')) or {}
                if not u:
                    continue
                n += 1
                f = PX['batch_discount'] if r.get('batch') else 1.0
                c += f * ((u.get('input_tokens') or 0) * PX['input'] + (u.get('cache_creation_input_tokens') or 0)
                          * PX['cache_write_5m'] + (u.get('cache_read_input_tokens') or 0) * PX['cache_read']
                          + (u.get('output_tokens') or 0) * PX['output']) / 1e6
                for k, uk in (('input', 'input_tokens'), ('cache_creation', 'cache_creation_input_tokens'),
                              ('cache_read', 'cache_read_input_tokens'), ('output', 'output_tokens')):
                    tok[k] += u.get(uk) or 0
        per[arm] = dict(n_billed=n, usd=round(c, 4), usd_per_req=(c / n) if n else None, tokens=tok,
                        mean_output_tokens=round(tok['output'] / n) if n else None)
        tot += c
    return tot, per


def submit(arm, sids, label):
    if not sids:
        return
    p = os.path.join(A.out_dir, f'_ids_{arm}_{label}.csv')
    pd.DataFrame({'StudyInstanceUID': sids}).to_csv(p, index=False)
    call(arm, 'batch_submit', p)
    log(f'submitted {arm} {label}: {len(sids)} requests')


# ------------------------------------------------------------------ main
def main():
    integrity()
    auth_check()
    ids74 = pd.read_csv(GOLD, dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    assert len(ids74) == 74
    # Stage A
    for arm in ARMS:
        q = attempts(arm)
        todo = [s for s in ids74[:STAGE_A] if s not in q]
        submit(arm, todo, 'stageA')
    wait_and_fetch()
    scope, rnd = None, 0
    while True:
        rnd += 1
        actual, per = spend()
        if actual > CAP:
            log(f'STOP: actual spend ${actual:.4f} above cap ${CAP:.2f} (should not happen)'); break
        for arm in ARMS:
            if not per[arm]['usd_per_req']:
                sys.exit(f'stopped: no billed {arm} result to measure cost from (all Stage A requests failed?)')

        def plan(n):
            ok = {a: ok_set(a) for a in ARMS}
            att = {a: attempts(a) for a in ARMS}
            return {a: [s for s in ids74[:n] if s not in ok[a] and att[a].get(s, 0) < MAX_ATTEMPTS] for a in ARMS}

        def proj(pl):
            return actual + MARGIN * sum(len(pl[a]) * per[a]['usd_per_req'] for a in ARMS)

        if scope is None:
            p74, p58 = proj(plan(74)), proj(plan(58))
            log(f'after stage A: actual ${actual:.4f}; mean $/req text {per["text"]["usd_per_req"]:.5f} img '
                f'{per["img"]["usd_per_req"]:.5f}; mean output tokens text {per["text"]["mean_output_tokens"]} img '
                f'{per["img"]["mean_output_tokens"]}; projection gold74 ${p74:.4f}, gold58 ${p58:.4f}, cap ${CAP:.2f}')
            scope = 74 if p74 <= CAP else (58 if p58 <= CAP else None)
            if scope is None:
                log('STOP: even the rest of gold58 projects above the cap. No model switch (prereg). Report to Eesh.')
                break
            log(f'scope: gold{scope}')
        pl = plan(scope)
        if not any(pl.values()):
            log('all requests in scope done (or retried once)'); break
        pr = proj(pl)
        if pr > CAP:
            if scope == 74 and proj(plan(58)) <= CAP:
                scope = 58; log(f'projection ${pr:.4f} > cap: scope narrowed to gold58'); pl = plan(58); pr = proj(pl)
                if not any(pl.values()):
                    log('gold58 done'); break
            else:
                log(f'STOP: projection ${pr:.4f} > cap ${CAP:.2f}'); break
        chunked = pr > CHUNK_ABOVE
        for arm in ARMS:
            submit(arm, pl[arm][:CHUNK] if chunked else pl[arm], f'r{rnd}')
        log(f'round {rnd}: projection ${pr:.4f} (cap ${CAP:.2f}){" chunked" if chunked else ""}')
        wait_and_fetch()
        if rnd > 12:
            log('STOP: too many rounds'); break
    actual, per = spend()
    json.dump(dict(actual_usd=round(actual, 4), cap_usd=CAP, per_arm=per, prices=PX,
                   time=time.strftime('%Y-%m-%d %H:%M:%S %z')), open(os.path.join(A.out_dir, 'spend.json'), 'w'), indent=1)
    log(f'FINAL actual spend ${actual:.4f} (cap ${CAP:.2f}); per arm ' +
        ', '.join(f'{a} {per[a]["n_billed"]} billed ${per[a]["usd"]:.4f}' for a in ARMS))
    pd.DataFrame({'StudyInstanceUID': ids74, 'pid': [pid_of(s) for s in ids74]}).to_csv(
        os.path.join(A.out_dir, 'id_map.csv'), index=False)
    # 3. eval
    ev = subprocess.run([sys.executable, os.path.join(HERE, 'pilot_eval.py'), '--model', TAG, '--prereg', PRF,
                         '--text', os.path.join(A.out_dir, f'soft_text_{TAG}.csv'),
                         '--img', os.path.join(A.out_dir, f'soft_img_{TAG}.csv'),
                         '--out', os.path.join(A.out_dir, f'pilot_eval_{TAG}_v2.json')],
                        capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                             if k != 'ANTHROPIC_API_KEY'})
    open(os.path.join(A.out_dir, 'pilot_eval_v2.log'), 'w', encoding='utf-8').write(ev.stdout + ev.stderr)
    print(ev.stdout + ev.stderr)
    log(f'pilot_eval exit {ev.returncode}')
    # 4. save
    if not A.no_save:
        os.makedirs(A.save_to, exist_ok=True)
        files = [PRF, os.path.join(HERE, 'prereg_v2.sha256'), os.path.join(HERE, 'pilot_eval.py'),
                 os.path.join(HERE, 'run_pilot.py'), os.path.join(HERE, 'spend_tally.py')]
        files += [f for f in glob.glob(os.path.join(A.out_dir, '*')) if os.path.isfile(f)]
        for f in files:
            shutil.copy2(f, A.save_to)
        log(f'saved {len(files)} files to {A.save_to}')


if __name__ == '__main__':
    main()
