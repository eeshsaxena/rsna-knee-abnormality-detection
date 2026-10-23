#!/usr/bin/env python3
"""relabel_call.py (gold_prep/relabel, 2026-10-06): sends the pre-registered relabel prompt (prompt_v1.py) to the
Anthropic Messages API, one request per (study, arm), and writes per-label soft labels to CSV.

  arm text : report only
  arm img  : report + the key-slice PNGs listed in the keyslice_select.py manifest (base64 image blocks, inline)

MODES
  --dry_run (or --mode dry_run)  NO network, no SDK import, no key needed. Builds every request exactly as it would be
                                 sent, prints JSON sizes, token and cost estimates (also for all 4,407 studies and for the
                                 uncertain subset), writes a sample request with the image bytes elided.
  --mode sync                    Messages API, --concurrency parallel calls (the 74-study pilot: minutes, not hours).
  --mode batch_submit | batch_poll | batch_fetch
                                 Message Batches API (50% price, async, usually < 1 h, max 24 h). Batch ids are kept in
                                 <out_dir>/batches_<arm>_<model>.json.

DATA AND RULES
  - Host pinned post 733965 ("Use of Commercially Hosted LLMs", host replies 08-27): hosted LLM APIs are allowed under
    rule 2.6(b) for labelling (reasonably accessible, minimal cost), sending competition data including report text to
    such an API is not prohibited sharing, and LLM-derived labels may be used. The images are Competition Data like the
    reports (same ruling, our reading). Expected spend: a few dollars (pilot) to ~$15-130 (scale-up), see --dry_run.
  - The ONLY network destination is the Anthropic API (api.anthropic.com). The script refuses to run if
    ANTHROPIC_BASE_URL points anywhere else (unless --allow_base_url). Images are sent inline (base64), never through the
    Files API, so nothing is stored at the provider beyond the request itself (batch results are kept by the provider
    for 29 days). Study ids are replaced by opaque ids (S + 12 hex of sha256); the map stays local in id_map.csv.
  - The API key is read from the environment variable ANTHROPIC_API_KEY by the SDK. This script never prints, logs,
    stores or passes the key on the command line; it only checks that the variable is set.
  - Outputs stay local: <out_dir>/raw_<arm>_<model>.jsonl (full responses), <out_dir>/soft_<arm>_<model>.csv.

USAGE (laptop or box; the live modes need `pip install -U anthropic` and an exported ANTHROPIC_API_KEY)
  python relabel_call.py --arm img --dry_run
  python relabel_call.py --arm text --mode sync          # pilot arm 1
  python relabel_call.py --arm img  --mode sync          # pilot arm 2
  python pilot_eval.py                                   # the pre-registered gate
  python relabel_call.py --arm img --ids uncertain_ids.csv --manifest manifest_uncertain.csv --mode batch_submit
"""
import os, sys, json, math, time, base64, hashlib, inspect, argparse, threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompt_v1 as P                                      # noqa: E402
import pandas as pd                                         # noqa: E402

H = os.path.expanduser
API_HOST = 'api.anthropic.com'
# $ per million tokens (input, output). Source: Claude API skill model table, cached 2026-09-25. Batch = 50% off.
PRICES = {'claude-opus-5-5': (4.0, 20.0), 'claude-sonnet-5-5': (2.0, 10.0), 'claude-haiku-4-5': (1.0, 5.0)}
BATCH_DISCOUNT = 0.5
# Output tokens per request, GUESS: thinking (adaptive, by effort) + the ~6-label JSON (~450 tokens). Haiku: no thinking.
OUT_TOK = {'low': 900, 'medium': 1800, 'high': 3500, 'xhigh': 6000, 'max': 9000, 'haiku': 500}


def first_existing(*cands):
    for c in cands:
        if c and os.path.exists(H(c)):
            return H(c)
    return H(cands[0])


ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, choices=['text', 'img'])
ap.add_argument('--mode', default='sync', choices=['dry_run', 'sync', 'batch_submit', 'batch_poll', 'batch_fetch'])
ap.add_argument('--dry_run', action='store_true')
ap.add_argument('--ids', default='gold74', help='gold74 | gold58 | csv with StudyInstanceUID')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--model', default=None, help='default: prereg_v1.json model (claude-opus-5-5)')
ap.add_argument('--effort', default=None, help='default: prereg effort (medium); ignored for Haiku')
ap.add_argument('--max_tokens', type=int, default=None)
ap.add_argument('--manifest', default=os.path.join(HERE, 'manifest_gold74.csv'))
ap.add_argument('--png_root', default=HERE, help='directory that contains the manifest png paths')
ap.add_argument('--wire_format', default='png', choices=['png', 'jpeg'])
ap.add_argument('--train_csv', default=None, help='train.csv with the Report column')
ap.add_argument('--gold', default=None, help='gold74_eval_set.csv')
ap.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
ap.add_argument('--concurrency', type=int, default=4)
ap.add_argument('--fallbacks', default='default', choices=['default', 'none'],
                help='sync only: server-side refusal fallback (Opus 5.5 / Sonnet 5.5). Not allowed in batches.')
ap.add_argument('--batch_max_mb', type=float, default=200.0, help='split batches below the 256 MB API limit')
ap.add_argument('--prereg', default=os.path.join(HERE, 'prereg_v1.json'))
ap.add_argument('--allow_prompt_change', action='store_true')
ap.add_argument('--allow_base_url', action='store_true')
ap.add_argument('--allow_agent_shell', action='store_true',
                help='live modes refuse to run inside a Claude Code session (CLAUDECODE set): that shell carries the '
                     "session's own ANTHROPIC_API_KEY, not the pilot key")
ap.add_argument('--n_all', type=int, default=4407)
ap.add_argument('--n_uncertain', type=int, default=None, help='default: rows of uncertain_ids.csv if present, else 1500')
A = ap.parse_args()
if A.dry_run:
    A.mode = 'dry_run'

# ---------------------------------------------------------------- pre-registration
PR = json.load(open(A.prereg)) if os.path.exists(A.prereg) else None
if PR is None:
    sys.exit(f'{A.prereg} missing: run `python prompt_v1.py --write_prereg` once, before the pilot')
if PR['prompt_sha256'] != P.prompt_sha() and not A.allow_prompt_change:
    sys.exit('prompt_v1.py changed after pre-registration (sha mismatch). Restore it, or pass --allow_prompt_change '
             'and treat the run as a NEW, non-pre-registered experiment.')
MODEL = A.model or PR['model']
EFFORT = A.effort or PR['effort']
MAX_TOK = A.max_tokens or PR['max_tokens']
IS_HAIKU = MODEL.startswith('claude-haiku')
TRAIN_CSV = first_existing(A.train_csv, '~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv',
                           '~/Downloads/rsna-knee/_data/train.csv')
GOLD = first_existing(A.gold, '~/rsna_train/gold74_eval_set.csv',
                      '~/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train/gold74_eval_set.csv')
os.makedirs(A.out_dir, exist_ok=True)
TAGM = MODEL.replace('claude-', '')


def pid_of(sid):
    return 'S' + hashlib.sha256(sid.encode()).hexdigest()[:12]


def load_ids():
    if A.ids in ('gold74', 'gold58'):
        g = pd.read_csv(GOLD, dtype={'StudyInstanceUID': str})
        if A.ids == 'gold58':
            g = g[g.source == 'original_gold58']
        ids = g.StudyInstanceUID.tolist()
    else:
        ids = pd.read_csv(H(A.ids), dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    return ids[:A.limit] if A.limit else ids


TR = pd.read_csv(TRAIN_CSV, dtype={'StudyInstanceUID': str}, usecols=['StudyInstanceUID', 'Report']).set_index('StudyInstanceUID')
IDS = load_ids()
MAN = None
if A.arm == 'img':
    MAN = pd.read_csv(A.manifest, dtype={'StudyInstanceUID': str})
    miss = sorted(set(IDS) - set(MAN.StudyInstanceUID))
    assert not miss, f'{len(miss)} ids have no key slices in {A.manifest} (first {miss[:2]})'


def image_block(path):
    raw = open(path, 'rb').read()
    if A.wire_format == 'jpeg':
        import io
        from PIL import Image
        b = io.BytesIO(); Image.open(io.BytesIO(raw)).convert('L').save(b, format='JPEG', quality=95); raw = b.getvalue()
        mt = 'image/jpeg'
    else:
        mt = 'image/png'
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': mt, 'data': base64.b64encode(raw).decode('ascii')}}


def rep_text(sid):
    rep = TR.Report.get(sid, '')
    return rep if isinstance(rep, str) and rep.strip() else '(empty report)'


def build_params(sid):
    rep = rep_text(sid)
    content = [{'type': 'text', 'text': P.REPORT_BLOCK.format(pid=pid_of(sid), report=rep)}]
    meta = dict(n_img=0, img_px=[], report_chars=len(rep))
    if A.arm == 'text':
        content.append({'type': 'text', 'text': P.NO_IMAGES_NOTE})
    else:
        rows = MAN[MAN.StudyInstanceUID == sid].sort_values('img_idx')
        content.append({'type': 'text', 'text': P.IMAGES_INTRO.format(n=len(rows))})
        for k, r in enumerate(rows.itertuples(index=False), 1):
            cap = P.CAPTION.format(k=k, n=len(rows), plane_text=P.PLANE_TEXT[r.view], contrast=r.contrast,
                                   i=int(r.stack_idx) + 1, n_stack=int(r.n_stack), direction=r.direction,
                                   targets=', '.join(r.targets.split(';')))
            content.append({'type': 'text', 'text': cap})
            content.append(image_block(os.path.join(A.png_root, r.png)))
            meta['img_px'].append(int(r.height) * int(r.width))
        meta['n_img'] = len(rows)
        content.append({'type': 'text', 'text': P.IMAGES_NOTE})
    oc = {'format': {'type': 'json_schema', 'schema': P.label_schema()}}
    if not IS_HAIKU:
        oc['effort'] = EFFORT
    params = {'model': MODEL, 'max_tokens': MAX_TOK,
              'system': [{'type': 'text', 'text': P.SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
              'messages': [{'role': 'user', 'content': content}], 'output_config': oc}
    return params, meta


def custom_id(sid):
    return f'{A.arm}-{pid_of(sid)}'


# ---------------------------------------------------------------- estimates (no network)
def est_input_tokens(params, meta):
    """Rough count: text chars / 3.0 (multilingual reports; conservative), ~400 for the schema, images w*h/750."""
    txt = len(P.SYSTEM) + sum(len(b['text']) for b in params['messages'][0]['content'] if b['type'] == 'text')
    return int(txt / 3.0 + 400 + sum(math.ceil(px / 750) for px in meta['img_px']))


def out_tokens():
    return OUT_TOK['haiku'] if IS_HAIKU else OUT_TOK.get(EFFORT, 1800)


def cost(model, tin, tout, batch):
    pi, po = PRICES[model]
    f = BATCH_DISCOUNT if batch else 1.0
    return (tin * pi + tout * po) / 1e6 * f


def dry_run():
    t0 = time.time()
    sizes, toks, nimg = [], [], []
    sample = None
    for sid in IDS:
        params, meta = build_params(sid)
        b = len(json.dumps({'custom_id': custom_id(sid), 'params': params}))
        sizes.append(b); toks.append(est_input_tokens(params, meta)); nimg.append(meta['n_img'])
        if sample is None:
            sample = json.loads(json.dumps(params))
            for blk in sample['messages'][0]['content']:
                if blk['type'] == 'image':
                    blk['source']['data'] = f'<{len(blk["source"]["data"])} base64 chars elided>'
    sp = os.path.join(A.out_dir, f'sample_request_{A.arm}.json')
    json.dump(sample, open(sp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    n = len(IDS); tin = sum(toks); tout = out_tokens() * n
    print(f'DRY RUN (no network). arm {A.arm} model {MODEL} effort {EFFORT if not IS_HAIKU else "-"} max_tokens {MAX_TOK} '
          f'prompt {P.PROMPT_VERSION} sha {P.prompt_sha()[:12]} wire {A.wire_format}')
    print(f'  requests {n} | JSON bytes per request: mean {sum(sizes) / n / 1e6:.3f} MB, max {max(sizes) / 1e6:.3f} MB '
          f'(API limit 32 MB/request), total {sum(sizes) / 1e6:.1f} MB | images/request mean {sum(nimg) / n:.2f}')
    nb = max(1, math.ceil(sum(sizes) / (A.batch_max_mb * 1e6)))
    print(f'  batch split at {A.batch_max_mb:.0f} MB: {nb} batch(es) (API limit 256 MB or 100,000 requests per batch)')
    print(f'  est. input tokens: mean {tin / n:.0f}/request, total {tin / 1e6:.3f} M | est. output tokens '
          f'{out_tokens()}/request (GUESS incl. adaptive thinking), total {tout / 1e6:.3f} M')
    print(f'  sample request (image bytes elided): {sp}')
    # extrapolation: all studies and the uncertain subset, per arm, all models
    rep_all = TR.Report.fillna('').str.len()
    per_req_fixed = tin / n - sum(len(rep_text(s)) for s in IDS) / n / 3.0   # non-report part
    mean_all = per_req_fixed + rep_all.mean() / 3.0
    nunc = A.n_uncertain
    if nunc is None:
        up = os.path.join(HERE, 'uncertain_ids.csv')
        nunc = len(pd.read_csv(up)) if os.path.exists(up) else 1500
    print(f'  per-request input for train studies (all reports, mean {rep_all.mean():.0f} chars): ~{mean_all:.0f} tokens')
    print('  COST (USD; input estimated, output at the GUESS above; batch = 50% off):')
    print(f'    {"model":<18}{"this set std":>14}{"this set batch":>16}{"uncertain " + str(nunc) + " batch":>22}'
          f'{"all " + str(A.n_all) + " batch":>18}')
    for m in PRICES:
        o = OUT_TOK['haiku'] if m.startswith('claude-haiku') else OUT_TOK.get(EFFORT, 1800)
        c_std = cost(m, tin, o * n, False); c_b = cost(m, tin, o * n, True)
        c_u = cost(m, mean_all * nunc, o * nunc, True); c_a = cost(m, mean_all * A.n_all, o * A.n_all, True)
        print(f'    {m:<18}{c_std:>14.2f}{c_b:>16.2f}{c_u:>22.2f}{c_a:>18.2f}')
    lo, hi = OUT_TOK['low'], OUT_TOK['high']
    print(f'    output-token sensitivity for {MODEL}: low effort ({lo}/req) ${cost(MODEL, tin, lo * n, False):.2f} std, '
          f'high effort ({hi}/req) ${cost(MODEL, tin, hi * n, False):.2f} std for this set')
    print(f'  built {n} requests in {time.time() - t0:.1f}s. Nothing was sent.')


# ---------------------------------------------------------------- live modes
def make_client():
    if os.environ.get('CLAUDECODE') and not A.allow_agent_shell:      # verify 10-06: an agent shell has its own key
        sys.exit('refusing: this shell belongs to a Claude Code session (CLAUDECODE is set), whose environment carries '
                 "the session's own ANTHROPIC_API_KEY, not the pilot key. Run the live modes yourself in your own "
                 'terminal, or pass --allow_agent_shell deliberately.')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('ANTHROPIC_API_KEY is not set. Export it in this shell (it is read from the environment only).')
    base = os.environ.get('ANTHROPIC_BASE_URL', '')
    if base and urlparse(base).hostname != API_HOST and not A.allow_base_url:
        sys.exit(f'ANTHROPIC_BASE_URL points to {urlparse(base).hostname}, not {API_HOST}: refusing (data may only go to '
                 f'the API provider). Unset it or pass --allow_base_url deliberately.')
    import anthropic
    client = anthropic.Anthropic(max_retries=4, timeout=900.0)
    host = urlparse(str(client.base_url)).hostname
    if host != API_HOST and not A.allow_base_url:
        sys.exit(f'client base_url host {host} != {API_HOST}: refusing')
    return client


def _split_kwargs(fn, kw):
    known = set(inspect.signature(fn).parameters)
    extra = {k: kw.pop(k) for k in list(kw) if k not in known}
    if extra:
        kw['extra_body'] = {**kw.get('extra_body', {}), **extra}
    return kw


def create_message(client, params):
    """One Messages call. Opus/Sonnet 5.5 sync calls opt into the server-side refusal fallback by default
    (beta server-side-fallback-2026-07-01, fallbacks "default"); the served model is recorded per row."""
    use_fb = A.fallbacks == 'default' and (MODEL.startswith('claude-opus-5-5') or MODEL.startswith('claude-sonnet-5-5'))
    if use_fb:
        fn = client.beta.messages.create
        kw = dict(params, fallbacks='default')
        if 'betas' in inspect.signature(fn).parameters:
            kw['betas'] = ['server-side-fallback-2026-07-01']
        else:
            kw['extra_headers'] = {'anthropic-beta': 'server-side-fallback-2026-07-01'}
    else:
        fn = client.messages.create
        kw = dict(params)
    return fn(**_split_kwargs(fn, kw))


def to_dict(msg):
    for a in ('to_dict', 'model_dump'):
        if hasattr(msg, a):
            return getattr(msg, a)()
    return json.loads(json.dumps(msg, default=lambda o: o.__dict__))


def parse_message(md):
    """-> (status, obj or None, err). Validates the schema client-side (enums, int 0-100, all 6 labels)."""
    stop = md.get('stop_reason')
    if stop == 'refusal':
        return 'refusal', None, str(md.get('stop_details'))[:200]
    if stop == 'max_tokens':
        return 'max_tokens', None, ''
    for b in md.get('content', []):
        if b.get('type') != 'text':
            continue
        try:
            obj = json.loads(b['text'])
        except Exception as e:
            return 'bad_json', None, str(e)[:120]
        for k, _ in P.LABELS:
            d = obj.get(k)
            if not isinstance(d, dict):
                return 'bad_schema', None, f'missing {k}'
            if d.get('report_evidence') not in P.REPORT_EVIDENCE or d.get('image_evidence') not in P.IMAGE_EVIDENCE \
                    or d.get('call') not in P.CALLS:
                return 'bad_schema', None, f'enum {k}'
            try:
                d['probability'] = int(min(100, max(0, round(float(d['probability'])))))
            except Exception:
                return 'bad_schema', None, f'probability {k}'
        return 'ok', obj, ''
    return 'no_text', None, ''


def provenance():
    """What the request was built with. Stored on every raw row (sync) or batch record (submit), so the CSV carries
    the values the request was SENT with, not the ones of the process that later writes the CSV (verify 10-06)."""
    return dict(prompt_sha=P.prompt_sha()[:16], model_requested=MODEL, effort=None if IS_HAIKU else EFFORT)


def row_from(sid, status, obj, md, err='', prov=None):
    u = (md or {}).get('usage') or {}
    pv = prov if prov and prov.get('prompt_sha') else provenance()
    r = dict(StudyInstanceUID=sid, pid=pid_of(sid), arm=A.arm, model_requested=pv['model_requested'],
             served_model=(md or {}).get('model'), status=status, error=err,
             input_tokens=u.get('input_tokens'), output_tokens=u.get('output_tokens'),
             cache_read_tokens=u.get('cache_read_input_tokens'), prompt_sha=pv['prompt_sha'], effort=pv['effort'])
    for k, name in P.LABELS:
        d = (obj or {}).get(k) or {}
        r[name] = d['probability'] / 100.0 if 'probability' in d else float('nan')
        r[f'{name}__call'] = d.get('call'); r[f'{name}__report_ev'] = d.get('report_evidence')
        r[f'{name}__image_ev'] = d.get('image_evidence'); r[f'{name}__note'] = d.get('note')
    return r


RAW = os.path.join(A.out_dir, f'raw_{A.arm}_{TAGM}.jsonl')
CSV = os.path.join(A.out_dir, f'soft_{A.arm}_{TAGM}.csv')
_lock = threading.Lock()


def append_raw(rec):
    with _lock, open(RAW, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def done_ids():
    ok = {}
    if os.path.exists(RAW):
        for line in open(RAW, encoding='utf-8'):
            r = json.loads(line)
            if r.get('status') == 'ok':
                ok[r['sid']] = r
    return ok


def write_csv():
    rows = {}
    if os.path.exists(RAW):
        for line in open(RAW, encoding='utf-8'):
            r = json.loads(line)
            if r['sid'] not in rows or r['status'] == 'ok':      # last ok wins; failures kept only if never ok
                rows[r['sid']] = row_from(r['sid'], r['status'], r.get('obj'), r.get('message'), r.get('error', ''),
                                          r.get('prov'))
    df = pd.DataFrame(list(rows.values()))
    df.to_csv(CSV, index=False)
    st = df.status.value_counts().to_dict() if len(df) else {}
    print(f'wrote {CSV}: {len(df)} studies, status {st}')
    return df


def save_id_map():
    pd.DataFrame({'StudyInstanceUID': IDS, 'pid': [pid_of(s) for s in IDS]}).to_csv(
        os.path.join(A.out_dir, 'id_map.csv'), index=False)


def run_sync():
    client = make_client()
    save_id_map()
    ok = done_ids()
    todo = [s for s in IDS if s not in ok]
    print(f'sync: {len(todo)} to run ({len(ok)} already ok), model {MODEL}, effort {EFFORT}, arm {A.arm}, '
          f'concurrency {A.concurrency}, fallbacks {A.fallbacks}')
    t0 = time.time()

    def one(sid):
        params, _ = build_params(sid)
        try:
            md = to_dict(create_message(client, params))
            st, obj, err = parse_message(md)
        except Exception as e:                         # typed SDK errors carry status + body, never the key
            md, st, obj, err = None, f'error:{type(e).__name__}', None, str(e)[:300]
        append_raw(dict(sid=sid, custom_id=custom_id(sid), arm=A.arm, status=st, error=err, obj=obj, message=md,
                        prov=provenance(), ts=time.strftime('%Y-%m-%d %H:%M:%S')))
        return sid, st

    with ThreadPoolExecutor(A.concurrency) as ex:
        futs = [ex.submit(one, s) for s in todo]
        for i, f in enumerate(as_completed(futs), 1):
            sid, st = f.result()
            if i % 10 == 0 or st != 'ok':
                print(f'  {i}/{len(todo)} {pid_of(sid)} {st} ({time.time() - t0:.0f}s)', flush=True)
    write_csv()


BFILE = os.path.join(A.out_dir, f'batches_{A.arm}_{TAGM}.json')


def batch_submit():
    client = make_client()
    save_id_map()
    ok = done_ids()
    todo = [s for s in IDS if s not in ok]
    rec = json.load(open(BFILE)) if os.path.exists(BFILE) else []
    last_b = {}                                            # custom_id -> the LATEST batch it was queued in
    for r in rec:
        for c in r.get('sids', {}):
            last_b[c] = r['id']
    failed_b = {}                                          # custom_id -> batches whose fetched result was not ok
    if os.path.exists(RAW):
        for line in open(RAW, encoding='utf-8'):
            r = json.loads(line)
            if r.get('status') != 'ok' and r.get('batch'):
                failed_b.setdefault(r['custom_id'], set()).add(r['batch'])
    # resubmit only if never queued, or its latest batch came back not ok (errored/expired/refusal). A request whose
    # latest batch is still pending is NOT queued twice (verify 10-06: the old rule re-queued every earlier failure).
    todo = [s for s in todo if custom_id(s) not in last_b or last_b[custom_id(s)] in failed_b.get(custom_id(s), ())]
    print(f'batch_submit: {len(todo)} requests to queue ({len(ok)} already ok, {len(last_b)} already in submitted batches)')

    def submit(ch):
        bt = client.messages.batches.create(requests=[req for _, req in ch])
        rec.append(dict(id=bt.id, n=len(ch), created=time.strftime('%Y-%m-%d %H:%M:%S'), prov=provenance(),
                        sids={req['custom_id']: sid for sid, req in ch}))
        json.dump(rec, open(BFILE, 'w'), indent=1)
        print(f'submitted batch {bt.id}: {len(ch)} requests, status {bt.processing_status}', flush=True)

    cur, cur_b = [], 0                                     # streamed: at most one batch (<= --batch_max_mb) in memory
    for sid in todo:
        params, _ = build_params(sid)
        req = {'custom_id': custom_id(sid), 'params': params}
        b = len(json.dumps(req))
        if cur and (cur_b + b > A.batch_max_mb * 1e6 or len(cur) >= 100000):
            submit(cur); cur, cur_b = [], 0
        cur.append((sid, req)); cur_b += b
    if cur:
        submit(cur)


def batch_poll():
    client = make_client()
    for r in json.load(open(BFILE)):
        b = client.messages.batches.retrieve(r['id'])
        print(f'{r["id"]}: {b.processing_status} counts {to_dict(b.request_counts)}')


def batch_fetch():
    client = make_client()
    cid2sid = {custom_id(s): s for s in IDS}
    for r in json.load(open(BFILE)):
        cid2sid.update(r.get('sids', {}))
        b = client.messages.batches.retrieve(r['id'])
        if b.processing_status != 'ended':
            print(f'{r["id"]}: {b.processing_status}, skipped (fetch again later)'); continue
        n = 0
        for res in client.messages.batches.results(r['id']):
            sid = cid2sid.get(res.custom_id)
            if sid is None:
                continue
            t = res.result.type
            if t == 'succeeded':
                md = to_dict(res.result.message); st, obj, err = parse_message(md)
            else:
                md, obj = None, None
                st = f'batch_{t}'
                err = str(getattr(res.result, 'error', ''))[:300]
            append_raw(dict(sid=sid, custom_id=res.custom_id, arm=A.arm, status=st, error=err, obj=obj, message=md,
                            batch=r['id'], prov=r.get('prov'), ts=time.strftime('%Y-%m-%d %H:%M:%S')))
            n += 1
        print(f'{r["id"]}: {n} results fetched')
    write_csv()


if __name__ == '__main__':
    print(f'[relabel_call] mode {A.mode} arm {A.arm} ids {A.ids} n {len(IDS)} model {MODEL} prereg sha ok '
          f'({P.prompt_sha()[:12]}); network destination: {API_HOST} only; key: from env, never printed')
    {'dry_run': dry_run, 'sync': run_sync, 'batch_submit': batch_submit, 'batch_poll': batch_poll,
     'batch_fetch': batch_fetch}[A.mode]()
