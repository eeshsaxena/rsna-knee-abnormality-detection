#!/usr/bin/env python
# watch.py (2026-10-14, Claude subagent; RESEARCH_960 lever 3): public-release watcher for rsna-knee-abnormality-detection.
# READ-ONLY: lists, pulls metadata, reads public pages. Never pushes, submits, creates or versions anything.
#
# One run:
#  (a) lists the competition's public notebooks by public score (kernels list --competition ... --sort-by scoreDescending,
#      --score-pages pages of 100) and by recency (--sort-by dateRun, --recent-pages pages), plus every competition notebook
#      of the key authors (heliosli, nartaa, varshithakolupuri, jingshuifu, sorted by dateRun);
#  (b) for every notebook not seen before (and for seen ones that may have changed: new run time, unknown score younger than
#      72 h, current top --recheck-top by score, key authors) reads the public score and the pinned inputs from Kaggle's own
#      notebook page model (anonymous GetKernelViewModel = the score the page shows; the built-in browser is the fallback:
#      refs that could not be read go to browser_todo.json, results come back with --browser-scores FILE), records its
#      inputs via `kaggle kernels pull REF -m` (python API, same call) for every notebook with score >= --pull-min, unknown
#      score, or a key author, and tests every input anonymously (dataset public + its current version vs the version the
#      scored run pinned; kernel sources public; model sources visible);
#  (c) flags RELEASE_CANDIDATE = public score >= --flag-min (.957) AND every input public AND only this competition's data;
#      HIGH_SCORE_NOT_COPYABLE = score >= .957 but some input is not public; WATCH alerts for new inputs (a dataset no public
#      notebook used before) on .955+ or unscored new notebooks, new versions of key-author notebooks, scores that appear;
#  (d) writes watch_state.json (machine state, cumulative) and appends one section to WATCH_LOG.md;
#  plus the dataset watch: every dataset of heliosli / nartaa (= `kaggle datasets list --user U`) and the RESEARCH_960
#  baseline datasets: new dataset, version bump, size change -> alert.
# Exit code: 10 if a RELEASE_CANDIDATE was raised in this run, 0 otherwise (2 = fatal error).
#   usage: python watch.py [--score-pages 3] [--recent-pages 2] [--flag-min 0.957] [--browser-scores FILE] [--loop-hours H]
import os, sys, json, time, argparse, datetime, traceback, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)            # our own helper only (works under python -I too); pulled files live in subdirs
import kgweb                        # noqa: E402

KEY_AUTHORS = ['heliosli', 'nartaa', 'varshithakolupuri', 'jingshuifu']
DATASET_USERS = ['heliosli', 'nartaa']
BASELINE_DATASETS = ['heliosli/rsna-knee-compact-r7cnxfsss', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
                     'nartaa/rsna-knee-publication-swa-weights-20261007', 'dreaddevelopment/raptor-knee-widedense',
                     'jessky/rsna-knee-local-cv-cnx50']
BROWSER_JS = ("await new Promise(r => setTimeout(r, 4500)); JSON.stringify([...document.querySelectorAll('a')]"
              ".filter(a => /scriptVersionId=/.test(a.href)).map(a => a.innerText.replace(/\\s+/g, ' ').trim())"
              ".filter(t => /^0\\.9\\d\\d\\b/.test(t)))")

ap = argparse.ArgumentParser()
ap.add_argument('--score-pages', type=int, default=3)
ap.add_argument('--recent-pages', type=int, default=2)
ap.add_argument('--page-size', type=int, default=100)
ap.add_argument('--flag-min', type=float, default=0.957)
ap.add_argument('--pull-min', type=float, default=0.955)
ap.add_argument('--recheck-top', type=int, default=60)
ap.add_argument('--pull-all', action='store_true', help='pull -m every new notebook, whatever its score')
ap.add_argument('--browser-scores', default=None, help='JSON {ref: "0.956 V1" | 0.956 | null} read in the built-in browser')
ap.add_argument('--loop-hours', type=float, default=0, help='repeat every H hours (dies with the session: re-arm)')
ap.add_argument('--no-log', action='store_true')
ap.add_argument('--dir', default=HERE, help='where watch_state.json, WATCH_LOG.md, browser_todo.json and pulled/ live')
A = ap.parse_args()
STATE_P = os.path.join(A.dir, 'watch_state.json')
LOG_P = os.path.join(A.dir, 'WATCH_LOG.md')
TODO_P = os.path.join(A.dir, 'browser_todo.json')
PULL_DIR = os.path.join(A.dir, 'pulled')      # untrusted data: parsed as JSON / text only, never executed
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t=None):
    return (t or now()).strftime('%Y-%m-%dT%H:%M:%SZ')


def ist(t=None):
    return ((t or now()) + datetime.timedelta(hours=5, minutes=30)).strftime('%m-%d %H:%M IST')


def tstr(x):
    return x.isoformat() if hasattr(x, 'isoformat') else (str(x) if x is not None else None)


def age_s(s):
    """Seconds since an ISO time ('...Z', '+00:00' or naive UTC); None if unparseable."""
    if not s:
        return None
    try:
        t = datetime.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        return (now() - t).total_seconds()
    except ValueError:
        return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_state():
    if os.path.isfile(STATE_P):
        return json.load(open(STATE_P, encoding='utf-8'))
    return {'version': 1, 'created': iso(), 'competition': kgweb.COMP, 'kernels': {}, 'inputs_seen': {}, 'datasets': {},
            'dataset_ids': {}, 'alerts': [], 'runs': []}


def save_state(st):
    tmp = STATE_P + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=1, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, STATE_P)


def run_once():
    t0 = now()
    st = load_state()
    st.setdefault('dataset_ids', {})
    first_run = not st['kernels']
    api = kgweb.kaggle_api()
    web = kgweb.Web()
    alerts, errors = [], []
    K = st['kernels']

    def alert(kind, ref, msg, level='WATCH'):
        a = {'utc': iso(), 'level': level, 'kind': kind, 'ref': ref, 'msg': msg}
        alerts.append(a)
        print('WATCH_%s %s %s %s' % ('FLAG' if level == 'FLAG' else 'ALERT', kind, ref, msg), flush=True)

    # ---- (a) listings ------------------------------------------------------------------------------------------------------
    listed = {}          # ref -> {'title', 'author', 'last_run_time', 'votes', 'score_pos', 'recent_pos', 'via'}
    def take(rows, via, pos_key=None, base=0):
        for i, k in enumerate(rows or []):
            if k is None:
                continue
            e = listed.setdefault(k.ref, {'title': k.title, 'author': k.author, 'last_run_time': tstr(k.last_run_time),
                                          'votes': k.total_votes, 'via': []})
            if via not in e['via']:
                e['via'].append(via)
            if pos_key and e.get(pos_key) is None:
                e[pos_key] = base + i + 1

    n_list_calls = 0
    for sort_by, pages, key in (('scoreDescending', A.score_pages, 'score_pos'), ('dateRun', A.recent_pages, 'recent_pos')):
        for p in range(1, pages + 1):
            try:
                rows = api.kernels_list(competition=kgweb.COMP, sort_by=sort_by, page_size=A.page_size, page=p) or []
                n_list_calls += 1
            except Exception as e:
                errors.append('kernels_list %s p%d: %s' % (sort_by, p, e))
                break
            take(rows, sort_by, key, (p - 1) * A.page_size)
            if len(rows) < A.page_size:
                break
    for u in KEY_AUTHORS:
        try:
            take(api.kernels_list(competition=kgweb.COMP, user=u, sort_by='dateRun', page_size=100) or [], 'author:' + u)
            n_list_calls += 1
        except Exception as e:
            errors.append('kernels_list user %s: %s' % (u, e))
    print('WATCH listed %d unique notebooks (%d list calls)' % (len(listed), n_list_calls), flush=True)

    # ---- (b) which ones to (re)read -------------------------------------------------------------------------------------------
    top_known = sorted([r for r in K if fnum(K[r].get('score')) is not None], key=lambda r: -fnum(K[r]['score']))[:A.recheck_top]
    todo = {}
    for ref, e in listed.items():
        k = K.get(ref)
        author = ref.split('/')[0]
        if k is None:
            todo[ref] = 'new'
        elif e['last_run_time'] != k.get('last_run_time'):
            todo[ref] = 'new_run'
        elif k.get('score') is None and (age_s(((k.get('kvm') or {}).get('run') or {}).get('date')) or 1e12) < 72 * 3600:
            todo[ref] = 'unscored_recent'
        elif ref in top_known:
            todo[ref] = 'top'
        elif author in KEY_AUTHORS:
            todo[ref] = 'key_author'
    for ref in top_known:                       # top notebooks that fell out of the listing pages are still re-read
        todo.setdefault(ref, 'top')
    print('WATCH to read: %d (%s)' % (len(todo), {w: list(todo.values()).count(w) for w in set(todo.values())}), flush=True)

    vis_cache = {}

    def visibility(kind, ref):
        key = kind + ':' + ref
        if key in vis_cache:
            return vis_cache[key]
        out = {'state': 'unknown'}
        if kind == 'competition':
            out = {'state': 'ours' if ref == kgweb.COMP else 'other_competition'}
        elif kind == 'dataset':
            s, j = web.dataset(ref)
            if s == 200 and j:
                out = {'state': 'public', 'cur_version_id': j.get('datasetVersionId'), 'cur_version': j.get('datasetVersionNumber'),
                       'last_update': j.get('lastUpdateTime'), 'bytes': (j.get('data') or {}).get('totalSize'), 'title': j.get('title')}
            elif s == 404:
                out = {'state': 'not_public'}
            else:
                out = {'state': 'unknown', 'http': s}
        elif kind == 'kernel':
            s, j = web.kvm(ref)
            out = {'state': 'public' if s == 200 else ('not_public' if s == 404 else 'unknown'), 'http': s}
        elif kind == 'model':
            try:
                parts = ref.split('/')
                api.model_get('/'.join(parts[:2]))
                out = {'state': 'visible_to_eesh', 'note': 'authenticated check (others\' private models are not visible)'}
            except Exception as e:
                out = {'state': 'not_public', 'err': str(e)[:120]}
        vis_cache[key] = out
        return out

    n_pull, n_read = 0, 0
    for ref, why in todo.items():
        e = listed.get(ref, {})
        prev = K.get(ref)
        rec = prev if prev is not None else {'first_seen': iso(), 'history': []}
        rec.update({k: e[k] for k in ('title', 'author', 'last_run_time', 'votes') if k in e})
        rec['list_pos'] = {'score': e.get('score_pos'), 'recent': e.get('recent_pos'), 'via': e.get('via', [])}
        s, j = web.kvm(ref)
        n_read += 1
        if s == 200 and j:
            kv = kgweb.summarize_kvm(j)
            rec['kvm'] = kv
            rec['kvm_checked'] = iso()
            rec.pop('kvm_error', None)
            rec.pop('gone', None)
            score, src = fnum(kv['best_public']), 'kvm'
        else:
            rec['kvm_error'] = {'utc': iso(), 'http': s}
            score, src = rec.get('score'), (rec.get('score_src') or 'unknown')
            if s == 404:
                rec['gone'] = iso()       # deleted or made private since it was listed
            errors.append('kvm %s http %s' % (ref, s))
        old_score, old_ver = (prev or {}).get('score'), ((prev or {}).get('kvm') or {}).get('cur_version')
        rec['score'], rec['score_src'] = score, (src if score is not None else ('unknown' if s == 200 else src))
        kv = rec.get('kvm') or {}
        if prev is not None and (score != old_score or kv.get('cur_version') != old_ver):
            rec['history'].append({'utc': iso(), 'best_public': score, 'cur_version': kv.get('cur_version'),
                                   'best_version': kv.get('best_version')})
            if score is not None and old_score is None:
                alert('SCORE_APPEARED', ref, 'public score %.3f (v%s)' % (score, kv.get('best_version')))
            elif score is not None and old_score is not None and score > old_score:
                alert('SCORE_UP', ref, '%.3f -> %.3f' % (old_score, score))
            if kv.get('cur_version') != old_ver and ref.split('/')[0] in KEY_AUTHORS:
                alert('KEY_AUTHOR_NEW_VERSION', ref, 'v%s -> v%s (score %s)' % (old_ver, kv.get('cur_version'), score))

        bv, cv = kv.get('best_version'), kv.get('cur_version')
        author = ref.split('/')[0]

        # pull -m (= kaggle kernels pull REF -m: the CURRENT version) for the notebooks that matter
        age = age_s((kv.get('run') or {}).get('date') or rec.get('last_run_time'))
        recent = age is None or age < 7 * 86400
        # unscored notebooks are pulled only when they ran in the last 7 days (a release may be published unsubmitted)
        want_pull = A.pull_all or (score is None and recent) or (score is not None and score >= A.pull_min) or author in KEY_AUTHORS
        if want_pull and s == 200 and (rec.get('meta_version') != cv or 'meta' not in rec):
            try:
                text, meta, fname = kgweb.get_kernel(api, ref)
                d = os.path.join(PULL_DIR, ref.replace('/', '__'))
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, fname), 'w', encoding='utf-8', newline='') as f:
                    f.write(text)
                with open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(kgweb.plain(meta), f, indent=2)
                rec['meta'] = {k: kgweb.plain(meta[k]) for k in ('dataset_sources', 'kernel_sources', 'competition_sources',
                                                                 'model_sources', 'docker_image', 'machine_shape',
                                                                 'enable_internet', 'enable_gpu', 'kernel_type', 'language')}
                rec['meta_version'] = cv
                rec['source_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
                rec['pulled'] = os.path.relpath(d, A.dir)
                n_pull += 1
            except Exception as ex:
                errors.append('pull %s: %s' % (ref, str(ex)[:160]))
        meta = rec.get('meta') if rec.get('meta_version') == cv else None

        def resolve(sd):
            """'?/SLUG' (bare mount) -> OWNER/SLUG via the dataset-id map or the pulled metadata; else stays unresolved."""
            if not sd['ref'].startswith('?/'):
                return sd
            slug = sd['ref'][2:]
            cand = None
            if sd['kind'] == 'dataset' and sd.get('dataset_id') is not None:
                cand = st['dataset_ids'].get(str(sd['dataset_id']))
            if cand is None and meta:
                pool = meta.get('dataset_sources') if sd['kind'] == 'dataset' else meta.get('kernel_sources')
                hits = [x for x in (pool or []) if x and x.split('/')[-1] == slug]
                cand = hits[0] if len(hits) == 1 else None
            return dict(sd, ref=cand, resolved_from='?/' + slug) if cand else sd

        cur_srcs = [resolve(x) for x in (kv.get('sources') or [])]
        for x in cur_srcs:
            if x['kind'] == 'dataset' and not x['ref'].startswith('?/') and x.get('dataset_id') is not None:
                st['dataset_ids'][str(x['dataset_id'])] = x['ref']
        # inputs = the pinned sources of the SCORED version (its own run), else the current run
        srcs, scored_from = cur_srcs, 'current_run'
        if bv and cv and bv != cv and kv.get('kernel_id'):
            sv, vj = web.versions(kv['kernel_id'])
            vt = kgweb.version_table(vj) if sv == 200 else {}
            if int(bv) in vt:
                srcs, scored_from = [resolve(x) for x in vt[int(bv)]['sources']], 'scored_version_run'
                rec['scored_version_run'] = {k: vt[int(bv)][k] for k in ('run_id', 'date', 'run_time_seconds', 'docker_digest', 'accelerator')}
            else:
                scored_from = 'current_run (scored version v%s not listed)' % bv
        detail, drift, all_public, new_inputs = {}, {}, True if srcs else None, []
        for sd in srcs:
            v = visibility(sd['kind'], sd['ref']) if not sd['ref'].startswith('?/') else {'state': 'unresolved_owner'}
            detail[sd['kind'] + ':' + sd['ref']] = v['state']
            if v['state'] not in ('public', 'ours'):
                all_public = False
            if sd['kind'] == 'dataset' and v['state'] == 'public' and sd.get('version_id') and v.get('cur_version_id')                     and int(sd['version_id']) != int(v['cur_version_id']):
                drift[sd['ref']] = {'pinned_version_id': sd['version_id'], 'current_version_id': v['cur_version_id'],
                                    'current_version': v.get('cur_version')}
            if sd['kind'] in ('dataset', 'model', 'kernel') and not sd['ref'].startswith('?/'):
                key = sd['kind'] + ':' + sd['ref']
                if key not in st['inputs_seen']:
                    st['inputs_seen'][key] = {'first_seen': iso(), 'first_ref': ref, 'first_score': score}
                    new_inputs.append(key)
        rec['inputs'] = {'all_public': all_public, 'detail': detail, 'drift': drift, 'n': len(srcs), 'from': scored_from}
        if meta:
            # pull -m slugs vs the CURRENT run's pinned sources (a mismatch = metadata edited after the last run)
            ms = set('dataset:' + x for x in meta.get('dataset_sources') or [] if x)
            rs = set('dataset:' + x['ref'] for x in cur_srcs if x['kind'] == 'dataset')
            if ms != rs:
                rec['inputs']['meta_vs_current_run'] = {'only_in_meta': sorted(ms - rs), 'only_in_run': sorted(rs - ms)}
            if meta.get('enable_internet'):
                rec['inputs']['internet_on'] = True

        # (c) flags
        flag = None
        if score is not None and score >= A.flag_min - 1e-9:
            only_ours = all(x['kind'] != 'competition' or x['ref'] == kgweb.COMP for x in srcs)
            flag = 'RELEASE_CANDIDATE' if (all_public and only_ours) else 'HIGH_SCORE_NOT_COPYABLE'
        if flag and flag != rec.get('flag'):
            build = 'python session_2026-10-15/watch/swap_builder.py %s/%s --name rsna-sub-rel-%s' % (
                ref, bv, ('%s-%s' % (author, ref.split('/')[1]))[:30].lower().strip('-'))
            alert(flag, ref, 'score %.3f v%s (current v%s), inputs %s%s%s' % (
                score, bv, cv, 'ALL PUBLIC' if all_public else detail, (', DRIFT ' + json.dumps(drift)) if drift else '',
                ('; next: ' + build) if flag == 'RELEASE_CANDIDATE' else ''),
                level='FLAG' if flag == 'RELEASE_CANDIDATE' else 'WATCH')
            if flag == 'RELEASE_CANDIDATE':
                # a file any poller can test for (the loop mode never exits); removed by hand once handled
                with open(os.path.join(A.dir, 'RELEASE_FLAG.txt'), 'a', encoding='utf-8') as f:
                    f.write('%s RELEASE_CANDIDATE %s score %.3f scored v%s current v%s drift %s\n  next: %s\n' % (
                        iso(), ref, score, bv, cv, json.dumps(drift), build))
        rec['flag'] = flag
        if new_inputs and not first_run and (score is None or score >= A.pull_min):
            alert('NEW_INPUT', ref, 'score %s uses never-seen %s' % (score, new_inputs))
        if prev is None and not first_run and author in KEY_AUTHORS:
            alert('KEY_AUTHOR_NEW_NOTEBOOK', ref, 'score %s, inputs %s' % (score, detail))
        K[ref] = rec
        if n_read % 50 == 0:
            print('WATCH read %d/%d (kvm calls %d, pulls %d)' % (n_read, len(todo), web.calls, n_pull), flush=True)
            save_state(st)

    # ---- browser fallback: merge scores read in the built-in browser; list what still has no readable page model ----------
    if A.browser_scores:
        bs = json.load(open(A.browser_scores, encoding='utf-8'))
        for ref, v in bs.items():
            if ref not in K:
                continue
            val = v[0] if isinstance(v, list) and v else v
            sc = fnum(str(val).split()[0]) if val not in (None, '', []) else None
            K[ref]['browser'] = {'utc': iso(), 'raw': v, 'score': sc}
            if not K[ref].get('kvm_checked') or K[ref].get('kvm_error'):     # page model never read: the browser decides
                K[ref]['score'], K[ref]['score_src'] = sc, ('browser' if sc is not None else 'unknown')
            else:
                km = K[ref].get('score')
                agree = (sc is None and km is None) or (sc is not None and km is not None and abs(sc - km) < 1e-9)
                K[ref]['browser']['agrees_with_page_model'] = agree
                if not agree:
                    alert('BROWSER_KVM_MISMATCH', ref, 'browser %s vs page model %s' % (sc, km))
    todo_b = sorted(r for r, k in K.items() if k.get('kvm_error') and k.get('score_src') != 'kvm' and not k.get('gone'))
    json.dump({'utc': iso(), 'refs': todo_b, 'how': 'built-in browser: navigate https://www.kaggle.com/code/<ref>, then '
               'javascript_tool: ' + BROWSER_JS + ' ; save {ref: result} and rerun watch.py --browser-scores FILE'},
              open(TODO_P, 'w', encoding='utf-8'), indent=1)

    # ---- dataset watch ------------------------------------------------------------------------------------------------------
    ds_seen = {}
    for u in DATASET_USERS:
        for p in range(1, 6):
            try:
                rows = api.dataset_list(user=u, page=p) or []
            except Exception as e:
                errors.append('dataset_list %s p%d: %s' % (u, p, e))
                break
            for d in rows:
                if d is None:
                    continue
                ds_seen[d.ref] = {'version': d.current_version_number, 'last_updated': tstr(d.last_updated),
                                  'bytes': d.total_bytes, 'is_private': d.is_private, 'title': d.title, 'via': 'datasets list --user ' + u}
            if len(rows) < 20:
                break
    for ref in BASELINE_DATASETS:
        s, j = web.dataset(ref)
        e = ds_seen.setdefault(ref, {'via': 'GetDatasetBasics (anonymous)'})
        if s == 200 and j:
            e.update({'version': j.get('datasetVersionNumber'), 'version_id': j.get('datasetVersionId'),
                      'last_updated': j.get('lastUpdateTime'), 'bytes': (j.get('data') or {}).get('totalSize'), 'title': j.get('title')})
        else:
            e.update({'http': s, 'state': 'not_public' if s == 404 else 'unknown'})
    ds_baseline = not st['datasets']           # the first dataset pass records, it does not alert
    for ref, e in ds_seen.items():
        old = st['datasets'].get(ref)
        if old is None:
            if not ds_baseline:
                alert('NEW_DATASET', ref, 'v%s %s bytes "%s"' % (e.get('version'), e.get('bytes'), e.get('title')))
        else:
            if e.get('version') is not None and old.get('version') is not None and e['version'] != old['version']:
                alert('DATASET_VERSION_BUMP', ref, 'v%s -> v%s (%s, %s bytes)' % (old['version'], e['version'], e.get('last_updated'), e.get('bytes')))
            elif e.get('bytes') and old.get('bytes') and e['bytes'] != old['bytes']:
                alert('DATASET_SIZE_CHANGE', ref, '%s -> %s bytes' % (old['bytes'], e['bytes']))
            if e.get('http') == 404 and old.get('version') is not None:
                alert('DATASET_GONE', ref, 'was v%s, now not public' % old['version'])
        st['datasets'][ref] = dict(e, checked=iso(), first_seen=(old or {}).get('first_seen', iso()))

    # ---- summary, state, log --------------------------------------------------------------------------------------------------
    ranked = sorted([r for r, k in K.items() if fnum(k.get('score')) is not None and not k.get('gone')],
                    key=lambda r: (-K[r]['score'], -(K[r].get('votes') or 0)))
    inv = 0
    sp = [(K[r]['list_pos']['score'], K[r]['score']) for r in K if (K[r].get('list_pos') or {}).get('score') and K[r].get('score') is not None]
    sp.sort()
    inv = sum(1 for (a, sa), (b, sb) in zip(sp, sp[1:]) if sb > sa + 1e-9)
    bands = {}
    for r in ranked:
        bands[('%.3f' % K[r]['score'])] = bands.get('%.3f' % K[r]['score'], 0) + 1
    flags = [r for r in K if K[r].get('flag') == 'RELEASE_CANDIDATE']
    new_flags = [a for a in alerts if a['kind'] == 'RELEASE_CANDIDATE']
    summary = {'utc': iso(t0), 'ist': ist(t0), 'seconds': round((now() - t0).total_seconds()), 'first_run': first_run,
               'listed': len(listed), 'known': len(K), 'read': n_read, 'new': sum(1 for w in todo.values() if w == 'new'),
               'pulled': n_pull, 'web_calls': web.calls, 'web_errors': web.errors, 'errors': len(errors),
               'max_public_score': K[ranked[0]]['score'] if ranked else None, 'bands_top': dict(sorted(bands.items(), reverse=True)[:4]),
               'score_sort_inversions': inv, 'release_candidates': flags, 'new_flags': len(new_flags),
               'alerts': len(alerts), 'browser_todo': len(todo_b)}
    st['alerts'].extend(alerts)
    st['runs'].append(summary)
    st['last_run'] = iso(t0)
    save_state(st)
    print('WATCH_SUMMARY', json.dumps(summary), flush=True)
    for e in errors[:20]:
        print('WATCH_ERROR', e, flush=True)
    if not A.no_log:
        write_log(st, summary, alerts, errors, ranked, K, first_run)
    return 10 if new_flags else 0


def write_log(st, summary, alerts, errors, ranked, K, first_run):
    new = not os.path.isfile(LOG_P)
    L = []
    if new:
        L += ['# WATCH_LOG: public-release watch for rsna-knee-abnormality-detection (watch.py; RESEARCH_960 lever 3)', '',
              'Read-only. One section per run, newest at the bottom. Scores = the public score the notebook page shows (Kaggle',
              'page model, read anonymously; display is truncated to 3 dp). "inputs" = every data source of the SCORED version,',
              'tested anonymously (public dataset = GetDatasetBasics 200). RELEASE_CANDIDATE = score >= .957 and all inputs public.',
              'DRIFT = a dataset has a newer version than the one the scored run pinned (a CLI copy would attach the newer one).', '']
    L += ['## Run %s (%s)%s' % (summary['utc'], summary['ist'], ' BASELINE' if first_run else ''), '',
          '- listed %d notebooks (score pages + recency pages + key authors), known %d, read %d (new %d), pulled -m %d, '
          'web calls %d (errors %d), %d s.' % (summary['listed'], summary['known'], summary['read'], summary['new'],
                                              summary['pulled'], summary['web_calls'], summary['web_errors'], summary['seconds']),
          '- max public score among public notebooks: **%s**; score bands (count of notebooks): %s; list sort inversions vs page '
          'scores: %d.' % (summary['max_public_score'], summary['bands_top'], summary['score_sort_inversions']),
          '- RELEASE_CANDIDATE (score >= %.3f and all inputs public): **%s**' % (A.flag_min, ', '.join(summary['release_candidates']) or 'none'),
          '- browser fallback queue (page model unreadable): %d (browser_todo.json)' % summary['browser_todo'], '']
    if alerts:
        L += ['Alerts this run:', '']
        L += ['- %s %s `%s` %s' % (a['level'], a['kind'], a['ref'], a['msg']) for a in alerts[:60]]
        if len(alerts) > 60:
            L.append('- ... %d more in watch_state.json' % (len(alerts) - 60))
        L.append('')
    alias = {'heliosli/rsna-knee-compact-r7cnxfsss': 'R7', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004': 'OF',
             'nartaa/rsna-knee-publication-swa-weights-20261007': 'NS', 'heliosli/rsna-knee-compact-r6cnxfsss': 'R6',
             'heliosli/rsna-knee-compact-v1': 'CV1', 'goodpjw2008/rsna-knee-2-5d-convnext-reader': 'GP'}
    def short(dsl):
        return '+'.join(sorted(alias.get(d, d) for d in dsl)) or '-'
    L += ['Top 15 public notebooks by page score (inputs: R7 = heliosli/rsna-knee-compact-r7cnxfsss, OF = heliosli/'
          'rsna-orthofoundation-fold0-epoch7-20261004, NS = nartaa/rsna-knee-publication-swa-weights-20261007, R6 = heliosli/'
          'rsna-knee-compact-r6cnxfsss, CV1 = heliosli/rsna-knee-compact-v1, GP = goodpjw2008/rsna-knee-2-5d-convnext-reader):', '',
          '| # | notebook | score | scored v / current v | inputs | drift | last run (UTC) |', '|---|---|---|---|---|---|---|']
    for i, r in enumerate(ranked[:15]):
        k = K[r]
        kv = k.get('kvm') or {}
        inp = k.get('inputs') or {}
        ds = [x.split(':', 1)[1] for x, v in (inp.get('detail') or {}).items() if x.startswith('dataset:')]
        L.append('| %d | `%s` | %.3f | v%s / v%s | %s %s | %s | %s |' % (
            i + 1, r, k['score'], kv.get('best_version'), kv.get('cur_version'),
            'public' if inp.get('all_public') else 'NOT all public:', short(ds),
            ', '.join(alias.get(d, d) for d in (inp.get('drift') or {})) or '-', (k.get('last_run_time') or '')[:16]))
    # free reads: unscored notebooks that ran in the last 72 h on all-public inputs including the R7 bundle
    pend = []
    for r, k in K.items():
        run = ((k.get('kvm') or {}).get('run') or {})
        age = age_s(run.get('date'))
        ds = [x.split(':', 1)[1] for x in ((k.get('inputs') or {}).get('detail') or {}) if x.startswith('dataset:')]
        if k.get('score') is None and age is not None and age < 72 * 3600 and (k.get('inputs') or {}).get('all_public')                 and 'heliosli/rsna-knee-compact-r7cnxfsss' in ds and not k.get('gone'):
            pend.append((run.get('date') or '', r, short(ds), (k.get('kvm') or {}).get('cur_version')))
    if pend:
        L += ['', 'Pending free reads (no public score yet; ran in the last 72 h; all inputs public, R7 bundle among them; a score',
              'appearing raises SCORE_APPEARED):', '']
        L += ['- `%s` v%s, ran %s UTC, inputs %s' % (r, v, d[:16], dsx) for d, r, dsx, v in sorted(pend, reverse=True)[:20]]
    watched = [(r, d) for r, d in sorted(st['datasets'].items()) if any(w in (r + ' ' + str(d.get('title', ''))).lower()
                                                                       for w in ('rsna', 'knee', 'raptor'))]
    if watched:
        L += ['', 'Watched datasets (heliosli + nartaa lists, RESEARCH_960 baseline set; knee-related only, all in watch_state.json):', '',
              '| dataset | version | last updated | bytes | via |', '|---|---|---|---|---|']
        L += ['| `%s` | v%s | %s | %s | %s |' % (r, d.get('version'), str(d.get('last_updated'))[:16], d.get('bytes'), d.get('via'))
              for r, d in watched]
    if errors:
        L += ['', 'Errors (%d): %s' % (len(errors), '; '.join(errors[:8]))]
    L.append('')
    with open(LOG_P, 'a', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    while True:
        try:
            rc = run_once()
        except Exception:
            traceback.print_exc()
            rc = 2
        if A.loop_hours <= 0:
            sys.exit(rc)
        print('WATCH next run in %.1f h (rc %d)' % (A.loop_hours, rc), flush=True)
        time.sleep(A.loop_hours * 3600)
