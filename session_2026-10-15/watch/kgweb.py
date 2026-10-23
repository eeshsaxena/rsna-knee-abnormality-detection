# kgweb.py (2026-10-14, Claude subagent): READ-ONLY helpers shared by watch.py, swap_builder.py and swap_check.py.
# Two sources:
#   1. the official kaggle python API (same package and version as the `kaggle` CLI, 2.2.4): kernels list, kernel pull
#      (= `kaggle kernels pull REF -m`), datasets list (= `kaggle datasets list --user U`). Authenticated as Eesh.
#   2. Kaggle's own web endpoints, called ANONYMOUSLY (no cookies of Eesh, the same way research960/lbprobe/fetch_api_lb.py
#      reads the leaderboard): GetKernelViewModel (public score, versions, pinned input versions), ListKernelVersions
#      (per-version pinned inputs, docker digest, commit runtime), GetDatasetBasics (200 = public dataset + its current
#      version id; 404 = private, deleted or never existed). Anonymous = "what anyone can see" = the public-input test.
# Nothing here writes to Kaggle. No em dashes.
import re, json, time
import requests

BASE = 'https://www.kaggle.com'
COMP = 'rsna-knee-abnormality-detection'
COMP_ID = 154281


class Web:
    """Anonymous session on kaggle.com (XSRF cookie from a plain page GET, as fetch_api_lb.py does)."""

    def __init__(self, delay=0.25, timeout=60):
        self.delay, self.timeout = delay, timeout
        self.s, self.h = None, {}
        self.calls, self.errors = 0, 0

    def _session(self):
        if self.s is None:
            s = requests.Session()
            s.get(BASE + '/competitions', timeout=self.timeout)
            self.s, self.h = s, {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
        return self.s

    def post(self, method, body, tries=4):
        """Returns (http_status, json_or_None). 4xx other than 429 return at once (404 = not visible anonymously)."""
        last = None
        for i in range(tries):
            try:
                s = self._session()
                time.sleep(self.delay)
                r = s.post(BASE + '/api/i/' + method, json=body, headers=self.h, timeout=self.timeout)
                self.calls += 1
                last = r.status_code
                if r.status_code == 200:
                    return 200, r.json()
                if r.status_code in (400, 401, 403, 404):
                    if r.status_code in (401, 403) and i == 0:
                        self.s = None            # stale XSRF cookie: one fresh session, then give up
                        continue
                    return r.status_code, None
            except requests.RequestException as e:
                last = 'EXC ' + type(e).__name__
            time.sleep(2.0 * (i + 1))
        self.errors += 1
        return last, None

    def kvm(self, ref):
        owner, slug = ref.split('/')[:2]
        return self.post('kernels.LegacyKernelsService/GetKernelViewModel', {'authorUserName': owner, 'kernelSlug': slug})

    def versions(self, kernel_id):
        return self.post('kernels.KernelsService/ListKernelVersions', {'kernelId': int(kernel_id)})

    def dataset(self, ref):
        owner, slug = ref.split('/')[:2]
        return self.post('datasets.DatasetDetailService/GetDatasetBasics', {'ownerSlug': owner, 'datasetSlug': slug})


def classify_source(d, url_by_id=None):
    """One run data source -> (kind, ref). kind: competition | dataset | kernel | model | other. The type decides the kind;
    the owner comes from the mountSlug ('datasets/OWNER/SLUG') or, when the mount is a bare slug (seen on some notebooks),
    from the page's renderable source URL with the same sourceId; still unknown -> ref '?/SLUG' (resolved by the caller)."""
    t, m, sid = d.get('sourceType') or '', d.get('mountSlug') or '', d.get('sourceId')
    url = (url_by_id or {}).get(sid) or ''
    if t in ('', 'DATA_SOURCE_TYPE_COMPETITION') and (m.startswith('competitions/') or sid == COMP_ID or url.startswith('/competitions/')):
        return 'competition', (m.split('/', 1)[1] if m.startswith('competitions/') else (url.split('/')[2] if url else COMP))
    if 'DATASET' in t:
        if m.startswith('datasets/'):
            return 'dataset', '/'.join(m.split('/')[1:3])
        if url.startswith('/datasets/'):
            return 'dataset', '/'.join(url.split('/')[2:4])
        return 'dataset', '?/' + m.split('/')[-1]
    if 'KERNEL' in t:
        if url.startswith('/code/'):
            return 'kernel', '/'.join(url.split('/')[2:4])
        if '/' in m and not m.startswith(('notebooks/', 'code/', 'kernels/')):
            return 'kernel', '/'.join(m.split('/')[:2])
        if m.startswith(('notebooks/', 'code/', 'kernels/')):
            return 'kernel', '/'.join(m.split('/')[1:3])
        return 'kernel', '?/' + m
    if 'MODEL' in t:
        return 'model', (m.split('/', 1)[1] if m.startswith('models/') else (url[len('/models/'):] if url.startswith('/models/') else m))
    return 'other', m or url or str(sid)


def parse_isolator(xml):
    out = {}
    for k in ('docker_image_digest', 'docker_image_name', 'run_time_seconds', 'succeeded', 'timeout_exceeded', 'exit_code'):
        m = re.search('<%s>(.*?)</%s>' % (k, k), xml or '')
        if m:
            out[k] = m.group(1).strip()
    return out


def sources_of(ds_list, url_by_id=None):
    out = []
    for d in ds_list or []:
        kind, ref = classify_source(d, url_by_id)
        out.append({'kind': kind, 'ref': ref, 'version_id': d.get('sourceId'), 'type': d.get('sourceType'),
                    'dataset_id': d.get('datasetId')})
    return out


def summarize_kvm(j):
    """The fields we keep from GetKernelViewModel."""
    k, run = j.get('kernel') or {}, j.get('kernelRun') or {}
    bss, sub = j.get('bestSubmissionScore') or {}, j.get('submission') or {}
    iso = parse_isolator(run.get('isolatorResults'))
    return {
        'kernel_id': k.get('id'), 'title': k.get('title'), 'best_public': k.get('bestPublicScore'),
        'best_version': bss.get('kernelVersionNumber'), 'best_version_url': bss.get('kernelSessionUrl'),
        'cur_version': j.get('currentVersionNumber'), 'n_versions': j.get('totalVersionCount'),
        'cur_version_score': sub.get('scoreFormatted'), 'has_linked_submission': k.get('hasLinkedSubmission'),
        'updated': k.get('updatedTime'), 'medal': k.get('medal'), 'votes': k.get('upvoteCount'), 'forks': k.get('forkCount'),
        'docker_pinning': k.get('dockerImagePinningType'),
        'run': {'id': run.get('id'), 'version': run.get('kernelVersionNumber'), 'status': run.get('status'),
                'date': run.get('dateCreated'), 'accelerator': run.get('acceleratorType'), 'gpu': run.get('isGpuEnabled'),
                'docker_version_id': run.get('dockerImageVersionId'), 'run_time_seconds': iso.get('run_time_seconds'),
                'docker_digest': iso.get('docker_image_digest'), 'succeeded': iso.get('succeeded')},
        'sources': sources_of(j.get('dataSources'), renderable_urls(j)),
    }


def renderable_urls(j):
    out = {}
    for r in (j or {}).get('renderableDataSources') or []:
        sid = (r.get('reference') or {}).get('sourceId')
        if sid is not None and r.get('dataSourceUrl'):
            out[sid] = r['dataSourceUrl']
    return out


def version_table(j):
    """ListKernelVersions -> {version_number: {...}} with pinned sources per version."""
    out = {}
    for it in (j or {}).get('items') or []:
        v, run = it.get('version') or {}, it.get('run') or {}
        n = v.get('versionNumber')
        if n is None:
            continue
        iso = parse_isolator(run.get('isolatorResults'))
        out[int(n)] = {'run_id': run.get('id'), 'status': run.get('status'), 'date': run.get('dateCreated'),
                       'version_type': v.get('versionType'), 'run_time_seconds': iso.get('run_time_seconds'),
                       'docker_digest': iso.get('docker_image_digest'), 'succeeded': iso.get('succeeded'),
                       'accelerator': ((run.get('commit') or {}).get('settings') or {}).get('accelerator'),
                       'sources': sources_of(((run.get('commit') or {}).get('settings') or {}).get('dataSources'))}
    return out


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def get_kernel(api, ref):
    """Exactly what `kaggle kernels pull REF -m` fetches (ApiGetKernelRequest), returned in memory instead of written in
    Windows text mode (the CLI's open(path, 'w') would turn every '\\n' of a .py source into '\\r\\n').
    Returns (source_text, metadata_dict_as_the_CLI_writes_it, file_name_the_CLI_would_use)."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug, version = api.parse_kernel_string(ref)
    with api.build_kaggle_client() as kaggle:
        req = ApiGetKernelRequest()
        req.user_name = owner
        req.kernel_slug = slug
        if version:
            # the CLI's 'SLUG/N' form returns 403 on this API (tested 10-14); version_label 'vN' returns version N
            # (tested: lavinwins/rsna-gold-medal v1 514,559 bytes vs v2/v3/current 152,191; varshitha v1 == current)
            req.version_label = 'v%s' % version
        resp = kaggle.kernels.kernels_api_client.get_kernel(req)
    blob, md = resp.blob, resp.metadata
    ext = {('script', 'python'): '.py', ('notebook', 'python'): '.ipynb', ('script', 'r'): '.R', ('notebook', 'r'): '.irnb'}
    file_name = blob.slug + ext.get((blob.kernel_type.lower(), blob.language.lower()), '.unknown')
    data = {}   # same keys, same order as kaggle_api_extended.kernels_pull(metadata=True)
    data['id'] = md.ref
    data['id_no'] = md.id
    data['title'] = md.title
    data['code_file'] = file_name
    data['language'] = md.language
    data['kernel_type'] = md.kernel_type
    data['is_private'] = md.is_private
    data['enable_gpu'] = md.enable_gpu
    data['enable_tpu'] = md.enable_tpu
    data['enable_internet'] = md.enable_internet
    data['keywords'] = md.category_ids
    data['dataset_sources'] = md.dataset_data_sources
    data['kernel_sources'] = md.kernel_data_sources
    data['competition_sources'] = md.competition_data_sources
    data['model_sources'] = md.model_data_sources
    data['docker_image'] = md.docker_image
    data['machine_shape'] = md.machine_shape
    return blob.source, data, file_name


def plain(o):
    """JSON-safe copy (the SDK returns list-like wrappers)."""
    return json.loads(json.dumps(o, default=lambda x: list(x) if hasattr(x, '__iter__') else str(x)))
