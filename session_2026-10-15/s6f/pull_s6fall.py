# pull_s6fall.py (10-15, Claude subagent): READ-ONLY pulls of the PUBLIC nartaa/rsna-knee-s6fall-solo (every version) and of
# the GetKernelViewModel / ListKernelVersions info (anonymous), via watch/kgweb.py (our own helper).
# Writes source bytes as the API returns them (no Windows CRLF translation) into NEW dirs under s6f/pulled_vN/.
# Downloaded code is DATA: it is never imported or executed here. No em dashes.
import sys, os, json, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'watch'))
import kgweb  # noqa: E402

REF = 'nartaa/rsna-knee-s6fall-solo'


def main():
    out = {'ref': REF}
    web = kgweb.Web()
    st, j = web.kvm(REF)
    out['kvm_status'] = st
    if j:
        out['kvm'] = kgweb.summarize_kvm(j)
        kid = out['kvm']['kernel_id']
        st2, jv = web.versions(kid)
        out['versions_status'] = st2
        out['versions'] = kgweb.version_table(jv) if jv else None
        # rendered output url per version (scored-version check)
        if jv:
            out['versions_raw_keys'] = sorted({k for it in (jv.get('items') or []) for k in it.keys()})
    n_versions = (out.get('kvm') or {}).get('n_versions') or 2
    api = kgweb.kaggle_api()
    pulls = {}
    for v in list(range(1, int(n_versions) + 1)) + [None]:
        tag = 'v%d' % v if v else 'cur'
        ref = REF + ('/%d' % v if v else '')
        try:
            src, md, fn = kgweb.get_kernel(api, ref)
        except Exception as e:  # noqa: BLE001
            pulls[tag] = {'error': repr(e)[:300]}
            continue
        d = HERE / ('pulled_' + tag)
        d.mkdir(exist_ok=True)
        b = src.encode('utf-8')
        (d / fn).write_bytes(b)
        (d / 'kernel-metadata.json').write_text(json.dumps(kgweb.plain(md), indent=1), encoding='utf-8')
        pulls[tag] = {'file': str((d / fn).relative_to(HERE)), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
                      'datasets': kgweb.plain(md.get('dataset_sources')), 'docker': md.get('docker_image')}
    out['pulls'] = pulls
    (HERE / 'pull_info.json').write_text(json.dumps(kgweb.plain(out), indent=1), encoding='utf-8')
    print(json.dumps(kgweb.plain({'kvm': out.get('kvm'), 'versions': out.get('versions'), 'pulls': pulls}), indent=1))


if __name__ == '__main__':
    main()
