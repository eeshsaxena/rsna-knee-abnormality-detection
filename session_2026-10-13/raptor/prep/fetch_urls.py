# fetch_urls.py (laptop): list the output files of the 3 prep kernels with their signed download URLs (no download, no token
# leaves the laptop: the URLs are short-lived signed links) -> urls.tsv (name \t url \t size). The box curls them.
import sys, os
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = KaggleApi(); api.authenticate()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urls.tsv')
rows = []
for k in range(3):
    token = None
    while True:
        with api.build_kaggle_client() as kg:
            r = ApiListKernelSessionOutputRequest(); r.user_name = 'eeshsaxena'; r.kernel_slug = f'rsna-raptor-d96-prep-s{k}'
            r.page_size = 200
            if token: r.page_token = token
            resp = kg.kernels.kernels_api_client.list_kernel_session_output(r)
        for f in resp.files or []:
            rows.append((f.file_name, f.url, getattr(f, 'size', '')))
        token = resp.next_page_token
        if not token: break
with open(out, 'w', newline='\n') as fh:
    for n, u, s in rows:
        fh.write(f'{n}\t{u}\t{s}\n')
print(len(rows), 'files ->', out)
for n, _, s in rows:
    if not n.endswith('.tar'): print(' ', n, s)
print(' tars:', sum(n.endswith('.tar') for n, _, _ in rows))
