# get_comp_url.py: print the final (redirected) download URL of the competition zip to STDOUT ONLY (piped, never shown).
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    req = ApiDownloadDataFilesRequest(); req.competition_name = 'rsna-knee-abnormality-detection'
    resp = kaggle.competitions.competition_api_client.download_data_files(req)
    url = resp.url; size = resp.headers.get('Content-Length'); resp.close()
sys.stderr.write('URL_HOST %s LEN %d SIZE %s\n' % (url.split('/')[2], len(url), size))
sys.stdout.write(url)
