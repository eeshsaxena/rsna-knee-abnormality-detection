cd /c/Users/eeshs/Downloads/rsna-knee
SLUG=slugify_guard
# DUP-GUARD: skip if a 'soup' submission already exists
already=$(python - << 'PY' 2>/dev/null
import sys,types,re,unicodedata
_m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode(); return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
_m.slugify=slugify; sys.modules['slugify']=_m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
print(sum(1 for s in api.competition_submissions('rsna-knee-abnormality-detection')[:8] if 'soup' in str(getattr(s,'description','')).lower()))
PY
)
if [ "$already" != "0" ] && [ -n "$already" ]; then echo "GUARD: soup already submitted ($already), skipping"; exit 0; fi
sleep 45
for i in $(seq 1 60); do s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-soup 2>/dev/null|tail -1); case "$s" in *COMPLETE*) echo "[$i] COMPLETE"; break;; *ERROR*|*CANCEL*) echo "[$i] $s"; break;; *) sleep 20;; esac; done
rm -rf _soupout; mkdir -p _soupout; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-soup _soupout 2>&1|tail -1
rows=$(tail -n +2 _soupout/submission.csv 2>/dev/null|wc -l); echo "rows=$rows"
grep -oE '"data":"[^"]*"' _soupout/*.log 2>/dev/null|sed 's/"data":"//;s/"$//'|grep -iE "loaded|heads|gold58|error" | head -3
if [ "$rows" -ge 3 ]; then python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-soup 1 "SOUP solo (fold1+fold2 avg) K=40 gold58 0.8822" 2>&1|tail -1; else echo "NOT valid"; fi
