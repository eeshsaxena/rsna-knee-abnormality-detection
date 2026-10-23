#!/usr/bin/env python3
"""Generate RSNA_Colab.ipynb: a self-contained, maximally-automated Colab notebook that
embeds colab_train.py, auths from Colab Secrets, auto-downloads data, trains end-to-end with
auto-resume from Google Drive, and pushes the trained model back to Kaggle. Run: python gen_colab_notebook.py"""
import json, os
HERE=os.path.dirname(os.path.abspath(__file__))
trainer=open(os.path.join(HERE,'colab_train.py'),encoding='utf-8').read()

def md(*t): return {"cell_type":"markdown","metadata":{},"source":[l if l.endswith('\n') else l+'\n' for l in t]}
def code(*t): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                      "source":[l if l.endswith('\n') else l+'\n' for l in t]}

cells=[
 md("# RSNA Knee — end-to-end training on Colab (free T4)",
    "",
    "**One-time setup (2 minutes):**",
    "1. **Runtime → Change runtime type → T4 GPU.**",
    "2. Left sidebar **🔑 Secrets** → add two secrets (notebook access ON):",
    "   `KAGGLE_USERNAME` = your kaggle username, `KAGGLE_KEY` = your kaggle key",
    "   (both are in your kaggle.json). This persists across sessions.",
    "3. **Runtime → Run all.**",
    "",
    "That's it. If Colab disconnects, just **Run all** again — it auto-resumes from the",
    "checkpoint on your Google Drive. No re-configuration needed."),

 md("## 1. Auth + Drive + deps (automatic)"),
 code("import os",
      "from google.colab import userdata, drive",
      "os.environ['KAGGLE_USERNAME']=userdata.get('KAGGLE_USERNAME')",
      "os.environ['KAGGLE_KEY']=userdata.get('KAGGLE_KEY')",
      "drive.mount('/content/drive')",
      "CKPT='/content/drive/MyDrive/rsna_ckpt'; os.makedirs(CKPT, exist_ok=True)",
      "print('ckpt dir:', CKPT)",
      "!pip -q install timm kaggle 2>/dev/null",
      "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — set Runtime>T4 GPU')"),

 md("## 2. Download data (automatic, idempotent)",
    "Shards come from the preprocess kernel output; the small CSVs from a Kaggle dataset.",
    "Skips anything already present."),
 code("import os, glob",
      "os.makedirs('/content/data', exist_ok=True); os.makedirs('/content/meta', exist_ok=True)",
      "if not glob.glob('/content/data/**/shard_*.npz', recursive=True):",
      "    print('downloading image shards (~10GB, a few min)...')",
      "    !kaggle kernels output eeshsaxena/rsna-knee-preprocess -p /content/data",
      "if not os.path.exists('/content/meta/train.csv'):",
      "    !kaggle datasets download eeshsaxena/rsna-knee-train-meta -p /content/meta --unzip",
      "print('shards:', len(glob.glob('/content/data/**/shard_*.npz', recursive=True)),",
      "      '| meta:', os.listdir('/content/meta'))"),

 md("## 3. Trainer (embedded — no external files)"),
 code("%%writefile /content/colab_train.py\n"+trainer),

 md("## 4. Train (auto-resumes from Drive on every run)",
    "Edit `--backbone` / `--res` / `--epochs` to taste. Default: EfficientNet-B3 @ 224px, fold 0.",
    "Re-run this cell after any disconnect; it continues from the last saved epoch."),
 code("!python /content/colab_train.py \\",
      "  --shards /content/data --labels /content/meta/pseudo_labels_v1.csv \\",
      "  --train /content/meta/train.csv --lang /content/meta/_reports_lang.csv \\",
      "  --ckpt {CKPT} --backbone tf_efficientnet_b3 --res 224 --K 12 --epochs 20 --fold 0"),

 md("## 5. Push best model back to Kaggle (for the inference kernel)",
    "Uploads the best checkpoint as a private Kaggle dataset so the submission kernel can load it."),
 code("import glob, os, json, shutil",
      "best=sorted(glob.glob(CKPT+'/*_BEST.pt'), key=os.path.getmtime)",
      "assert best, 'no BEST checkpoint yet — run training first'",
      "up='/content/model_up'; os.makedirs(up, exist_ok=True)",
      "for b in best: shutil.copy(b, up)",
      "json.dump({'title':'rsna-knee-colab-model','id':'eeshsaxena/rsna-knee-colab-model',",
      "           'licenses':[{'name':'CC0-1.0'}]}, open(up+'/dataset-metadata.json','w'))",
      "from kaggle.api.kaggle_api_extended import KaggleApi",
      "api=KaggleApi(); api.authenticate()",
      "try: api.dataset_create_new(up, public=False, dir_mode='zip'); print('created dataset')",
      "except Exception: api.dataset_create_version(up, 'update', dir_mode='zip'); print('new version')"),
]

nb={"cells":cells,
    "metadata":{"accelerator":"GPU","colab":{"provenance":[]},
                "kernelspec":{"display_name":"Python 3","name":"python3"},
                "language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":0}
out=os.path.join(HERE,'RSNA_Colab.ipynb')
json.dump(nb, open(out,'w',encoding='utf-8'), indent=1)
print('wrote', out, '| cells', len(cells))
