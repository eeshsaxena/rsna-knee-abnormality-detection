#!/usr/bin/env python3
"""Qwen2.5-7B report -> 12-label extractor on the box V100. Bigger model than the 3B (0.73) to
lift label quality (the confirmed ceiling). Validates gold-58 FIRST; only runs all 4407 if it
beats the 3B. Saves pseudo_labels_qwen7b.csv (gold kept as-is for the 58)."""
import os, glob, re, json, time, argparse, numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap=argparse.ArgumentParser()
ap.add_argument('--train', default=os.path.expanduser('~/rsna_train/train.csv'))
ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
ap.add_argument('--bs', type=int, default=1); ap.add_argument('--all', type=int, default=1)
ap.add_argument('--load8bit', type=int, default=1)
a=ap.parse_args()

KEYS=['ACL','MCL','MedialMeniscus','LateralMeniscus','MedialOA','LateralOA','PFOA',
      'Effusion','Synovitis','Bakers','Contusion','Fracture']
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
        'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
K2L=dict(zip(KEYS,LABELS))
SYS=('You are a musculoskeletal radiologist labeling knee MRI reports. Reports may be in any '
     'language (Spanish, English, Dutch, German, etc.). Judge each finding from the report '
     'content, handling negations (intact/normal/no/sin/geen mean ABSENT).')
DEFS=('For each finding output 1 if the report indicates it is PRESENT/abnormal, else 0 (absent, '
      'normal, or not mentioned). Findings:\n'
      '- ACL: anterior cruciate ligament tear/rupture/sprain\n- MCL: medial collateral ligament tear/injury\n'
      '- MedialMeniscus: medial meniscus tear\n- LateralMeniscus: lateral meniscus tear\n'
      '- MedialOA: medial compartment osteoarthritis / cartilage loss / osteophytes\n'
      '- LateralOA: lateral compartment osteoarthritis / cartilage loss / osteophytes\n'
      '- PFOA: patellofemoral or trochlear osteoarthritis / chondral loss\n- Effusion: joint effusion / joint fluid\n'
      '- Synovitis: synovitis / synovial thickening\n- Bakers: Baker (popliteal) cyst\n'
      '- Contusion: bone contusion / bruise / bone-marrow edema\n- Fracture: fracture\n'
      'Respond with ONLY a compact single-line JSON object (no newlines, no spaces) with exactly '
      'these keys and 0/1 values: '+','.join(KEYS)+'.')

def parse(txt):
    m=re.search(r'\{[^{}]*\}', txt, re.DOTALL); d={}
    if m:
        try: d=json.loads(m.group(0))
        except Exception:
            for k in KEYS:
                mm=re.search(rf'"{k}"\s*:\s*(\d)', m.group(0))
                if mm: d[k]=int(mm.group(1))
    return {L:int(d.get(k,0) or 0) for k,L in K2L.items()}

print('loading', a.model, 'fp16 full-GPU', flush=True)
tok=AutoTokenizer.from_pretrained(a.model); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
# fully on GPU (no CPU offload -> fast). 7B fp16 ~15GB fits the free 16GB V100. bs=1, short prompts.
model=AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to('cuda').eval()
print('loaded | GPU GB', round(torch.cuda.memory_allocated()/1e9,1), '| bs', a.bs, flush=True)

@torch.no_grad()
def extract_batch(reports):
    texts=[tok.apply_chat_template([{'role':'system','content':SYS},
            {'role':'user','content':DEFS+'\n\nREPORT:\n'+str(r)[:3500]}],
            tokenize=False, add_generation_prompt=True) for r in reports]
    inp=tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to('cuda')
    out=model.generate(**inp, max_new_tokens=110, do_sample=False, pad_token_id=tok.eos_token_id)
    n=inp.input_ids.shape[1]
    return [parse(tok.decode(out[i][n:], skip_special_tokens=True)) for i in range(len(reports))]

def run(reports, tag):
    rows=[]; t0=time.time()
    for i in range(0, len(reports), a.bs):
        rows+=extract_batch(list(reports[i:i+a.bs]))
        if i % 50 == 0: print(f'  {tag} {i}/{len(reports)} {time.time()-t0:.0f}s', flush=True)
    return pd.DataFrame(rows)[LABELS]

df=pd.read_csv(a.train)
gold=df[df['ACL'].notna()].reset_index(drop=True)
pred=run(gold['Report'].values, 'gold')
aucs=[]
for lab in LABELS:
    y=(gold[lab].values==1.0).astype(int); pp=pred[lab].values.astype(int)
    tp=((pp==1)&(y==1)).sum(); fp=((pp==1)&(y==0)).sum(); fn=((pp==0)&(y==1)).sum(); tn=((pp==0)&(y==0)).sum()
    tpr=tp/(tp+fn) if tp+fn else 0.0; tnr=tn/(tn+fp) if tn+fp else 0.0; aucs.append((tpr+tnr)/2)
    print(f'{lab:18s} bal-acc {(tpr+tnr)/2:.3f}', flush=True)
macro=float(np.mean(aucs))
print(f'\n=== Qwen-7B gold-58 MACRO {macro:.4f}  (3B was 0.7313) ===', flush=True)

if a.all:
    print('\nrunning ALL 4407...', flush=True)
    ap_=run(df['Report'].values, 'all')
    gm=df['ACL'].notna().values
    out=df[['StudyInstanceUID']].copy()
    for lab in LABELS: out[lab]=np.where(gm, df[lab].values, ap_[lab].values.astype(float))
    dst=os.path.expanduser('~/rsna_train/pseudo_labels_qwen7b.csv'); out.to_csv(dst, index=False)
    print('WROTE', dst, out.shape, flush=True)
