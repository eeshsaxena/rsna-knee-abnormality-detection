import torch, pandas as pd, os
from transformers import AutoModelForCausalLM, AutoTokenizer
KEYS=['ACL','MCL','MedialMeniscus','LateralMeniscus','MedialOA','LateralOA','PFOA',
      'Effusion','Synovitis','Bakers','Contusion','Fracture']
SYS=('You are a musculoskeletal radiologist labeling knee MRI reports. Reports may be in any '
     'language (Spanish, English, Dutch, German, etc.). Judge each finding from the report '
     'content, handling negations (intact/normal/no/sin/geen mean ABSENT).')
DEFS=('For each finding output 1 if the report indicates it is PRESENT/abnormal, else 0. '
      'Respond with ONLY a JSON object with exactly these keys and 0/1 values: '+', '.join(KEYS)+'.')
tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.float16).to('cuda').eval()
df=pd.read_csv(os.path.expanduser('~/rsna_train/train.csv'))
gold=df[df['ACL'].notna()]
for r in gold['Report'].values[:3]:
    msgs=[{'role':'system','content':SYS},{'role':'user','content':DEFS+'\n\nREPORT:\n'+str(r)[:3500]}]
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp=tok(text, return_tensors='pt').to('cuda')
    out=model.generate(**inp, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    gen=tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print('RAWLEN', len(gen), 'RAW:', repr(gen[:350]), flush=True)
    print('---', flush=True)
