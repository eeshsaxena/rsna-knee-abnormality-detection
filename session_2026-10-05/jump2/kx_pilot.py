#!/usr/bin/env python3
"""kx_pilot.py (jump2 label track, written 2026-10-05; NOT RUN on a real LLM yet; plumbing tested with --tiny_test on CPU).

Multilingual per-label knee-report extractor, scored by next-token LOG-PROBS (no free generation), so every
study x label gets a SOFT score (binary reads lose ~.07 gold58 AUC: steven_v4 .8927 soft vs .8187 binarized, MEASURED).
Per label two probes on a shared KV-cached prefix (system + definitions + report):
  Q1 state : A present | B absent/normal | C uncertain/possible | D not mentioned      -> pA pB pC pD
  Q2 grade : 1 low (mild/small/grade I-II/sprain/partial low-grade) | 2 moderate | 3 high  -> g1 g2 g3 (given present)
PRE-REGISTERED score (fixed before any gold run; do not tune on gold58):
  s = 0*pB + .15*pD + .45*pC + pA*(.60*g1 + .85*g2 + 1.00*g3)
Optional audit (--spans 1): short greedy generation of a verbatim evidence quote for cells with pA+pC>.5,
checked as a substring of the report (hallucination rate).

Usage on the box (GPU job: needs Eesh's OK and a free GPU; ~1-3 min for gold74 after load, MODEL):
  HF_HUB_OFFLINE=1 ~/miniconda3/envs/rsna/bin/python kx_pilot.py --model Qwen/Qwen2.5-7B-Instruct --device cuda \
      --ids gold74 --out ~/jump2/kx/gold74_qwen25_7b.csv            [--load4bit 1 to share the GPU with the fleet]
  then the gate is printed automatically for --ids gold74 (needs ~/rsna_train + ~/jump2/labels_in).
Plumbing test (CPU, random 2-layer model, real tokenizer, numbers meaningless):
  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python kx_pilot.py --tiny_test 1 --ids gold74 --limit 3 --out ~/jump2/kx/tiny.csv
Writes only to the --out path (must be under ~/jump2/).
"""
import os, re, sys, copy, time, json, argparse, unicodedata
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
ap.add_argument('--device', default='cpu')
ap.add_argument('--load4bit', type=int, default=0)
ap.add_argument('--ids', default='gold74', help='gold74 | all | path to csv with StudyInstanceUID')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--spans', type=int, default=0)
ap.add_argument('--tiny_test', type=int, default=0)
ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()
H = os.path.expanduser('~')
assert os.path.abspath(os.path.expanduser(a.out)).startswith(f'{H}/jump2/'), 'write only under ~/jump2/'
torch.set_num_threads(a.threads)
ID = 'StudyInstanceUID'; RT = f'{H}/rsna_train'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]

SYSTEM = ("You are an experienced musculoskeletal radiologist. You read knee MRI reports written in any language "
          "(English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French) and answer questions about "
          "ONE finding at a time, using only what the report says. Handle negation carefully (no / without / intact / "
          "normal / sin / yok / bez / nema / δεν / χωρίς / няма / kein / ohne / geen / pas de mean ABSENT), and keep "
          "medial (internal, interno, iç, medijalni, έσω, медиален, mediaal, interne) and lateral (external, externo, "
          "dış, lateralni, έξω, латерален, lateraal, externe) apart.")
DEFS = {
 "ACL": "anterior cruciate ligament (native or graft) tear, rupture, avulsion or sprain (complete or partial)",
 "MCL": "medial collateral ligament sprain or tear (any grade). Fluid or bursitis next to an intact MCL alone is not a tear",
 "Medial Meniscus": "tear of the MEDIAL (internal) meniscus: any tear type, root tear, re-tear after meniscectomy. "
                    "Intrasubstance degeneration or grade 1-2 signal without a tear is not a tear",
 "Lateral Meniscus": "tear of the LATERAL (external) meniscus: any tear type, root tear, re-tear after meniscectomy. "
                     "Intrasubstance degeneration or grade 1-2 signal without a tear is not a tear",
 "Medial OA": "osteoarthritis of the MEDIAL femorotibial compartment (medial femoral condyle / medial tibial plateau): "
              "cartilage thinning, chondral defects or chondromalacia, osteophytes, joint-space narrowing, subchondral cysts",
 "Lateral OA": "osteoarthritis of the LATERAL femorotibial compartment (lateral femoral condyle / lateral tibial plateau): "
               "cartilage thinning, chondral defects or chondromalacia, osteophytes, joint-space narrowing, subchondral cysts",
 "PF OA": "patellofemoral osteoarthritis or chondropathy: patellar cartilage (any facet, medial or lateral facet "
          "included) or trochlear cartilage damage, patellofemoral osteophytes",
 "Effusion": "knee joint effusion: increased intra-articular fluid, hydrops or hemarthrosis (including suprapatellar recess fluid)",
 "Synovitis": "synovitis: synovial thickening, synovial hypertrophy or inflamed synovium",
 "Baker's": "Baker (popliteal) cyst, including a ruptured or leaking Baker cyst; not a parameniscal cyst or ganglion",
 "Contusion": "traumatic bone contusion / bone bruise / post-traumatic bone-marrow edema. Degenerative subchondral "
              "edema next to cartilage loss is not a contusion",
 "Fracture": "any fracture: impaction, osteochondral, avulsion (e.g. Segond, tibial spine), insufficiency or stress fracture",
}
Q1 = ("Question about: {lab}.\nDefinition: {d}.\nAccording to the report, is this finding present in this knee?\n"
      "A = present (definite or probable)\nB = absent, or the structure is described as normal/intact\n"
      "C = uncertain, possible, or cannot be excluded\nD = not mentioned at all\nAnswer with a single letter.")
Q2 = ("How severe is it according to the report?\n1 = low (trace, minimal, small, mild, slight, grade I-II, sprain, "
      "low-grade partial, superficial)\n2 = moderate (moderate, grade III, high-grade partial)\n"
      "3 = high (large, marked, severe, complete, full-thickness, grade IV, displaced)\nAnswer with a single digit.")
QS = "Quote the exact words from the report (at most 12 words, original language) that support your answer."
W_STATE = {'pB': 0.0, 'pD': 0.15, 'pC': 0.45}; W_GRADE = (0.60, 0.85, 1.00)   # PRE-REGISTERED, do not tune on gold58

def norm_report(t):
    t = unicodedata.normalize('NFKC', str(t)).replace('µ', 'μ')  # Greek reports use the micro sign for mu
    t = re.sub(r'[ \t]+', ' ', t)
    return t[:3500]

# ---------------- model ----------------
from transformers import AutoTokenizer, AutoModelForCausalLM
OFFLINE = bool(a.tiny_test) or os.environ.get('HF_HUB_OFFLINE', '0') == '1'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct' if a.tiny_test else a.model, local_files_only=OFFLINE)
if a.tiny_test:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
    model = Qwen2ForCausalLM(cfg).eval()
else:
    kw = {}
    if a.load4bit:
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw['device_map'] = {'': 0}
    dt = torch.float16 if a.device.startswith('cuda') else torch.float32   # V100: no bf16
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, local_files_only=OFFLINE, **kw)
    if not a.load4bit: model = model.to(a.device)
    model.eval()
dev = next(model.parameters()).device
def one_tok(s):
    ids = tok.encode(s, add_special_tokens=False); assert len(ids) == 1, (s, ids); return ids[0]
T_STATE = [one_tok(x) for x in 'ABCD']; T_GRADE = [one_tok(x) for x in '123']

def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

@torch.no_grad()
def fwd(ids, cache):
    x = torch.tensor([ids], device=dev)
    out = model(input_ids=x, past_key_values=cache, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values

def lcp(seqs):
    n = min(len(s) for s in seqs); i = 0
    while i < n and all(s[i] == seqs[0][i] for s in seqs): i += 1
    return i

@torch.no_grad()
def extract(report):
    user0 = 'REPORT:\n"""\n' + report + '\n"""\n\n'
    seq1 = {l: chat([{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': user0 + Q1.format(lab=l, d=DEFS[l])}]) for l in L}
    k = lcp(list(seq1.values())) - 1                      # shared prefix (report + system), keep >=1 token per suffix
    from transformers import DynamicCache
    _, pref = fwd(seq1[L[0]][:k], DynamicCache())
    res = {}
    for l in L:
        s1 = seq1[l]
        lg, cache = fwd(s1[k:], copy.deepcopy(pref))
        p = torch.softmax(lg[T_STATE], 0).cpu().numpy()
        msgs2 = [{'role': 'system', 'content': SYSTEM},
                 {'role': 'user', 'content': user0 + Q1.format(lab=l, d=DEFS[l])},
                 {'role': 'assistant', 'content': 'A'}, {'role': 'user', 'content': Q2}]
        s2 = chat(msgs2)
        if s2[:len(s1)] == s1:
            lg2, _ = fwd(s2[len(s1):], cache)
        else:                                             # template did not extend cleanly: recompute from the prefix
            lg2, _ = fwd(s2[k:], copy.deepcopy(pref))
        g = torch.softmax(lg2[T_GRADE], 0).cpu().numpy()
        pA, pB, pC, pD = [float(x) for x in p]
        score = W_STATE['pB'] * pB + W_STATE['pD'] * pD + W_STATE['pC'] * pC + pA * float(np.dot(W_GRADE, g))
        r = {l: score, l + '__pA': pA, l + '__pB': pB, l + '__pC': pC, l + '__pD': pD,
             l + '__g1': float(g[0]), l + '__g2': float(g[1]), l + '__g3': float(g[2])}
        if a.spans and pA + pC > .5:
            msgs3 = msgs2[:2] + [{'role': 'assistant', 'content': 'ABCD'[int(np.argmax(p))]}, {'role': 'user', 'content': QS}]
            ids3 = torch.tensor([chat(msgs3)], device=dev)
            gen = model.generate(ids3, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
            q = tok.decode(gen[0, ids3.shape[1]:], skip_special_tokens=True).strip().strip('"').strip()
            nq = re.sub(r'\s+', ' ', q.lower()); nr = re.sub(r'\s+', ' ', report.lower())
            r[l + '__span'] = q[:200]; r[l + '__span_ok'] = int(len(nq) > 3 and nq in nr)
        res.update(r)
    return res

# ---------------- data ----------------
tr = pd.read_csv(f'{RT}/train.csv', dtype={ID: str})
if a.ids == 'gold74':
    want = pd.read_csv(f'{RT}/gold74_eval_set.csv', dtype={ID: str})[ID].tolist()
elif a.ids == 'all':
    want = tr[ID].tolist()
else:
    want = pd.read_csv(a.ids, dtype={ID: str})[ID].tolist()
if a.limit: want = want[:a.limit]
rep = tr.drop_duplicates(ID).set_index(ID).loc[want, 'Report']
os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
rows = []; t0 = time.time()
for i, (u, t) in enumerate(rep.items()):
    rows.append({ID: u, **extract(norm_report(t))})
    if i % 10 == 0 or i == len(rep) - 1:
        print(f'{i + 1}/{len(rep)} {time.time() - t0:.0f}s ({(time.time() - t0) / (i + 1):.2f} s/report)', flush=True)
out = pd.DataFrame(rows); out.to_csv(os.path.expanduser(a.out), index=False)
print('wrote', a.out, out.shape, f'{(time.time() - t0) / max(1, len(rep)):.2f} s/report')

# ---------------- gate (pre-registered bars; see jump2/labels.md) ----------------
if a.ids == 'gold74':
    from sklearn.metrics import roc_auc_score
    from scipy.special import logit, expit
    S = out.set_index(ID)[L].astype(float)
    gold = tr[tr['ACL'].notna()].set_index(ID)[L].astype(int)
    g74 = pd.read_csv(f'{RT}/gold74_eval_set.csv', dtype={ID: str}).set_index(ID)
    ex16 = g74[g74.source != 'original_gold58'][L].astype(float)
    def rd(p): return pd.read_csv(p, dtype={ID: str}).drop_duplicates(ID).set_index(ID)[L].astype(float)
    v4 = rd(f'{RT}/llm_labels_v4_blend.csv'); mix = rd(f'{RT}/labels_mix50_logit.csv'); wd = rd(f'{RT}/teacher_train_probs.csv')
    g = [u for u in gold.index if u in S.index]
    if len(g) < (10 if a.tiny_test else 50):
        print('gold coverage', len(g), '(partial run: gate skipped)'); sys.exit(0)
    G = gold.loc[g]
    def mac(D): return float(np.mean([roc_auc_score(G[l], D.loc[g, l]) for l in L if 0 < G[l].sum() < len(g)]))
    def lm(x, ref):
        lx = logit(np.clip(x, 1e-4, 1 - 1e-4)); lr = logit(np.clip(ref, 1e-4, 1 - 1e-4))
        return expit((lx - lx.mean()) / lx.std() * lr.std() + lr.mean())
    R = lambda D: D.loc[g, L].rank(pct=True)
    cand = {'KX text-only': S, 'rank-mean(KX, v4)': (R(S) + R(v4)) / 2,
            'mix(KX, wd) logit .5': pd.DataFrame({l: .5 * S.loc[g, l].values + .5 * lm(wd.loc[g, l].values, S.loc[g, l].values) for l in L}, index=g),
            'steven_v4': v4, 'mix50_logit': mix}
    rng = np.random.default_rng(0); B = rng.integers(0, len(g), size=(1000, len(g)))
    def boot(D):
        Dv = D.loc[g, L].values; Gv = G.values; res = []
        for ix in B:
            res.append(np.mean([roc_auc_score(Gv[ix, j], Dv[ix, j]) for j in range(12) if 0 < Gv[ix, j].sum() < len(ix)]))
        return np.array(res)
    bv4, bmix = boot(v4), boot(mix)
    for k, D in cand.items():
        bb = boot(D)
        e16 = [roc_auc_score(ex16[l].dropna(), D.reindex(ex16.index)[l][ex16[l].notna()]) for l in L
               if ex16[l].nunique() == 2] if k in ('KX text-only', 'steven_v4', 'mix50_logit') and ex16.index.isin(D.index).all() else []
        print(f'{k:24s} gold58 {mac(D):.4f} | P>v4 {np.mean(bb > bv4):.2f} | P>mix50 {np.mean(bb > bmix):.2f}'
              + (f' | ex16 {np.mean(e16):.4f}' if e16 else ''))
    if a.spans:
        ok = out[[c for c in out.columns if c.endswith('__span_ok')]].values.ravel(); ok = ok[~pd.isna(ok)]
        print(f'evidence spans verbatim: {ok.mean():.3f} of {len(ok)}')
    print('BARS: G1 ex16>=.975 | G2 KX>=.905 & P>v4>=.95 | G3 rank-mean>=.901 & P>v4>=.90 | G4 mix(KX,wd)>=.934 & P>mix50>=.90')
