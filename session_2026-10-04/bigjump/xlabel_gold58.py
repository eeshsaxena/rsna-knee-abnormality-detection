# CPU-only diagnostic: on gold58, how much does a sibling label's prediction carry for the weak labels?
import pandas as pd, numpy as np, sys, io
from sklearn.metrics import roc_auc_score
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
P={"proxy":pd.read_csv(S+"/teacher_v2_out/raptor_proxy_v5v8.csv").set_index("StudyInstanceUID")[LAB],
   "v4":pd.read_csv(S+"/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB]}
print("gold58 positives:", g.sum().astype(int).to_dict())
print("gold58 phi (selected):")
for a,b in [("Synovitis","Effusion"),("Synovitis","Baker's"),("Lateral OA","Medial OA"),("Lateral OA","PF OA"),("PF OA","Medial OA"),("Contusion","ACL"),("Contusion","Fracture"),("Fracture","Contusion"),("Lateral Meniscus","Lateral OA")]:
    print(f"  {a}~{b}: {np.corrcoef(g[a],g[b])[0,1]:+.2f}")
for name,p in P.items():
    p=p.reindex(g.index).rank(pct=True)
    print(f"\n[{name}] own-label AUC and sibling-only AUC on gold58:")
    for tgt,sibs in [("Synovitis",["Effusion","Baker's"]),("Lateral OA",["Medial OA","PF OA","Lateral Meniscus"]),("PF OA",["Medial OA","Lateral OA"]),("Contusion",["ACL","Fracture"]),("Fracture",["Contusion"]),("Lateral Meniscus",["Lateral OA","ACL"])]:
        own=roc_auc_score(g[tgt],p[tgt]); s=" ".join(f"{x[:6]}={roc_auc_score(g[tgt],p[x]):.3f}" for x in sibs)
        mix=" ".join(f"own+.3{x[:4]}={roc_auc_score(g[tgt],p[tgt]+0.3*p[x]):.3f}" for x in sibs)
        print(f"  {tgt:16s} own={own:.3f} | {s} | {mix}")
