# CPU-only: per-report-language (site proxy) label prevalence vs image-teacher prevalence; gold58 per-language.
import pandas as pd, numpy as np, sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
D="C:/Users/eeshs/Downloads/rsna-knee/_data"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
lang=pd.read_csv(D+"/_reports_lang.csv",usecols=["StudyInstanceUID","lang"]).set_index("StudyInstanceUID")["lang"]
v4=pd.read_csv(S+"/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB]
t5=pd.read_csv(S+"/teacher_v2_out/teacher_v2_maxspan_v5_shard0.csv").set_index("StudyInstanceUID")[LAB]
t8=pd.read_csv(S+"/teacher_v2_out/teacher_v2_native384_v8_shard0.csv").set_index("StudyInstanceUID")[LAB]
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
print("lang counts (all train):"); print(lang.value_counts().to_string())
print("gold58 langs:", lang.reindex(g.index).value_counts().to_dict())
ids=v4.index.intersection(t5.index).intersection(lang.index).difference(g.index)
L=lang.reindex(ids); top=[l for l,c in L.value_counts().items() if c>=150]
def tab(df,name):
    m=df.reindex(ids).groupby(L).mean().loc[top]
    print(f"\n{name} mean per language (rows) x label (cols, abbreviated):")
    print(m.round(3).rename(columns=lambda c:c[:6]).to_string())
    return m
mv=tab(v4,"steven_v4 label"); mt=tab(t5,"teacher maxspan_v5 prob")
# rank-based: mean within-label percentile rank of teacher per language (prevalence-free of calibration)
rt=t5.reindex(ids).rank(pct=True); rv=v4.reindex(ids).rank(pct=True)
d=(rv.groupby(L).mean()-rt.groupby(L).mean()).loc[top]
print("\nmean pct-rank(label) - mean pct-rank(teacher) per language (positive = reports call MORE than images suggest):")
print(d.round(3).rename(columns=lambda c:c[:6]).to_string())
print("\nspread (max-min over languages) per label of label-minus-teacher rank gap:")
print((d.max()-d.min()).round(3).rename(lambda c:c[:8]).to_string())
