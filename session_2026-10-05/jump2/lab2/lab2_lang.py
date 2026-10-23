import pandas as pd, numpy as np
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID="StudyInstanceUID"; RT="/home/training/rsna_train"; LI="/home/training/jump2/labels_in"
r=pd.read_csv(f"{RT}/_reports_lang.csv").set_index(ID); ids=r.index
def rd(p): return pd.read_csv(p).drop_duplicates(ID).set_index(ID).reindex(ids)
v4=rd(f"{RT}/llm_labels_v4_blend.csv")[L]; sf=rd(f"{LI}/steven_full.csv")[L]; pk=rd(f"{RT}/report_labels_v2.csv")
lx=rd(f"{LI}/lixin_gpt56_conf.csv")[L]; jv=rd(f"{LI}/jev_final2_scores.csv")[L]
pkv=pd.DataFrame({l: pk[l+"__verdict"].fillna("UNK").values for l in L}, index=ids)
votes=(v4>.5).astype(int)+(sf>.5).astype(int)+(pkv=="YES").astype(int)+(lx>=.5).astype(int)+(jv>.5).astype(int)
cont=((votes>0)&(votes<5))
print("contested-cell rate (1-4 of 5 readers say present) by language, all 4407 studies")
df=pd.DataFrame({"lang":r.lang, "cont":cont.mean(axis=1), "unk":(pkv=="UNK").mean(axis=1), "n":1, "len":r.Report.astype(str).str.len()})
print(df.groupby("lang").agg(n=("n","sum"), contested=("cont","mean"), pilk_UNK=("unk","mean"), median_chars=("len","median")).round(3).sort_values("contested").to_string())
print("contested rate by label:", cont.mean().round(3).to_dict())
rep=r.Report.astype(str)
print("reports with U+00B5 micro sign:", rep.str.contains("µ").sum(), "| greek reports:", (r.lang=="el").sum())
sh=rep.str.lower().str.contains(r"shoulder|subscapular|supraspinat|рамо|раменн|hip joint|ankle|tobillo|cadera|omuz|ayak bile", regex=True)
print("possible non-knee anatomy mentions:", sh.sum())
for t in rep[sh].head(5): print("   ", t[:140].replace("\n"," "))
