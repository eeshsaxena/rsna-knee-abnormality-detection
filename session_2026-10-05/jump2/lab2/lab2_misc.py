import json, pandas as pd
R=json.load(open("/home/training/jump2/label_eval.json"))
for k in ["mix50_logit","mixSA_logit","gated75_logit","jev_hybrid","steven_v2","CONS_llm4","pilkwang_v2","teacher_wd","T_raptor_proxy","steven_full","jev_final2"]:
    r=R[k]
    print(k.ljust(16), "g58 %.4f" % r["gold58_macro"], "dV4", r.get("d_vs_steven_v4"), "d90", r.get("d90_steven_v4"), "P", r.get("P_gt_steven_v4"), "| dMix", r.get("d_vs_mix50_logit"), "d90", r.get("d90_mix50_logit"), "ex16", round(r.get("ex16_macro") or 0, 4))
tr=pd.read_csv("/home/training/rsna_train/train.csv"); rep=tr.Report.fillna("").astype(str)
low=rep.str.lower()
knee=r"knee|rodilla|diz|koljen|γόνατ|γονατ|knie|genou|коля|menisc|menisk|μηνίσκ|менискус|menisküs|μηνισκ"
print("reports total", len(rep), "| empty/<80 chars", (rep.str.len()<80).sum(), "| no knee/menisc keyword", (~low.str.contains(knee)).sum(), "| bilateral note", rep.str.contains("BILATERAL NOTE").sum())
nk=tr[~low.str.contains(knee)]
for t in nk.Report.astype(str).head(8): print("  NONKNEE?", t[:170].replace("\n"," "))
