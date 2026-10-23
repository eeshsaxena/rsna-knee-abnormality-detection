cd /c/Users/eeshs/Downloads/rsna-knee
sleep 60
for i in $(seq 1 90); do
  s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-ens-f1 2>/dev/null | tail -1)
  case "$s" in *COMPLETE*) echo "[$i] COMPLETE"; break;; *ERROR*|*CANCEL*) echo "[$i] $s"; break;; *) sleep 20;; esac
done
rm -rf _f1out; mkdir -p _f1out; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-ens-f1 _f1out 2>&1 | tail -1
grep -oE '"data":"[^"]*"' _f1out/*.log 2>/dev/null | sed 's/"data":"//;s/"$//' | grep -iE "frozen heads|ft gold|heads|error|traceback|finetune_f1" | head -4
rows=$(tail -n +2 _f1out/submission.csv 2>/dev/null | wc -l); echo "rows=$rows"
if [ "$rows" -ge 3 ] && head -1 _f1out/submission.csv | grep -q "StudyInstanceUID,ACL"; then
  python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-ens-f1 1 "Frozen+fold1 ens K=20 (gold58 0.8817, target >0.906)" 2>&1 | tail -1
else echo "NOT valid, not submitting"; fi
