cd /c/Users/eeshs/Downloads/rsna-knee; sleep 40
for i in $(seq 1 90); do s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-ens-f1 2>/dev/null|tail -1); case "$s" in *COMPLETE*) echo "[$i] COMPLETE"; break;; *ERROR*|*CANCEL*) echo "[$i] $s"; break;; *) sleep 20;; esac; done
rm -rf _f1b; mkdir -p _f1b; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-ens-f1 _f1b 2>&1|tail -1
rows=$(tail -n +2 _f1b/submission.csv 2>/dev/null|wc -l); echo "rows=$rows"
if [ "$rows" -ge 3 ]; then python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-ens-f1 2 "frozen(0.4)+fold1(0.6) K=18 (K20 timed out); gold58 0.8817" 2>&1|tail -1; else echo "NOT valid"; fi
