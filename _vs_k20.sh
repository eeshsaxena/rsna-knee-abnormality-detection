cd /c/Users/eeshs/Downloads/rsna-knee
for i in $(seq 1 90); do
  s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-ens-ft 2>/dev/null | tail -1)
  case "$s" in *COMPLETE*) echo "[$i] COMPLETE"; break;; *ERROR*|*CANCEL*) echo "[$i] $s"; break;; *) sleep 20;; esac
done
rm -rf _ensft20; mkdir -p _ensft20; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-ens-ft _ensft20 2>&1 | tail -1
rows=$(tail -n +2 _ensft20/submission.csv 2>/dev/null | wc -l); echo "rows=$rows"
if [ "$rows" -ge 3 ] && head -1 _ensft20/submission.csv | grep -q "StudyInstanceUID,ACL"; then
  python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-ens-ft 3 "Frozen+FineTuned ens K=20 (denser; K16 was 0.903)" 2>&1 | tail -1
else echo "NOT valid (kernel still running or failed)"; fi
