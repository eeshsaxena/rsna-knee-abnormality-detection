cd /c/Users/eeshs/Downloads/rsna-knee
for i in $(seq 1 45); do
  s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-ens-ft 2>/dev/null | tail -1)
  echo "[$i] $s"; case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 20
done
rm -rf _ensft2; mkdir -p _ensft2; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-ens-ft _ensft2 2>&1 | tail -1
rows=$(tail -n +2 _ensft2/submission.csv 2>/dev/null | wc -l); echo "rows=$rows"
if [ "$rows" -ge 3 ] && head -1 _ensft2/submission.csv | grep -q "StudyInstanceUID,ACL"; then
  python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-ens-ft 2 "Frozen+FineTuned ens K=16 (K40 timed out); gold58 0.8759" 2>&1 | tail -1
else echo "NOT valid, not submitting"; fi
