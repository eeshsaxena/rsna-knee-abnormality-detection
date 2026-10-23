cd /c/Users/eeshs/Downloads/rsna-knee
for i in $(seq 1 80); do
  s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-ens-ft 2>/dev/null | tail -1)
  case "$s" in *COMPLETE*) echo "[$i] COMPLETE"; break;; *ERROR*|*CANCEL*) echo "[$i] $s"; break;; *) sleep 30;; esac
done
rm -rf _ensft4; mkdir -p _ensft4; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-ens-ft _ensft4 2>&1 | tail -1
rows=$(tail -n +2 _ensft4/submission.csv 2>/dev/null | wc -l); echo "rows=$rows"
if [ "$rows" -ge 3 ] && head -1 _ensft4/submission.csv | grep -q "StudyInstanceUID,ACL"; then
  echo "=== submitting ==="; python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-ens-ft 2 "Frozen+FineTuned ens K=16; gold58 0.8759" 2>&1 | tail -1
else echo "NOT valid (rows=$rows) -- kernel may have errored; not submitting"; fi
