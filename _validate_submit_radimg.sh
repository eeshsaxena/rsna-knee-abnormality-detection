cd /c/Users/eeshs/Downloads/rsna-knee
for i in $(seq 1 40); do
  s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-radimagenet 2>/dev/null | tail -1)
  echo "[$i] $s"; case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 20
done
rm -rf _radimg2; mkdir -p _radimg2; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-radimagenet _radimg2 2>&1 | tail -1
echo "=== log ==="; grep -oE '"data":"[^"]*"' _radimg2/*.log 2>/dev/null | sed 's/"data":"//;s/"$//' | grep -viE "warning|deprecat|frozen|breakpoint|Debug|validation" | tail -8
rows=$(tail -n +2 _radimg2/submission.csv 2>/dev/null | wc -l)
echo "=== submission rows: $rows ==="
head -2 _radimg2/submission.csv 2>/dev/null
# auto-submit if valid (>=3 rows, header present)
if [ "$rows" -ge 3 ] && head -1 _radimg2/submission.csv | grep -q "StudyInstanceUID,ACL"; then
  echo "=== VALID -> submitting v2 (K=24) ==="
  python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-radimagenet 2 "RadImageNet K=24 dense (match train ~45/series) gold58 0.8718" 2>&1 | tail -2
else
  echo "=== NOT valid, NOT submitting ==="
fi
