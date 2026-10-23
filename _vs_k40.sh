cd /c/Users/eeshs/Downloads/rsna-knee
for i in $(seq 1 40); do
  s=$(python kaggle_do.py kstatus eeshsaxena/rsna-knee-infer-radimagenet 2>/dev/null | tail -1)
  echo "[$i] $s"; case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 20
done
rm -rf _radimg3; mkdir -p _radimg3; python kaggle_do.py koutput eeshsaxena/rsna-knee-infer-radimagenet _radimg3 2>&1 | tail -1
rows=$(tail -n +2 _radimg3/submission.csv 2>/dev/null | wc -l)
echo "rows=$rows"; grep -oE '"data":"WROTE[^"]*"' _radimg3/*.log 2>/dev/null | tail -1
if [ "$rows" -ge 3 ] && head -1 _radimg3/submission.csv | grep -q "StudyInstanceUID,ACL"; then
  python kaggle_do.py submit rsna-knee-abnormality-detection eeshsaxena/rsna-knee-infer-radimagenet 3 "RadImageNet K=40 denser inference (K24 was 0.903)" 2>&1 | tail -1
else echo "NOT valid"; fi
