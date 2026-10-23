#!/bin/bash
# box: fetch the two public weight files needed for the px-cache test (anchor V-Net + side classifier), with 429 backoff
export PATH=$HOME/kagvenv/bin:$PATH
mkdir -p ~/r7port/weights && cd ~/r7port/weights
get() { ds=$1; f=$2; for t in 1 2 3 4 5 6 7 8 9 10; do kaggle datasets download $ds -f $f -p . >/dev/null 2>&1; [ -s "$f" ] && { echo "OK $f"; return; }; [ -s "$f.zip" ] && { unzip -o -q "$f.zip" && rm -f "$f.zip"; echo "OK $f (zip)"; return; }; echo "retry $t $f"; sleep $((t*20)); done; echo "FAIL $f"; }
get heliosli/rsna-orthofoundation-fold0-epoch7-20261004 anatomy.pt
get heliosli/rsna-knee-compact-r7cnxfsss side_cls.pt
sha256sum *.pt
echo DONE_W
